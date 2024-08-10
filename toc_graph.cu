#include "graph_shatter_types.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/zip_function.h>
#include <thrust/scan.h>

using namespace Realm;

// This seems important
#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#endif

extern Logger log_app;

// TODO Make this work for all index spaces
__device__ Point<1> indexToPoint(IndexSpace<1>& is, size_t point){

    return Point<1>(point);
}

// Doubles every element
__global__ void doubleKernel(Rect<1> is, AffineAccessor<int, 1> linear_accessor){
    Point<1> p = blockIdx.x * blockDim.x + threadIdx.x;
    if(is.contains(p)){
        linear_accessor[p] += linear_accessor[p];
        printf("%d \n", linear_accessor[p]);
    }
}


__host__ void examplelauncher(Rect<1> is, AffineAccessor<int, 1> linear_accessor){
      // Run the kernel
    size_t threadsPerBlock = 32;
    size_t numBlocks = (is.volume() + threadsPerBlock - 1) / threadsPerBlock;

    doubleKernel<<<numBlocks, threadsPerBlock>>>(Rect<1>(0,5), linear_accessor);
    cudaDeviceSynchronize();
}

__global__ void findInputEdges(size_t* buffer, size_t size){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < size){
        if(tid != 0 && buffer[tid] != buffer[tid - 1]){
            buffer[tid] = 1;
        }
        else{
            buffer[tid] = 0;
        }
    }
}

__host__ void generateNeighborSets(
    RegionInstance *ins,
    RegionInstance *ons,
    RegionInstance edges,
    IndexSpace<1> edgesSpace,
    Memory deviceMemory,
    RegionInstance *insBuffer,
    RegionInstance *bufferInputIds //edges'
)
{
    // In refers to u in directed edge (u,v)
    AffineAccessor<size_t,1>inAcc(edges, IN_VERTEX);
    AffineAccessor<size_t,1>outAcc(edges, OUT_VERTEX);

    // Make sure we can use thrust to process the values as a single buffer
    assert(inAcc.is_dense_arbitrary(edgesSpace.bounds));
    size_t* inPtr = inAcc.ptr(edgesSpace.bounds.lo);

    // From this buffer we will generate the ins set
    // TODO see if there's lost performance on using these device_vectors. I would guess not for this case
    thrust::device_vector<size_t> analysisBuffer(inPtr, inPtr + edgesSpace.volume());
    thrust::device_vector<size_t> indicesBuffer(edgesSpace.volume());
    thrust::sequence(indicesBuffer.begin(), indicesBuffer.end());

    auto inputAndIndexBegin = thrust::make_zip_iterator(analysisBuffer.begin(), indicesBuffer.begin());
    auto inputAndIndexEnd = thrust::make_zip_iterator(analysisBuffer.end(), indicesBuffer.end());

    // Sort by original values (id of input vertices)
    thrust::sort(inputAndIndexBegin, inputAndIndexEnd);
    // Save this sorted buffer for eventually generating ins itself
    thrust::device_vector<size_t> insAnalysis(analysisBuffer.begin(), analysisBuffer.end());

    // Set values to define the points
    findInputEdges<<<(edgesSpace.volume() + 255) / 256, 256>>>(analysisBuffer.data().get(), edgesSpace.volume());

    // External sum to get buffer vertex ids
    thrust::inclusive_scan(analysisBuffer.begin(), analysisBuffer.end(), analysisBuffer.begin());

    // Sort back into original order
    inputAndIndexBegin = thrust::make_zip_iterator(indicesBuffer.begin(), analysisBuffer.begin());
    inputAndIndexEnd = thrust::make_zip_iterator(indicesBuffer.end(), analysisBuffer.end());
    thrust::sort(inputAndIndexBegin, inputAndIndexEnd);

    //Create the copy of the input vertices with their buffer ids
    std::map<FieldID, size_t> fieldSizes;
    fieldSizes[IN_VERTEX] = sizeof(size_t);
    // TODO see if there's any way to connect Realm events with CUDA streams
    RegionInstance::create_instance(
        *bufferInputIds,
        deviceMemory,
        edgesSpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    ).wait();

    //Actually populates bufferInputIds with data from our analysis
    AffineAccessor<size_t,1>bufferIdsAcc(*bufferInputIds, IN_VERTEX);
    thrust::device_ptr<size_t> bufferIdsPtr(bufferIdsAcc.ptr(edgesSpace.bounds.lo));
    thrust::copy(analysisBuffer.begin(), analysisBuffer.end(), bufferIdsPtr);
    
    // Takes our saved sorted buffer of the input edges
    auto new_end = thrust::unique(insAnalysis.begin(), insAnalysis.end());
    size_t ins_size = new_end - insAnalysis.begin();

    // Build the INS set and moves in data
    // Just waiting on event might lower throughput but I'm trying to reuse the buffer
    IndexSpace<1> insSpace(Rect<1>(0,ins_size - 1));
    Event insReadyEvent = RegionInstance::create_instance(
        *ins,
        deviceMemory,
        insSpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    );

    fieldSizes.clear();
    fieldSizes[VERTEX_ID] = sizeof(vertex);
    //Set out our ins buffer that our GPU will use for every iteration of the algorithm
    Event insBufferReadyEvent = RegionInstance::create_instance(
        *insBuffer,
        deviceMemory,
        insSpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    );

    insReadyEvent.wait();

    //Copy data into our new ins regionInstance
    AffineAccessor<size_t,1>insAcc(*ins, IN_VERTEX);
    thrust::device_ptr<size_t> insPtr(insAcc.ptr(insSpace.bounds.lo));
    thrust::copy(insAnalysis.begin(), insAnalysis.begin() + ins_size, insPtr);

    //Begin ons analysis
    assert(outAcc.is_dense_arbitrary(edgesSpace.bounds));
    thrust::device_ptr<size_t> outPtr(outAcc.ptr(edgesSpace.bounds.lo));
    thrust::copy(outPtr, outPtr + edgesSpace.volume(), analysisBuffer.begin());
    // Use this buffer to also get inputs of the ons
    // TODO modify this when we scale up to multiple nodes
    thrust::sequence(indicesBuffer.begin(), indicesBuffer.end(), (size_t)edgesSpace.bounds.lo.x);

    //Find the output vectors
    // TODO I shouldn't actually have to do this sort given the input
    // thrust::sort(analysisBuffer.begin(), analysisBuffer.end());
    auto unique_end = thrust::unique_by_key(analysisBuffer.begin(), analysisBuffer.end(), indicesBuffer.begin());

    size_t ons_size = unique_end.first - analysisBuffer.begin();
    IndexSpace<1> onsSpace(Rect<1>(0,ons_size - 1));

    //Build the INS set and moves in data
    fieldSizes.clear();
    fieldSizes[OUT_VERTEX] = sizeof(size_t);
    fieldSizes[START_INDEX] = sizeof(size_t);
    // Just waiting on event might lower throughput but I'm trying to reuse the buffer
    RegionInstance::create_instance(
        *ons,
        deviceMemory,
        onsSpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    ).wait();

    AffineAccessor<size_t,1>onsAcc(*ons, OUT_VERTEX);
    thrust::device_ptr<size_t> onsPtr(onsAcc.ptr(onsSpace.bounds.lo));
    thrust::copy(analysisBuffer.begin(), analysisBuffer.begin() + ons_size, onsPtr);

    AffineAccessor<size_t,1>onsIndexAcc(*ons, START_INDEX);
    thrust::device_ptr<size_t> onsIndexPtr(onsIndexAcc.ptr(onsSpace.bounds.lo));
    thrust::copy(indicesBuffer.begin(), indicesBuffer.begin() + ons_size, onsIndexPtr);
    
    insBufferReadyEvent.wait();
    cudaDeviceSynchronize();
    // It seems device_vectors automatically deallocate memory
    // analysisBuffer.clear();
}


// Toy initialization function that always resets a vertex to void
__device__ vertex initializeVertexBase(vertex old){
    return {0};
}


__device__ __inline__ void computeBase(
    vertex *newVertexValues /* Shared memory holding our output values*/,
    size_t tempIndex,
    size_t edgeNumber,
    AffineAccessor<size_t,1>& biiAcc,
    AffineAccessor<vertex,1>& insBufferAcc
)
{
    // Don't actually use the input for this example
    vertex input = insBufferAcc[biiAcc[edgeNumber]];
    atomicAdd(&newVertexValues[tempIndex].value, 1);
}


__global__ void iterationKernel(
    Rect<1> edgesSpace,
    Rect<1> onsSpace,
    AffineAccessor<vertex,1> insBufferAcc,
    AffineAccessor<vertex,1> verticesAcc,
    AffineAccessor<size_t,1> biiAcc,
    AffineAccessor<size_t,1> onsAcc,
    AffineAccessor<size_t,1> onsIndexAcc,
    AffineAccessor<size_t,1> outputsAcc
)
{
    // Gives us a single index within ons
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blockBase = blockIdx.x * blockDim.x;
    size_t threadCount = blockDim.x;

    // Presumes that the ons set is congruous by ID as specified by original paper
    __shared__ vertex newVertexValues[256];
    // The final thread that corresponds to an actual vertex in ons
    __shared__ size_t finalEdgesIndex;
    __shared__ size_t firstEdgesIndex;

    // This thread corresponds to a single vertex within ons
    if(onsSpace.contains(tid)){
        // This thread is the final one corresponding to a vertex
        if((tid == blockBase + blockDim.x - 1 && tid <= onsSpace.hi) ||
            tid == onsSpace.hi){
            finalEdgesIndex = ((tid == onsSpace.hi) ? (size_t) edgesSpace.hi + 1: onsIndexAcc[tid + 1]);
        }
        // This is the first thread in the block
        if(tid == blockBase){
            firstEdgesIndex = onsIndexAcc[tid];
        }

        newVertexValues[tid - blockBase] = initializeVertexBase(verticesAcc[onsAcc[tid]]);
    }
    __syncthreads();

    // Threads cooperatively run computations on edges
    // TODO build out full cooperative scheduling implementation
    for(size_t i = firstEdgesIndex + threadIdx.x; i < finalEdgesIndex; i += threadCount){
        computeBase(
            newVertexValues,
            outputsAcc[i] - outputsAcc[firstEdgesIndex] /* TODO most likely not the best approach to figure out output vertex*/,
            i,
            biiAcc,
            insBufferAcc
        );
    }
    __syncthreads();

    //Write results back to zero copy memory
    if(onsSpace.contains(tid)){
        verticesAcc[onsAcc[tid]] = newVertexValues[tid - blockBase];
    }

}

// Presumes ownership over every  
__host__ void runIteration(
    IndexSpace<1> edgesSpace /* Represents the edges with outputs represented by ons*/,
    IndexSpace<1> onsSpace,
    AffineAccessor<vertex,1> insBufferAcc /* Read inputs to edges here*/,
    AffineAccessor<vertex,1> verticesAcc /* Write final results here*/,
    AffineAccessor<size_t,1> biiAcc /* index of the input edge to be read in insBufferAcc*/,
    AffineAccessor<size_t,1> onsAcc /* Used to get the outputs for writing back to zero_copy memory*/,
    AffineAccessor<size_t,1> onsIndexAcc /* Where to start scan for a given vertex*/,
    AffineAccessor<size_t,1> outputsAcc
)
{
    assert(onsSpace.dense());
    assert(edgesSpace.dense());
    size_t onsCount = onsSpace.volume();

    int threadsPerBlock = 256;
    int numBlocks = (onsCount + threadsPerBlock - 1) / threadsPerBlock;
    // log_app.print() << "onsSpace " << onsSpace;
    // log_app.print() << "edgesSpace " << edgesSpace;
    iterationKernel<<<numBlocks, threadsPerBlock>>>(
        edgesSpace.bounds,
        onsSpace.bounds,
        insBufferAcc,
        verticesAcc,
        biiAcc,
        onsAcc,
        onsIndexAcc,
        outputsAcc
    );

    cudaDeviceSynchronize();
}

__global__ void loadInsVerticesKernel(
    AffineAccessor<vertex,1> verticesAcc,
    AffineAccessor<size_t,1> insAcc,
    AffineAccessor<vertex,1> insBufferAcc,
    Rect<1> insSpace
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(insSpace.contains(tid)){
        insBufferAcc[tid] = verticesAcc[insAcc[tid]];
    }
}

__host__ void loadInsVertices(
    AffineAccessor<vertex,1> verticesAcc,
    AffineAccessor<size_t,1> insAcc,
    AffineAccessor<vertex,1> insBufferAcc,
    IndexSpace<1> insSpace
)
{
    int threadsPerBlock = 256;
    assert(insSpace.dense());
    int numBlocks = (insSpace.volume() + threadsPerBlock - 1) / threadsPerBlock;
    loadInsVerticesKernel<<<numBlocks, threadsPerBlock>>>(
        verticesAcc,
        insAcc,
        insBufferAcc,
        insSpace.bounds
    );

    cudaDeviceSynchronize();
}
