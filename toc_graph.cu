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

// namespace GraphShatter{

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
    RegionInstance *edges,
    IndexSpace<1> edgesSpace,
    Memory deviceMemory,
    RegionInstance *insBuffer,
    RegionInstance *bufferInputIds //edges'
)
{
    // In refers to u in directed edge (u,v)
    AffineAccessor<size_t,1>inAcc(*edges, IN_VERTEX);
    AffineAccessor<size_t,1>outAcc(*edges, OUT_VERTEX);

    // Make sure we can use thrust to process the values as a single buffer
    assert(inAcc.is_dense_arbitrary(edgesSpace.bounds));
    size_t* inPtr = inAcc.ptr(edgesSpace.bounds.lo);

    // From this buffer we will generate the ins set
    // TODO see if there's lost performance on using these device_vectors. I would guess not for this case
    thrust::device_vector<size_t> analysisBuffer(inPtr, inPtr + edgesSpace.volume());
    thrust::device_vector<size_t> inputIndices(edgesSpace.volume());
    thrust::sequence(inputIndices.begin(), inputIndices.end());

    auto inputAndIndexBegin = thrust::make_zip_iterator(analysisBuffer.begin(), inputIndices.begin());
    auto inputAndIndexEnd = thrust::make_zip_iterator(analysisBuffer.end(), inputIndices.end());

    // Sort by original values (id of input vertices)
    thrust::sort(inputAndIndexBegin, inputAndIndexEnd);
    // Save this sorted buffer for eventually generating ins itself
    thrust::device_vector<size_t> insAnalysis(analysisBuffer.begin(), analysisBuffer.end());

    // Set values to define the points
    findInputEdges<<<256, (edgesSpace.volume() + 255) / 256>>>(analysisBuffer.data().get(), edgesSpace.volume());

    // External sum to get buffer vertex ids
    thrust::inclusive_scan(analysisBuffer.begin(), analysisBuffer.end(), analysisBuffer.begin());

    // Sort back into original order
    inputAndIndexBegin = thrust::make_zip_iterator(inputIndices.begin(), analysisBuffer.begin());
    inputAndIndexEnd = thrust::make_zip_iterator(inputIndices.end(), analysisBuffer.end());
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
    //Set out our ins buffer that our GPU will use for ever iteration of the algorithm
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

    //Find the output vectors
    // TODO I shouldn't actually have to do this sort given the input
    thrust::sort(analysisBuffer.begin(), analysisBuffer.end());
    new_end = thrust::unique(analysisBuffer.begin(), analysisBuffer.end());

    size_t ons_size = new_end - analysisBuffer.begin();
    IndexSpace<1> onsSpace(Rect<1>(0,ons_size - 1));

    //Build the INS set and moves in data
    fieldSizes.clear();
    fieldSizes[OUT_VERTEX] = sizeof(size_t);
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
    
    insBufferReadyEvent.wait();
    cudaDeviceSynchronize();
    // It seems device_vectors automatically deallocate memory
    // analysisBuffer.clear();
}

// From the ons set, each thread gets its own vertex
__global__ void iterationKernel(
    IndexSpace<1> onsSpace,
    AffineAccessor<size_t, 1> onsAcc,
    IndexSpace<1> edgesSpace,
    AffineAccessor<size_t, 1> inputAcc
)
{
    size_t did = threadIdx.x;
    size_t bid = blockIdx.x * blockDim.x;
    size_t tid = bid + did;

    // Presumes that the ons set is congruous by ID
    // __shared__ vertex newVertexValues[256];

    // // Each thread initializes a node
    // if(onsSpace.contains(tid)){
    //     newVertexValues[did] = {0};
    // }

    // // Each thread statically handles a vertex
    // for(Point<1> i = edgesSpace.bounds.lo; i <= edgesSpace.bounds.hi; i.x += blockDim.x){
    //     // vertex input = inputAcc[i]
    // }

}

__host__ void runIteration(
    RegionInstance edges,
    RegionInstance vertices,
    RegionInstance insBuffer,
    RegionInstance ons
)
{
    IndexSpace<1> edgesSpace = edges.get_indexspace<1>();
    IndexSpace<1> insSpace = insBuffer.get_indexspace<1>();
    assert(edgesSpace.dense());

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
    int numBlocks = 256;
    assert(insSpace.dense());
    int numThreadsPerBlock = (insSpace.volume() + numBlocks - 1) / numBlocks;

    loadInsVerticesKernel<<<numBlocks, numThreadsPerBlock>>>(
        verticesAcc,
        insAcc,
        insBufferAcc,
        insSpace.bounds
    );

    cudaDeviceSynchronize();
}

// }