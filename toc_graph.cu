#include "graph_shatter_types.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

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

__host__ void generateNeighborSets(
    RegionInstance *ins,
    RegionInstance *ons,
    RegionInstance *edges,
    IndexSpace<1> edgesSpace,
    Memory deviceMemory,
    RegionInstance *insBuffer
)
{
    // In refers to u in directed edge (u,v)
    AffineAccessor<size_t,1>inAcc(*edges, IN_VERTEX);
    AffineAccessor<size_t,1>outAcc(*edges, OUT_VERTEX);

    // Make sure we can use thrust to process the values as a single buffer
    assert(inAcc.is_dense_arbitrary(edgesSpace.bounds));
    size_t* inPtr = inAcc.ptr(edgesSpace.bounds.lo);

    // From this buffer we will generate the ins set
    // log_app.print() << edgesSpace.volume();
    thrust::device_vector<size_t> analysisBuffer(inPtr, inPtr + edgesSpace.volume());

    // Get unique values
    // TODO in practice we will not need to sort for either ONS or INS
    thrust::sort(analysisBuffer.begin(), analysisBuffer.end());
    auto new_end = thrust::unique(analysisBuffer.begin(), analysisBuffer.end());
    
    size_t ins_size = new_end - analysisBuffer.begin();

    IndexSpace<1> insSpace(Rect<1>(0,ins_size - 1));

    //Build the INS set and moves in data
    std::map<FieldID, size_t> fieldSizes;
    fieldSizes[IN_VERTEX] = sizeof(size_t);
    // Just waiting on event might lower throughput but I'm trying to reuse the buffer
    // TODO see if there's any way to connect Realm events with CUDA streams
    RegionInstance::create_instance(
        *ins,
        deviceMemory,
        insSpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    ).wait();

    //Copy data into our new ins regionInstance
    AffineAccessor<size_t,1>insAcc(*ins, IN_VERTEX);
    thrust::device_ptr<size_t> insPtr(insAcc.ptr(insSpace.bounds.lo));
    thrust::copy(analysisBuffer.begin(), analysisBuffer.begin() + ins_size, insPtr);

    //Begin ons analysis
    assert(outAcc.is_dense_arbitrary(edgesSpace.bounds));
    thrust::device_ptr<size_t> outPtr(outAcc.ptr(edgesSpace.bounds.lo));
    thrust::copy(outPtr, outPtr + edgesSpace.volume(), analysisBuffer.begin());

    //Find the output vectors
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

    //Set out our ons buffer that our GPU will use for ever iteration of the algorithm
    fieldSizes.clear();
    fieldSizes[IN_VERTEX] = sizeof(size_t);
    RegionInstance::create_instance(
        *insBuffer,
        deviceMemory,
        onsSpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    ).wait();
    
    cudaDeviceSynchronize();
    analysisBuffer.clear();
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
    __shared__ vertex newVertexValues[256];

    // Each thread initializes a node
    if(onsSpace.contains(tid)){
        newVertexValues[did] = {0};
    }

    // Each thread statically handles a vertex
    for(Point<1> i = edgesSpace.bounds.lo; i <= edgesSpace.bounds.hi; i.x += blockDim.x){
        // vertex input = inputAcc[i]
    }

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

// }