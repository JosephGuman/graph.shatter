struct LDSArgs{
  IndexSpace<1> is;
  RegionInstance data;
};

void examplelauncher(Rect<1> is, AffineAccessor<int, 1> linear_accessor);

static void load_double_store(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p){
  const LDSArgs &gpuArgs = *reinterpret_cast<const LDSArgs *>(args);

  // Create instance in GPU device memory
  Memory gpuMem = Machine::MemoryQuery(Machine::get_machine())
      .has_capacity(gpuArgs.is.volume() * sizeof(int))
      .best_affinity_to(p)
      .first();
  std::map<FieldID, size_t> fieldSizes;
    fieldSizes[1] = sizeof(int);
  RegionInstance deviceInstance;
  Event deviceInstanceE = RegionInstance::create_instance(
    deviceInstance,
    gpuMem,
    gpuArgs.is,
    fieldSizes,
    0,
    ProfilingRequestSet(),
    Event::NO_EVENT
  );

  // Load data into logical instance
  std::vector<CopySrcDstField> src(1);
  std::vector<CopySrcDstField> dst(1);
  src[0].set_field(gpuArgs.data, 1, sizeof(int));
  dst[0].set_field(deviceInstance, 1, sizeof(int));
  gpuArgs.is.copy(src, dst, ProfilingRequestSet(), deviceInstanceE).wait();

  //Launch 
  AffineAccessor<int,1> acc(deviceInstance, 1);
  examplelauncher(Rect<1>(0,5), acc);

  //Load data back into CPU memory
  src[0].set_field(deviceInstance, 1, sizeof(int));
  dst[0].set_field(gpuArgs.data, 1, sizeof(int));
  gpuArgs.is.copy(src, dst, ProfilingRequestSet()).wait();


}


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



namespace GraphShatter{

template<typename T>
Ptable<T>::Ptable(IndexSpace<1> space, Processor p) {
  // Get system memory on the processor
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::SYSTEM_MEM)
      .has_capacity(sizeof(T) * space.volume())
      .best_affinity_to(p);
  
  if(mq.begin() == mq.end()){
      log_app.error() << "Not enough space in proc " << p << " to back Ltable";
  }

  std::map<FieldID, size_t> field_sizes;
  field_sizes[fid] = sizeof(T);
  Event e = RegionInstance::create_instance(
      data,
      *mq.begin(),
      space,
      field_sizes,
      0,
      ProfilingRequestSet()
  );

  // For now just wait
  e.wait();

  accessor = AffineAccessor<T,1>(data, fid);
  // AffineAccessor<T, 1, int> accessor(data, fid);
};

template<typename T>
Ptable<T>::~Ptable(){
  data.destroy();
}

template<typename T>
Event Ptable<T>::fill(T value, Event prior){
  std::vector<CopySrcDstField> dsts(1);
  dsts[0].set_field(data, fid, sizeof(T));
  return data.get_indexspace<1>().fill(
      dsts,
      ProfilingRequestSet(),
      &value,
      sizeof(T),
      prior);
};

template<typename T>
T Ptable<T>::read_concurrent(Point<1> p, Event prior){
  prior.wait();
  return accessor[p];
};

}




namespace GraphShatter{


// The physical instantiation of an Ltable
// Presume the field id is always 1 for now
// Exists on a single processor
// Not thread safe
template<typename T>
class Ptable{
    public:
    RegionInstance data;
    FieldID fid = 1;
    AffineAccessor<T, 1>accessor;


    public:
    Ptable(IndexSpace<1> space, Processor p);
    ~Ptable();

    Event fill(T value, Event prior = Event::NO_EVENT);
    T read_concurrent(Point<1> p, Event prior = Event::NO_EVENT);
};

// Highest level of abstraction for a graph in graph.shatter.
// Does not exist on single proc
// Starting as 1d array with T indices
// Presume Ptable always on p
// Not thread safe
template<typename T>
class Ltable{
    public:
    IndexSpace<1> totalSpace;
    Ptable<T> physicalBacking;
    

    public:
    // The starting processor Ltable's backing is on
    Ltable(int size, Processor p) : 
        totalSpace(Rect<1>(0, size - 1)),
        physicalBacking(totalSpace, p){};

    //Fill every element with a value
    Event fill(T value){
        return physicalBacking.fill(value);
    };

    T read_concurrent(Point<1> p, Event prior = Event::NO_EVENT){
        return physicalBacking.read_concurrent(p, prior);
    };
};


}


void legacy_graph_building(){
  //Get our initial fake vertices
  //TODO get a distributed file system to load in real graph data
  // int vertexCount = 5;
  // int edgeCount = 7;
  // int outEdgeCount = 4;
  // IndexSpace<1> vertexSpace(Rect<1>(0,vertexCount - 1));
  // IndexSpace<1> edgeSpace(Rect<1>(0,edgeCount - 1));
  // IndexSpace<1> boundariesSpace(Rect<1>(0,outEdgeCount - 1));

  // //Get memory to put graph onto zero copy/ best affinity memory for our initial ground truth
  // Machine::MemoryQuery mq(Machine::get_machine());
  // Memory nodeMemory = mq.only_kind(Memory::SYSTEM_MEM)
  //   .has_capacity(sizeof(vertex) * (vertexCount + 2 * sizeof(size_t)))
  //   .best_affinity_to(p).first();

  // //Put our graph on node memory
  // //TODO track effect of memory layout
  // RegionInstance vertices;
  // std::map<FieldID, size_t> fieldSizes;
  // fieldSizes[VERTEX_ID] = sizeof(vertex);
  // Event verticesEvent = RegionInstance::create_instance(
  //     vertices,
  //     nodeMemory,
  //     vertexSpace,
  //     fieldSizes,
  //     0,
  //     ProfilingRequestSet()
  // );
  // fieldSizes.clear();

  // RegionInstance edges;
  // fieldSizes[IN_VERTEX] = sizeof(size_t);
  // fieldSizes[OUT_VERTEX] = sizeof(size_t);
  // Event edgesEvent = RegionInstance::create_instance(
  //   edges,
  //   nodeMemory,
  //   edgeSpace,
  //   fieldSizes,
  //   0,
  //   ProfilingRequestSet()
  // );
  // fieldSizes.clear();

  // RegionInstance edgeBoundaries;
  // fieldSizes[OUT_VERTEX] = sizeof(size_t);
  // Event boundariesEvent = RegionInstance::create_instance(
  //   edgeBoundaries,
  //   nodeMemory,
  //   boundariesSpace,
  //   fieldSizes,
  //   0,
  //   ProfilingRequestSet()
  // );


  // Event graphReadyEvent = Event::merge_events(
  //   verticesEvent,
  //   edgesEvent,
  //   boundariesEvent
  // );

  // graphReadyEvent.wait();
  // loadFakeVertices(vertices);
  // loadFakeEdges(edges, edgeBoundaries);
}

void legacy_ons_analysis(){
  // //Begin ons analysis
  // assert(outAcc.is_dense_arbitrary(edgesSpace.bounds));
  // thrust::device_ptr<size_t> outPtr(outAcc.ptr(edgesSpace.bounds.lo));
  // thrust::copy(outPtr, outPtr + edgesSpace.volume(), analysisBuffer.begin());
  // // Use this buffer to also get inputs of the ons
  // // TODO modify this when we scale up to multiple nodes
  // thrust::sequence(indicesBuffer.begin(), indicesBuffer.end(), (size_t)edgesSpace.bounds.lo.x);

  // //Find the output vectors
  // auto unique_end = thrust::unique_by_key(analysisBuffer.begin(), analysisBuffer.end(), indicesBuffer.begin());

  // size_t ons_size = unique_end.first - analysisBuffer.begin();
  // IndexSpace<1> onsSpace(Rect<1>(0,ons_size - 1));

  // //Build the INS set and moves in data
  // fieldSizes.clear();
  // fieldSizes[OUT_VERTEX] = sizeof(size_t);
  // fieldSizes[START_INDEX] = sizeof(size_t);
  // // Just waiting on event might lower throughput but I'm trying to reuse the buffer
  // RegionInstance::create_instance(
  //     *ons,
  //     deviceMemory,
  //     onsSpace,
  //     fieldSizes,
  //     0,
  //     ProfilingRequestSet()
  // ).wait();

  // AffineAccessor<size_t,1>onsAcc(*ons, OUT_VERTEX);
  // thrust::device_ptr<size_t> onsPtr(onsAcc.ptr(onsSpace.bounds.lo));
  // thrust::copy(analysisBuffer.begin(), analysisBuffer.begin() + ons_size, onsPtr);

  // AffineAccessor<size_t,1>onsIndexAcc(*ons, START_INDEX);
  // thrust::device_ptr<size_t> onsIndexPtr(onsIndexAcc.ptr(onsSpace.bounds.lo));
  // thrust::copy(indicesBuffer.begin(), indicesBuffer.begin() + ons_size, onsIndexPtr);
  
  // insBufferReadyEvent.wait();
  // cudaDeviceSynchronize();
  // It seems device_vectors automatically deallocate memory
  // analysisBuffer.clear();
}