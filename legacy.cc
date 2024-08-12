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