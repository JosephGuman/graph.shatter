#include "graph_shatter_types.h"

using namespace Realm;

Logger log_app("app");

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
  log_app.print() << "yay";

  //Load data back into CPU memory
  src[0].set_field(deviceInstance, 1, sizeof(int));
  dst[0].set_field(gpuArgs.data, 1, sizeof(int));
  gpuArgs.is.copy(src, dst, ProfilingRequestSet()).wait();


}


void generateNeighborSets(
  RegionInstance *ins,
  RegionInstance *ons,
  RegionInstance *edges,
  IndexSpace<1> edgesSpace,
  Memory deviceMemory,
  RegionInstance *insBuffer,
  RegionInstance *bufferInputIds
);

void loadInsVertices(
  AffineAccessor<vertex,1> verticesAcc,
  AffineAccessor<size_t,1> insAcc,
  AffineAccessor<vertex,1> insBufferAcc,
  IndexSpace<1> insSpace
);

void runIteration(
  IndexSpace<1> edgesSpace,
  IndexSpace<1> onsSpace,
  AffineAccessor<vertex,1> insBufferAcc,
  AffineAccessor<vertex,1> verticesAcc,
  AffineAccessor<size_t,1> biiAcc,
  AffineAccessor<size_t,1> onsAcc,
  AffineAccessor<size_t,1> onsIndexAcc,
  AffineAccessor<size_t,1> outputsAcc
);

template<typename T>
void printGeneralRegion(RegionInstance region, FieldID id){
  GenericAccessor<T,1> acc(region, id);
  for(IndexSpaceIterator<1>i(region.get_indexspace<1>()); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      log_app.print() << acc.read(j.p);
    }
  }
};

//Loads edges, ons, and ins on gpu
static void prepare_graph(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  const PrepareGraphArgs &prepareGraphArgs = *reinterpret_cast<const PrepareGraphArgs *>(args);

  Memory deviceMemory = Machine::MemoryQuery(Machine::get_machine())
    .has_capacity(1 /*TODO make this a real memory check*/)
    .best_affinity_to(p) // TODO ensure this is using device memory
    .first();

  //Creates edgesGpu and duplicates edge data
  IndexSpace<1> edgesSpace = prepareGraphArgs.edges.get_indexspace<1>();
  std::map<FieldID, size_t> fieldSizes;
  fieldSizes[IN_VERTEX] = sizeof(size_t);
  fieldSizes[OUT_VERTEX] = sizeof(size_t);
  Event metadataFetch = prepareGraphArgs.edges.fetch_metadata(p);
  Event edgesGpuEvent = RegionInstance::create_instance(
    *prepareGraphArgs.edgesGpu,
    deviceMemory,
    edgesSpace,
    fieldSizes,
    0,
    ProfilingRequestSet(),
    metadataFetch
  );

  std::vector<CopySrcDstField> src(2);
  std::vector<CopySrcDstField> dst(2);
  src[0].set_field(prepareGraphArgs.edges, IN_VERTEX, sizeof(size_t));
  dst[0].set_field(*prepareGraphArgs.edgesGpu, IN_VERTEX, sizeof(size_t));
  src[1].set_field(prepareGraphArgs.edges, OUT_VERTEX, sizeof(size_t));
  dst[1].set_field(*prepareGraphArgs.edgesGpu, OUT_VERTEX, sizeof(size_t));
  Event edgesCopyEvent = edgesSpace.copy(src, dst, ProfilingRequestSet(), edgesGpuEvent);

  //Ensures data is copied into edge
  edgesCopyEvent.wait();
  log_app.print() << "about to generate neighbor set";
  generateNeighborSets(
    prepareGraphArgs.ins,
    prepareGraphArgs.ons,
    prepareGraphArgs.edgesGpu,
    edgesSpace,
    deviceMemory,
    prepareGraphArgs.insBuffer,
    prepareGraphArgs.bufferInputIds
  );
  //Now that we have generate out ins and ons set let's see what we got

  GenericAccessor<size_t,1> insAcc(*prepareGraphArgs.ins, IN_VERTEX);
  GenericAccessor<size_t,1> onsAcc(*prepareGraphArgs.ons, OUT_VERTEX);
  GenericAccessor<size_t,1> bIIAcc(*prepareGraphArgs.bufferInputIds, IN_VERTEX);
  GenericAccessor<vertex,1> insBufferAcc(*prepareGraphArgs.insBuffer, VERTEX_ID);


  log_app.print() << "ins";
  for(IndexSpaceIterator<1>i(insAcc.inst.get_indexspace<1>()); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      log_app.print() << insAcc.read(j.p);
    }
  }
  log_app.print() << "_________";
  
  log_app.print() << "ons";
  for(IndexSpaceIterator<1>i(onsAcc.inst.get_indexspace<1>()); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      log_app.print() << onsAcc.read(j.p);
    }
  }
  log_app.print() << "_________";

  log_app.print() << "bII";
  for(IndexSpaceIterator<1>i(bIIAcc.inst.get_indexspace<1>()); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      log_app.print() << bIIAcc.read(j.p);
    }
  }
  log_app.print() << "_________";

  log_app.print() << "insBuffer";
  for(IndexSpaceIterator<1>i(insBufferAcc.inst.get_indexspace<1>()); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      log_app.print() << insBufferAcc.read(j.p).value;
    }
  }
}

static void update_graph(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  const UpdateGraphArgs &updateGraphArgs = *reinterpret_cast<const UpdateGraphArgs *>(args);
  RegionInstance vertices = updateGraphArgs.vertices;
  RegionInstance edgesGpu = *updateGraphArgs.edgesGpu;
  RegionInstance ins = *updateGraphArgs.ins;
  RegionInstance ons = *updateGraphArgs.ons;
  RegionInstance insBuffer = *updateGraphArgs.insBuffer;
  RegionInstance bufferInputIds = *updateGraphArgs.bufferInputIds;

  AffineAccessor<vertex,1> verticesAcc(vertices, VERTEX_ID);
  AffineAccessor<size_t,1> insAcc(ins, IN_VERTEX);
  AffineAccessor<size_t,1> onsAcc(ons, OUT_VERTEX);
  AffineAccessor<size_t,1> onsIndexAcc(ons, START_INDEX);
  AffineAccessor<vertex,1> insBufferAcc(insBuffer, VERTEX_ID);
  AffineAccessor<size_t,1> outputsAcc(edgesGpu, OUT_VERTEX);
  AffineAccessor<size_t,1> biiAcc(bufferInputIds, IN_VERTEX);

  // Loads the vertices from ins into device memory
  loadInsVertices(
    verticesAcc,
    insAcc,
    insBufferAcc,
    ins.get_indexspace<1>()
  );

  log_app.print() << "initial vertices values";
  printGeneralRegion<vertex>(vertices, VERTEX_ID);

  //Runs the actual computation
  runIteration(
    edgesGpu.get_indexspace<1>(),
    ons.get_indexspace<1>(),
    insBufferAcc,
    verticesAcc,
    biiAcc,
    onsAcc,
    onsIndexAcc,
    outputsAcc
  );

  log_app.print() << "final vertices values";
  printGeneralRegion<vertex>(vertices, VERTEX_ID);

}

static void loadFakeVertices(RegionInstance region){
  AffineAccessor<vertex,1> acc(region, VERTEX_ID);
  IndexSpace<1> space = region.get_indexspace<1>();
  
  //Needleslly convoluted to pretend its general
  std::vector<int> toFill = {0,1,2,3,4};
  int index = 0;
  for(IndexSpaceIterator<1>i(space); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      acc[j.p].value = toFill[index++];
    }
  }
}

static void loadFakeEdges(RegionInstance region){
  AffineAccessor<size_t,1> inAcc(region, IN_VERTEX);
  AffineAccessor<size_t,1> outAcc(region, OUT_VERTEX);
  IndexSpace<1> space = region.get_indexspace<1>();
  
  std::vector<int> inFill = {3,3,0,4,1};
  std::vector<int> outFill = {0,1,2,2,2};
  int index = 0;
  for(IndexSpaceIterator<1>i(space); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      inAcc[j.p] = inFill[index];
      outAcc[j.p] = outFill[index++];
    }
  }
}

// This is the main task entry point that will initialize and coordinate all the various
// tasks we have.
static void main_task(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  //Get our initial fake vertices
  //TODO get a distributed file system to load in real graph data
  int vertexCount = 5;
  int edgeCount = 5;
  IndexSpace<1> vertexSpace(Rect<1>(0,vertexCount - 1));
  IndexSpace<1> edgeSpace(Rect<1>(0,edgeCount - 1));

  //Get memory to put graph onto zero copy/ best affinity memory for our initial ground truth
  Machine::MemoryQuery mq(Machine::get_machine());
  Memory nodeMemory = mq.only_kind(Memory::SYSTEM_MEM)
    .has_capacity(sizeof(vertex) * (vertexCount + 2 * sizeof(size_t)))
    .best_affinity_to(p).first();

  //Put our graph on node memory
  //TODO track effect of memory layout
  RegionInstance vertices;
  std::map<FieldID, size_t> fieldSizes;
  fieldSizes[VERTEX_ID] = sizeof(vertex);
  Event verticesEvent = RegionInstance::create_instance(
      vertices,
      nodeMemory,
      vertexSpace,
      fieldSizes,
      0,
      ProfilingRequestSet()
  );
  fieldSizes.clear();

  RegionInstance edges;
  fieldSizes[IN_VERTEX] = sizeof(size_t);
  fieldSizes[OUT_VERTEX] = sizeof(size_t);
  Event edgesEvent = RegionInstance::create_instance(
    edges,
    nodeMemory,
    edgeSpace,
    fieldSizes,
    0,
    ProfilingRequestSet()
  );

  Event graphReadyEvent = Event::merge_events(verticesEvent, edgesEvent);

  graphReadyEvent.wait();
  loadFakeVertices(vertices);
  loadFakeEdges(edges);

  //TODO deploy across multiple GPUs.
  Processor gpuProc = Machine::ProcessorQuery(Machine::get_machine())
    .local_address_space()
    .only_kind(Processor::TOC_PROC)
    .first();

  // Get the in-neighbor and out-neighbor set for our single GPU
  // TODO as always change for eventual multiple GPUs and nodes
  PrepareGraphArgs prepareGraphArgs = {
    vertices,
    edges,
    new RegionInstance,
    new RegionInstance,
    new RegionInstance,
    new RegionInstance,
    new RegionInstance
  };

  // Recall PREPARE_GRAPH must be run on the same address space
  Event graphProcessedEvent = gpuProc.spawn(
    PREPARE_GRAPH, 
    &prepareGraphArgs, 
    sizeof(PrepareGraphArgs), 
    graphReadyEvent
  );
  
  //Run a single iteration of our algorithm
  UpdateGraphArgs updateGraphArgs = {
    prepareGraphArgs.vertices,
    prepareGraphArgs.edgesGpu,
    prepareGraphArgs.ins,
    prepareGraphArgs.ons,
    prepareGraphArgs.insBuffer,
    prepareGraphArgs.bufferInputIds
  };
  Event GraphIteratedEvent = gpuProc.spawn(
    UPDATE_GRAPH,
    &updateGraphArgs,
    sizeof(updateGraphArgs),
    graphProcessedEvent
  );

  Runtime::get_runtime().shutdown(GraphIteratedEvent, 0 /*success*/);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  //Register application tasks
  {
    Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                    CodeDescriptor(main_task), ProfilingRequestSet(), 0, 0)
        .wait();
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, LOAD_DOUBLE_STORE,
                                    CodeDescriptor(load_double_store), ProfilingRequestSet(), 0, 0)
        .wait();
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, PREPARE_GRAPH,
                                    CodeDescriptor(prepare_graph), ProfilingRequestSet(), 0, 0)
        .wait();
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, UPDATE_GRAPH,
                                    CodeDescriptor(update_graph), ProfilingRequestSet(), 0, 0)
        .wait();
  }

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  rt.collective_spawn(p, MAIN_TASK, 0, 0);

  return rt.wait_for_shutdown();
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

