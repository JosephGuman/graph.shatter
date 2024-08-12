#include "graph_shatter_types.h"


using namespace Realm;


Logger log_app("app");


void generateNeighborSets(
  RegionInstance *ins,
  RegionInstance *ons,
  RegionInstance edges,
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


//Loads edges, ons, and ins on gpu
static void prepare_graph(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  PrepareGraphArgs &prepareGraphArgs = *reinterpret_cast<PrepareGraphArgs *>(const_cast<void*>(args));
  std::pair<RegionInstance, Memory> temp = prepareGraphArgs.graphPartition->getChild(prepareGraphArgs.partitionColor);
  Memory deviceMemory = temp.second;

  GenericAccessor<size_t,1> testAcc(temp.first, OUT_VERTEX);

  IndexSpace<1> edgesSpace = temp.first.get_indexspace<1>();
  generateNeighborSets(
    prepareGraphArgs.ins,
    prepareGraphArgs.ons,
    temp.first,
    edgesSpace,
    deviceMemory,
    prepareGraphArgs.insBuffer,
    prepareGraphArgs.bufferInputIds
  );

  // Kep this around in preparation for TODOs in generateNeighborSets
  //Now that we have generate out ins and ons set let's see what we got
  /*
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
  */
}


// Loads ins set into device memory for pull model
static void load_graph(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  LoadGraphArgs &loadGraphArgs = *reinterpret_cast<LoadGraphArgs *>(const_cast<void*>(args));

  RegionInstance vertices = loadGraphArgs.vertices;
  RegionInstance ins = loadGraphArgs.ins;
  RegionInstance insBuffer = loadGraphArgs.insBuffer;

  AffineAccessor<vertex,1> verticesAcc(vertices, VERTEX_ID);
  AffineAccessor<size_t,1> insAcc(ins, IN_VERTEX);
  AffineAccessor<vertex,1> insBufferAcc(insBuffer, VERTEX_ID);

  // Loads the vertices from ins into device memory
  loadInsVertices(
    verticesAcc,
    insAcc,
    insBufferAcc,
    ins.get_indexspace<1>()
  );
}


static void update_graph(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  UpdateGraphArgs &updateGraphArgs = *reinterpret_cast<UpdateGraphArgs *>(const_cast<void*>(args));
  RegionInstance vertices = updateGraphArgs.vertices;
  RegionInstance edgesGpu = updateGraphArgs.graphPartition->getChild(updateGraphArgs.partitionColor).first;
  RegionInstance ons = *updateGraphArgs.ons;
  RegionInstance insBuffer = *updateGraphArgs.insBuffer;
  RegionInstance bufferInputIds = *updateGraphArgs.bufferInputIds;

  AffineAccessor<vertex,1> verticesAcc(vertices, VERTEX_ID);
  AffineAccessor<size_t,1> onsAcc(ons, OUT_VERTEX);
  AffineAccessor<size_t,1> onsIndexAcc(ons, START_INDEX);
  AffineAccessor<vertex,1> insBufferAcc(insBuffer, VERTEX_ID);
  AffineAccessor<size_t,1> outputsAcc(edgesGpu, OUT_VERTEX);
  AffineAccessor<size_t,1> biiAcc(bufferInputIds, IN_VERTEX);

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

  // log_app.print() << "final vertices values";
  // printGeneralRegion<vertex>(vertices, VERTEX_ID);

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


static void loadFakeEdges(RegionInstance region, RegionInstance edgeBoundaries){
  AffineAccessor<size_t,1> inAcc(region, IN_VERTEX);
  AffineAccessor<size_t,1> outAcc(region, OUT_VERTEX);
  IndexSpace<1> space = region.get_indexspace<1>();
  
  std::vector<int> inFill = {2,4,0,3,4,0,4};
  std::vector<int> outFill = {0,0,1,1,2,3,3};
  int index = 0;
  for(IndexSpaceIterator<1>i(space); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      inAcc[j.p] = inFill[index];
      outAcc[j.p] = outFill[index++];
    }
  }

  // Fill in the edge boundaries
  // Just hardprogram this because it will be obvious for our file inputs
  std::vector<int> boundaryVector = {0,2,4,5};
  AffineAccessor<size_t,1> boundaryAcc(edgeBoundaries, OUT_VERTEX);
  for(uint i = 0; i < boundaryVector.size(); i++){
    boundaryAcc[i] = boundaryVector[i];
  }
}


static void main_task(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  //Get our initial fake vertices
  //TODO get a distributed file system to load in real graph data
  int vertexCount = 5;
  int edgeCount = 7;
  int outEdgeCount = 4;
  IndexSpace<1> vertexSpace(Rect<1>(0,vertexCount - 1));
  IndexSpace<1> edgeSpace(Rect<1>(0,edgeCount - 1));
  IndexSpace<1> boundariesSpace(Rect<1>(0,outEdgeCount - 1));

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
  fieldSizes.clear();

  RegionInstance edgeBoundaries;
  fieldSizes[OUT_VERTEX] = sizeof(size_t);
  Event boundariesEvent = RegionInstance::create_instance(
    edgeBoundaries,
    nodeMemory,
    boundariesSpace,
    fieldSizes,
    0,
    ProfilingRequestSet()
  );


  Event graphReadyEvent = Event::merge_events(
    verticesEvent,
    edgesEvent,
    boundariesEvent
  );

  graphReadyEvent.wait();
  loadFakeVertices(vertices);
  loadFakeEdges(edges, edgeBoundaries);

  //TODO deploy across multiple GPUs.
  std::vector<Processor> availableGpuProcs = GraphPartition::getDefaultNodeGpus();
  Rect<1> partitionSpace(0, availableGpuProcs.size() - 1);
  GraphPartition baseNodePartition(edges, partitionSpace, availableGpuProcs, edgeBoundaries);

  std::vector<PrepareGraphArgs> prepareGraphArgs;
  std::vector<Event> graphReadyEvents(partitionSpace.volume());
  for(PointInRectIterator<1> partitionPoint(partitionSpace); partitionPoint.valid; partitionPoint.step()){
    // Get the in-neighbor and out-neighbor set for our single GPU
    prepareGraphArgs.push_back({
      &baseNodePartition,
      partitionPoint.p,
      new RegionInstance,
      new RegionInstance,
      new RegionInstance,
      new RegionInstance
    });

    // Recall PREPARE_GRAPH must be run on the same address space
    graphReadyEvents[partitionPoint.p] = baseNodePartition.getProc(partitionPoint.p).spawn(
      PREPARE_GRAPH, 
      &prepareGraphArgs[partitionPoint.p], 
      sizeof(PrepareGraphArgs), 
      graphReadyEvent
    );

  }

  //Wait until each GPU has finished its shard's processing
  Event::merge_events(graphReadyEvents).wait();

  std::vector<Event> graphLoadEvents(partitionSpace.volume());
  for(PointInRectIterator<1> partitionPoint(partitionSpace); partitionPoint.valid; partitionPoint.step()){
    //Run a single iteration of our algorithm
    LoadGraphArgs loadGraphArgs = {
      vertices,
      *prepareGraphArgs[partitionPoint.p].ins,
      *prepareGraphArgs[partitionPoint.p].insBuffer
    };

    graphLoadEvents[partitionPoint.p] = baseNodePartition.getProc(partitionPoint.p).spawn(
      LOAD_GRAPH,
      &loadGraphArgs,
      sizeof(loadGraphArgs),
      Event::NO_EVENT
    );
  }

  //Wait until each GPU has finished its shard's loading
  Event::merge_events(graphLoadEvents).wait();

  std::vector<Event> graphIterateEvents(partitionSpace.volume());
  for(PointInRectIterator<1> partitionPoint(partitionSpace); partitionPoint.valid; partitionPoint.step()){
    //Run a single iteration of our algorithm
    UpdateGraphArgs updateGraphArgs = {
      &baseNodePartition,
      partitionPoint.p,
      vertices,
      prepareGraphArgs[partitionPoint.p].ons,
      prepareGraphArgs[partitionPoint.p].insBuffer,
      prepareGraphArgs[partitionPoint.p].bufferInputIds
    };

    // TODO modify Event logic for multiple iterations
    graphIterateEvents[partitionPoint.p] = baseNodePartition.getProc(partitionPoint.p).spawn(
      UPDATE_GRAPH,
      &updateGraphArgs,
      sizeof(updateGraphArgs),
      Event::NO_EVENT
    );

    // continue;
  }

  Event totalIterateEvent = Event::merge_events(graphIterateEvents);
  totalIterateEvent.wait();
  log_app.print() << "we are done here" << vertices.get_indexspace<1>();
  printGeneralRegion<vertex>(vertices, VERTEX_ID);

  Runtime::get_runtime().shutdown(totalIterateEvent, 0 /*success*/);
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
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, PREPARE_GRAPH,
                                    CodeDescriptor(prepare_graph), ProfilingRequestSet(), 0, 0)
        .wait();
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, LOAD_GRAPH,
                                    CodeDescriptor(load_graph), ProfilingRequestSet(), 0, 0)
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