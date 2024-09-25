#include "graph_shatter_types.h"
#include "utils.cc"

using namespace Realm;


Logger log_app("app");


void generateNeighborSets(
  RegionInstance *ins,
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
  IndexSpace<1> boundariesSpace,
  AffineAccessor<vertex,1> insBufferAcc,
  AffineAccessor<vertex,1> verticesAcc,
  AffineAccessor<size_t,1> biiAcc,
  AffineAccessor<size_t,1> boundariesAcc
);

void writeHardCoded(std::string fileName);
GraphRegions readGraphFromFile(std::string fileName);

//Loads edges, ons, and ins on gpu
static void prepare_graph(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  PrepareGraphArgs &prepareGraphArgs = *reinterpret_cast<PrepareGraphArgs *>(const_cast<void*>(args));
  PartitionChild partitionData = prepareGraphArgs.graphPartition->getChild(prepareGraphArgs.partitionColor);
  RegionInstance deviceData = partitionData.data;
  Memory deviceMemory = partitionData.memory;

  IndexSpace<1> edgesSpace = deviceData.get_indexspace<1>();
  generateNeighborSets(
    prepareGraphArgs.ins,
    deviceData,
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
  PartitionChild temp = updateGraphArgs.graphPartition->getChild(updateGraphArgs.partitionColor);
  RegionInstance edgesGpu = temp.data;
  RegionInstance boundariesGpu = temp.boundaries;
  RegionInstance insBuffer = *updateGraphArgs.insBuffer;
  RegionInstance bufferInputIds = *updateGraphArgs.bufferInputIds;

  AffineAccessor<vertex,1> verticesAcc(vertices, VERTEX_ID);
  AffineAccessor<vertex,1> insBufferAcc(insBuffer, VERTEX_ID);
  AffineAccessor<size_t,1> biiAcc(bufferInputIds, IN_VERTEX);
  AffineAccessor<size_t,1> boundariesAcc(boundariesGpu, OUT_VERTEX);

  //Runs the actual computation
  runIteration(
    edgesGpu.get_indexspace<1>(),
    boundariesGpu.get_indexspace<1>(),
    insBufferAcc,
    verticesAcc,
    biiAcc,
    boundariesAcc
  );
}


static void main_task(const void *args, size_t datalen, const void *userdata,
                      size_t userlen, Processor p)
{
  std::string data_file = "data_files/exampleload.bin";
  writeHardCoded(data_file);
  GraphRegions graphRegions = readGraphFromFile(data_file);

  RegionInstance vertices = graphRegions.vertices;
  RegionInstance edges = graphRegions.edges;
  RegionInstance edgeBoundaries = graphRegions.edgeBoundaries;

  //TODO deploy across multiple GPUs.
  std::vector<Processor> availableGpuProcs = GraphPartition::getDefaultNodeGpus();
  Rect<1> partitionSpace(0, availableGpuProcs.size() - 1);
  GraphPartition* baseNodePartition = new GraphPartition(edges, partitionSpace, availableGpuProcs, edgeBoundaries);

  std::vector<PrepareGraphArgs> prepareGraphArgs;
  std::vector<Event> graphReadyEvents(partitionSpace.volume());
  for(PointInRectIterator<1> partitionPoint(partitionSpace); partitionPoint.valid; partitionPoint.step()){
    // Get the in-neighbor and out-neighbor set for our single GPU
    prepareGraphArgs.push_back({
      baseNodePartition,
      partitionPoint.p,
      new RegionInstance,
      new RegionInstance,
      new RegionInstance
    });

    // Recall PREPARE_GRAPH must be run on the same address space
    graphReadyEvents[partitionPoint.p] = baseNodePartition->getProc(partitionPoint.p).spawn(
      PREPARE_GRAPH, 
      &prepareGraphArgs[partitionPoint.p], 
      sizeof(PrepareGraphArgs), 
      Event::NO_EVENT
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

    graphLoadEvents[partitionPoint.p] = baseNodePartition->getProc(partitionPoint.p).spawn(
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
      baseNodePartition,
      partitionPoint.p,
      vertices,
      prepareGraphArgs[partitionPoint.p].insBuffer,
      prepareGraphArgs[partitionPoint.p].bufferInputIds
    };

    // TODO modify Event logic for multiple iterations
    graphIterateEvents[partitionPoint.p] = baseNodePartition->getProc(partitionPoint.p).spawn(
      UPDATE_GRAPH,
      &updateGraphArgs,
      sizeof(updateGraphArgs),
      Event::NO_EVENT
    );
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