// #include "graph_shatter_partition.h"
#include <cuda_runtime_api.h>
#include <realm.h>
#include <realm/cuda/cuda_access.h>
#include <iostream>

#include <map>

using namespace Realm;

extern Logger log_app;

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

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE,
  LOAD_DOUBLE_STORE,
  PREPARE_GRAPH,
  LOAD_GRAPH,
  UPDATE_GRAPH,
};


// fields for edges/nodes
enum
{
  IN_VERTEX,
  OUT_VERTEX,
  VERTEX_ID,
  START_INDEX,
};


struct vertex{
  int value;

  friend std::ostream& operator<<(std::ostream& os, const vertex& s) {
    os << s.value;
    return os;
  }
};

struct LDSArgs{
  IndexSpace<1> is;
  RegionInstance data;
};


/*
* A representation of a RegionInstance partition on a single node
*/
class GraphPartition{
  RegionInstance parent;
  RegionInstance boundaries;
  AffineAccessor<size_t,1> boundariesAcc;
  std::vector<RegionInstance> children;
  std::vector<Memory> childrenMemories;
  std::vector<Processor> processors;
  Rect<1> colorSpace;

  public:
  GraphPartition(
    RegionInstance pa,
    Rect<1> cS,
    std::vector<Processor> procs,
    RegionInstance bo
  ) : 
    parent(pa),
    boundaries(bo),
    boundariesAcc(boundaries, OUT_VERTEX),
    children(colorSpace.volume()),
    childrenMemories(colorSpace.volume()),
    processors(procs),
    colorSpace(cS)
  {}
  
  // Return region instance on corresponding processor
  std::pair<RegionInstance, Memory> getChild(Point<1> p){
    assert(colorSpace.contains(p));

    if(children[p].exists()){
      return {children[p], childrenMemories[p]};
    }

    std::map<FieldID, size_t> fieldSizes;
    fieldSizes[IN_VERTEX] = sizeof(size_t);
    fieldSizes[OUT_VERTEX] = sizeof(size_t);

    Memory deviceMemory = Machine::MemoryQuery(Machine::get_machine())
      .has_capacity(1 /*TODO make this a real memory check*/)
      .best_affinity_to(getProc(p)) // TODO ensure this is using device memory
      .first();

    childrenMemories[p] = deviceMemory;
    IndexSpace<1> childSpace = getSubSpace(p);

    Event childRegionEvent = RegionInstance::create_instance(
      children[p],
      deviceMemory,
      childSpace,
      fieldSizes,
      0,
      ProfilingRequestSet()
    );
    
    std::vector<CopySrcDstField> srcs(2), dsts(2);
    srcs[0].set_field(parent, IN_VERTEX, sizeof(size_t));
    dsts[0].set_field(children[p], IN_VERTEX, sizeof(size_t));
    srcs[1].set_field(parent, OUT_VERTEX, sizeof(size_t));
    dsts[1].set_field(children[p], OUT_VERTEX, sizeof(size_t));
    // TODO I'm not sure if this is important
    Event metadataFetch = parent.fetch_metadata(getProc(p));
    // TODO do I really want to hang here?
    childSpace.copy(srcs, dsts, ProfilingRequestSet(), Event::merge_events(childRegionEvent, metadataFetch)).wait();

    return {children[p], childrenMemories[p]};
  }

  Processor getProc(Point<1> p){
    assert(colorSpace.contains(p));
    return processors[p % processors.size()];
  }

  // TODO create general programmatically and manually defined partitions
  // For now we are just doing the most basic partitioning strategy
  IndexSpace<1> getSubSpace(Point<1> p){
    assert(colorSpace.contains(p));
    IndexSpace<1> boundarySpace = boundaries.get_indexspace<1>();
    assert(boundarySpace.dense());
    size_t boundaryVolume = boundarySpace.volume();
    size_t baseWidth = (boundaryVolume + colorSpace.volume() - 1) / colorSpace.volume();
    
    //The bounds for where we look within boundaries to find the actual bounds for parent
    Point<1> lowerBound = p * baseWidth;
    Point<1> upperBound = std::min((size_t) boundarySpace.bounds.hi, lowerBound + baseWidth - 1);

    size_t maxBound = lowerBound + baseWidth - 1;
    if(maxBound < (size_t) boundarySpace.bounds.hi){
      return IndexSpace<1>(Rect<1>(boundariesAcc[lowerBound], boundariesAcc[upperBound + 1] - 1));
    }
    return IndexSpace<1>(Rect<1>(boundariesAcc[lowerBound], parent.get_indexspace<1>().bounds.hi));
  }

  static std::vector<Processor> getDefaultNodeGpus(){
    //Find available processors on this node
    std::vector<Processor> availableGpuProcs;
    Machine::ProcessorQuery gpuProcquery = Machine::ProcessorQuery(Machine::get_machine())
      .local_address_space()
      .only_kind(Processor::TOC_PROC);
    for(auto it = gpuProcquery.begin(); it != gpuProcquery.end(); it++){
      availableGpuProcs.push_back(*it);
    }

    return availableGpuProcs;
  }
};

//Note: this assumes prepare_graph callee is on the same node as caller
// TODO add a layer of extraction when expanding to multiple nodes
struct PrepareGraphArgs{
  GraphPartition *graphPartition; // partition over edges
  // RegionInstance edgesGpu;
  // Memory deviceMemory;
  Point<1> partitionColor; // color for that GPU processor
  RegionInstance *ins;
  RegionInstance *ons;
  RegionInstance *insBuffer;
  RegionInstance *bufferInputIds;
};

struct LoadGraphArgs{
  RegionInstance vertices;
  RegionInstance ins;
  RegionInstance insBuffer;
};

struct UpdateGraphArgs{
  GraphPartition *graphPartition; // partition over edges
  Point<1> partitionColor; // color for that GPU processor
  RegionInstance vertices;
  RegionInstance *ons;
  RegionInstance *insBuffer;
  RegionInstance *bufferInputIds;
};

template<typename T>
void printGeneralRegion(RegionInstance region, FieldID id);