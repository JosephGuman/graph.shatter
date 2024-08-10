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
  );
  
  // Return region instance on corresponding processor
  std::pair<RegionInstance, Memory> getChild(Point<1> p);
  Processor getProc(Point<1> p);
  IndexSpace<1> getSubSpace(Point<1> p);

  static std::vector<Processor> getDefaultNodeGpus();
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