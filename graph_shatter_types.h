// #include "graph_shatter_partition.h"
#include <cuda_runtime_api.h>
#include <realm.h>
#include <realm/cuda/cuda_access.h>
#include <iostream>

#include <map>

using namespace Realm;

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
  UPDATE_GRAPH,
};


// fields for edges
enum
{
  IN_VERTEX,
  OUT_VERTEX,
};

// Field for nodes
enum
{
  VERTEX_ID
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

//Note: this assumes prepare_graph callee is on the same node as caller
// TODO add a layer of extraction when expanding to multiple nodes
struct PrepareGraphArgs{
  RegionInstance vertices;
  RegionInstance edges;
  RegionInstance *edgesGpu;
  RegionInstance *ins;
  RegionInstance *ons;
  RegionInstance *insBuffer;
  RegionInstance *bufferInputIds;
};

struct UpdateGraphArgs{
  RegionInstance vertices;
  RegionInstance *edgesGpu;
  RegionInstance *ins;
  RegionInstance *ons;
  RegionInstance *insBuffer;
  RegionInstance *bufferInputIds;
};

template<typename T>
void printGeneralRegion(RegionInstance region, FieldID id);