/*
* Read from graphs in CSC format
*/
#include "graph_shatter_types.h"
#include "utils.cc"


using namespace Realm;

void writeHardCoded(std::string fileName){
    std::ofstream outFile(fileName, std::ios::binary);
    assert(outFile);
    
    size_t vertexCount = 5;
    size_t edgeCount = 7;
    outFile.write(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));
    outFile.write(reinterpret_cast<char*>(&edgeCount), sizeof(edgeCount));

    size_t colIndices[] = {0, 2, 4, 5, 7, 7};
    outFile.write(reinterpret_cast<char*>(colIndices), sizeof(colIndices));

    size_t rowIndices[] = {2,4,0,3,4,0,4};
    outFile.write(reinterpret_cast<char*>(rowIndices), sizeof(rowIndices));

    vertex vertices[] = {{0},{1},{2},{3},{4}};
    outFile.write(reinterpret_cast<char*>(vertices), sizeof(vertices));

    outFile.close();
}


GraphRegions readGraphFromFile(std::string fileName){
    FILE* fd = fopen(fileName.c_str(), "rb");

    size_t vertexCount, edgeCount;

    fread(&vertexCount, sizeof(vertexCount), 1, fd);
    fread(&edgeCount, sizeof(edgeCount), 1, fd);

    IndexSpace<1> vertexSpace(Rect<1>(0,vertexCount - 1));
    IndexSpace<1> edgeSpace(Rect<1>(0,edgeCount - 1));
    IndexSpace<1> boundariesSpace(Rect<1>(0,vertexCount));

    Processor p = Machine::ProcessorQuery(Machine::get_machine())
        .only_kind(Processor::LOC_PROC)
        .first();

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

    Event::merge_events(
        verticesEvent,
        edgesEvent,
        boundariesEvent
    ).wait();

    // This part of the code loads in edges and edgeBoundaries
    AffineAccessor<size_t,1>boundariesAcc(edgeBoundaries, OUT_VERTEX);
    AffineAccessor<size_t,1>inputAcc(edges, IN_VERTEX);
    AffineAccessor<size_t,1>verticesAcc(vertices, VERTEX_ID);

    fread(boundariesAcc.ptr(0), sizeof(size_t) * (vertexCount + 1), 1, fd);
    fread(inputAcc.ptr(0), sizeof(size_t) * edgeCount, 1, fd);
    fread(verticesAcc.ptr(0), sizeof(vertex) * vertexCount, 1, fd);

    return {vertices, edges, edgeBoundaries};
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