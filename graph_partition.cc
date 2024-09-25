#include "graph_shatter_types.h"


GraphPartition::GraphPartition(
    RegionInstance pa,
    Rect<1> cS,
    std::vector<Processor> procs,
    RegionInstance bo
) : 
    parent(pa),
    boundaries(bo),
    boundariesAcc(boundaries, OUT_VERTEX),
    colorSpace(cS),
    children(colorSpace.volume()),
    childrenBoundaries(colorSpace.volume()),
    childrenMemories(colorSpace.volume()),
    valid(colorSpace.volume(), false),
    processors(procs)
{}

PartitionChild GraphPartition::getChild(Point<1> p){
    assert(colorSpace.contains(p));

    if(valid[0]){
        return {children[p], childrenBoundaries[p], childrenMemories[p]};
    }

    Memory deviceMemory = Machine::MemoryQuery(Machine::get_machine())
        .has_capacity(1 /*TODO make this a real memory check*/)
        .best_affinity_to(getProc(p)) // TODO ensure this is using device memory
        .first();

    childrenMemories[p] = deviceMemory;

    std::pair<IndexSpace<1>, IndexSpace<1>> childSpaces = getSubSpace(p);
    IndexSpace<1> childSpace = childSpaces.first;
    IndexSpace<1> boundarySpace = childSpaces.second;

    std::map<FieldID, size_t> fieldSizes;
    fieldSizes[IN_VERTEX] = sizeof(size_t);
    Event childRegionEvent = RegionInstance::create_instance(
        children[p],
        deviceMemory,
        childSpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    );
    fieldSizes.clear();

    fieldSizes[OUT_VERTEX] = sizeof(size_t);
    Event boundaryRegionEvent = RegionInstance::create_instance(
        childrenBoundaries[p],
        deviceMemory,
        boundarySpace,
        fieldSizes,
        0,
        ProfilingRequestSet()
    );

    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(parent, IN_VERTEX, sizeof(size_t));
    dsts[0].set_field(children[p], IN_VERTEX, sizeof(size_t));
    Event metadataFetch = parent.fetch_metadata(getProc(p));

    std::vector<CopySrcDstField> boundarysrcs(1), boundarydsts(1);
    boundarysrcs[0].set_field(boundaries, OUT_VERTEX, sizeof(size_t));
    boundarydsts[0].set_field(childrenBoundaries[p], OUT_VERTEX, sizeof(size_t));
    Event boundaryMetadataFetch = boundaries.fetch_metadata(getProc(p));

    Event dataTransferEvent = childSpace.copy(srcs, dsts, ProfilingRequestSet(), Event::merge_events(childRegionEvent, metadataFetch));
    Event boundaryTransferEvent = boundarySpace.copy(
        boundarysrcs,
        boundarydsts,
        ProfilingRequestSet(),
        Event::merge_events(boundaryRegionEvent, boundaryMetadataFetch)
    );

    Event::merge_events(dataTransferEvent, boundaryTransferEvent).wait();

    valid[p] = true;
    return {children[p], childrenBoundaries[p], childrenMemories[p]};
}

Processor GraphPartition::getProc(Point<1> p){
    assert(colorSpace.contains(p));
    return processors[p % processors.size()];
}

// TODO create general programmatically and manually defined partitions
// For now we are just doing the most basic partitioning strategy
std::pair<IndexSpace<1>, IndexSpace<1>> GraphPartition::getSubSpace(Point<1> p){
    assert(colorSpace.contains(p));

    IndexSpace<1> boundarySpace = boundaries.get_indexspace<1>();
    assert(boundarySpace.dense());
    Rect<1> selectionSpace(boundarySpace.bounds.lo, boundarySpace.bounds.hi - 1);

    size_t boundaryVolume = selectionSpace.volume();
    size_t baseWidth = (boundaryVolume + colorSpace.volume() - 1) / colorSpace.volume();

    //The bounds for where we look within boundaries to find the actual bounds for parent
    Point<1> lowerBound = p * baseWidth;
    Point<1> upperBound = std::min((size_t) selectionSpace.hi + 1, lowerBound + baseWidth);

    // TODO make sure geometrically impossible Rects do not cause problems
    IndexSpace<1> childBoundarySpace = {Rect<1>(lowerBound, upperBound)};
    IndexSpace<1> childDataSpace =  {Rect<1>(boundariesAcc[lowerBound], boundariesAcc[upperBound] - 1)};

    return {childDataSpace, childBoundarySpace};
}

std::vector<Processor> GraphPartition::getDefaultNodeGpus(){
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
