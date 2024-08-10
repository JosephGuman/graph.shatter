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
    children(colorSpace.volume()),
    childrenMemories(colorSpace.volume()),
    processors(procs),
    colorSpace(cS)
{}

std::pair<RegionInstance, Memory> GraphPartition::getChild(Point<1> p){
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

Processor GraphPartition::getProc(Point<1> p){
    assert(colorSpace.contains(p));
    return processors[p % processors.size()];
}

// TODO create general programmatically and manually defined partitions
// For now we are just doing the most basic partitioning strategy
IndexSpace<1> GraphPartition::getSubSpace(Point<1> p){
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
