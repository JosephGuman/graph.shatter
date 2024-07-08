#include <cuda_runtime_api.h>
#include <realm.h>
#include <realm/cuda/cuda_access.h>

#include <map>

using namespace Realm;



namespace GraphShatter{

template<typename T>
class Ltable;

template<int D>
class Partition{
    IndexSpace<D> colorSpace;

    Partition(IndexSpace<D> cS){
        this->colorSpace = cS;
    };
    virtual Ltable<int> getChild(Point<D> p) = 0;
};



}