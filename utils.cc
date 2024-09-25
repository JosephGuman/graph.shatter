template<typename T>
void printGeneralRegion(RegionInstance region, FieldID id){
  GenericAccessor<T,1> acc(region, id);
  for(IndexSpaceIterator<1>i(region.get_indexspace<1>()); i.valid; i.step()){
    for (PointInRectIterator<1> j(i.rect); j.valid; j.step()) {
      log_app.print() << acc.read(j.p);
    }
  }
  log_app.print() << "---------";
}

template<typename T>
void printGeneralRegion(RegionInstance region, FieldID id, std::string label){
  log_app.print() << label;
  printGeneralRegion<T>(region, id);
}