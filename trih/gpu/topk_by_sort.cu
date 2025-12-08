#include "topk_by_sort.cuh"

// using namespace cub;

void topk_by_sort(
  cub::DoubleBuffer<float> &dis,
  cub::DoubleBuffer<int> &ind,
  int n,
  size_t temp_storage_bytes,
  void *d_temp_storage
)
{
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, dis, ind, n);
}