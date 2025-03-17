#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include <stddef.h>



void topk_by_sort(
  cub::DoubleBuffer<float> &dis,
  cub::DoubleBuffer<int> &ind,
  int n,
  size_t temp_storage_bytes,
  void *d_temp_storage
);