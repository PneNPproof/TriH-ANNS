/**
 * @file topk_by_sort.cuh
 * @brief GPU-based top-k selection using CUB radix sort
 * 
 * This header provides GPU-accelerated top-k selection using NVIDIA's CUB
 * library radix sort. While not as specialized as dedicated top-k algorithms,
 * radix sort provides consistent performance and can be efficiently used
 * for moderately sized candidate sets.
 * 
 * The implementation uses CUB's double-buffering mechanism for optimal
 * memory utilization and performance on modern GPU architectures.
 * 
 * Dependencies:
 * - NVIDIA CUB library for device-wide primitives
 * - Sufficient GPU memory for double-buffered operations
 */

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include <stddef.h>

/**
 * @brief GPU top-k selection using radix sort
 * 
 * Performs top-k selection by sorting distance-index pairs using CUB's
 * highly optimized radix sort. After sorting, the first k elements
 * contain the k smallest distances and their corresponding indices.
 * 
 * @param dis Double buffer for distances (input/output)
 * @param ind Double buffer for indices (input/output)
 * @param n Total number of elements to sort
 * @param temp_storage_bytes Size of temporary storage required
 * @param d_temp_storage Temporary GPU memory for sorting operations
 * 
 * @note Uses CUB's DoubleBuffer for optimal memory access patterns
 * @note Caller must query temp_storage_bytes before calling
 * @note Results are sorted in ascending order (smallest distances first)
 * @note More general than specialized top-k but very fast for moderate sizes
 */
void topk_by_sort(
  cub::DoubleBuffer<float> &dis,
  cub::DoubleBuffer<int> &ind,
  int n,
  size_t temp_storage_bytes,
  void *d_temp_storage
);