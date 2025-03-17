#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launchComputeGroupMinimaKernel(
  const float* d_input,
  int n,
  int m,
  int group_size,
  float **d_output_min,
  int **d_output_idx,
  int block_size);

// New function for half-precision row-wise minimum index finding
void half_matrix_reduce(
  const half* dists_per_query,
  half* reduced_dists_per_query,
  int* reduced_ids_per_query,
  int segment_size,
  int segment_num,
  int seg_num_per_query,
  cudaStream_t stream
);

cudaError_t segmented_sort_topk_pairs_fp16(
  half* reduced_dists,    // [query_num * group_num] in/out - half precision
  int*  reduced_inds,     // [query_num * group_num] in/out
  int   query_num,        // number of segments
  int   group_num,        // length of each segment
  int   phase1_topk);     // how many elements to extract from each segment