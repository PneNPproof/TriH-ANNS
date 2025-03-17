#include <cuda_runtime.h>

#include <stdio.h>
// input is float **pdis_per_query and int **pidx_per_query, and float *src_data, float *query, int query_num, int record_num, int dim, int phase1_topk, int phase1_topk
// for query i, pdis_per_query[i] is an array of size phase1_topk, and pidx_per_query[i] is an array of size phase1_topk
// src_data is the row-major data matrix of size record_num x dim
// for each query, based on the phase1_topk neighbors, we could extract the corresponding rows from src_data and compute the exact distances
// and exact distance is stored in pdis_per_query[i]
// each block process one query
__global__ void compute_exact_distances_kernel(
  float** pdis_per_query,
  int** pidx_per_query,
  const float* src_data,
  const float* queries,
  int record_num,
  int dim,
  int phase1_topk)
{
  extern __shared__ float shared_query[];

  int query_idx = blockIdx.x;
  int neighbor_idx = threadIdx.x;

  // Load query vector into shared memory
  if (neighbor_idx < dim) {
      shared_query[neighbor_idx] = queries[query_idx * dim + neighbor_idx];
  }
  __syncthreads();

  if (neighbor_idx >= phase1_topk) return;

  // Get record index for this neighbor
  int record_idx = pidx_per_query[query_idx][neighbor_idx];

  // Compute squared distance
  const float* record_vec = src_data + record_idx * dim;
  float sum = 0.0f;
  for (int d = 0; d < dim; ++d) {
      float diff = shared_query[d] - record_vec[d];
      sum += diff * diff;
  }
  pdis_per_query[query_idx][neighbor_idx] = sum;
}

void compute_exact_distances(
  float** pdis_per_query,
  int** pidx_per_query,
  const float* src_data,
  const float* queries,
  int query_num,
  int record_num,
  int dim,
  int phase1_topk,
  int block_size)  // Block size (threads per block) passed as argument
{
  // Validate block size
  if (block_size > 1024) {
      printf("Error: Block size cannot exceed 1024 (CUDA limit)\n");
      return;
  }

  // Calculate shared memory size
  size_t shared_mem_size = dim * sizeof(float);

  // Launch kernel
  compute_exact_distances_kernel<<<query_num, block_size, shared_mem_size>>>(
      pdis_per_query,
      pidx_per_query,
      src_data,
      queries,
      record_num,
      dim,
      phase1_topk
  );

  cudaStreamSynchronize(0);

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }
}