#include "gist.h"
#include "pca.h"

#include "gpu_pca_anns.cuh"
#include "l2mm.cuh"
#include "reduce_min.cuh"
#include "rerank.cuh"
#include "thread_pool.h"
#include "rerank.h"

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/util_type.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <vector>



// Define a kernel to convert half to float
__global__ void half_to_float_kernel(half* input, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __half2float(input[idx]);
  }
}

void gpu_anns
(
  float *query,
  int query_num,
  float *src_data,
  pca_index &index,
  float *distances,
  int *neighbors,
  int reduce_group_size,
  int phase1_topk,
  int phase2_topk,
  int *ground_truth_neighbors
)
{

  // Define alignment boundary - 64 bytes (typical cache line size)
  constexpr size_t alignment = 64;
  
  // Create aligned copies of the input data
  float *aligned_query = nullptr;
  float *aligned_src_data = nullptr;
  
  // Calculate sizes and ensure they're multiples of alignment
  size_t query_size = query_num * index.dim * sizeof(float);
  size_t src_data_size = index.record_num * index.dim * sizeof(float);
  
  // Allocate aligned memory
  #if defined(_MSC_VER)
    // Windows aligned allocation
    aligned_query = (float*)_aligned_malloc(query_size, alignment);
    aligned_src_data = (float*)_aligned_malloc(src_data_size, alignment);
  #else
    // POSIX aligned allocation
    aligned_query = (float*)aligned_alloc(alignment, query_size);
    aligned_src_data = (float*)aligned_alloc(alignment, src_data_size);
  #endif
  
  if (!aligned_query || !aligned_src_data) {
    printf("ERROR: Failed to allocate aligned memory\n");
    return;
  }
  
  // Copy data to aligned memory
  memcpy(aligned_query, query, query_size);
  memcpy(aligned_src_data, src_data, src_data_size);

  query = aligned_query;
  src_data = aligned_src_data;

  /// calculate squared norms for src_data
  float *src_data_norms;
  cudaMallocHost(&src_data_norms, index.record_num * sizeof(float));
  for (int i = 0; i < index.record_num; i++)
  {
    src_data_norms[i] = 0;
    for (int j = 0; j < index.dim; j++)
    {
      src_data_norms[i] += src_data[i * index.dim + j] * src_data[i * index.dim + j];
    }
  }
  ///

  float *gpu_src_data;
  float *gpu_query;

  /// allocate memory for src_data and query in GPU, and copy data from CPU to GPU
  cudaMalloc(&gpu_src_data, index.record_num * index.dim * sizeof(float));
  cudaMalloc(&gpu_query, query_num * index.dim * sizeof(float));
  cudaMemcpy(gpu_src_data, src_data, index.record_num * index.dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_query, query, query_num * index.dim * sizeof(float), cudaMemcpyHostToDevice);

  /// project the query
  float *query_proj;
  cudaMallocHost(&query_proj, query_num * index.column_num * sizeof(float));
  for (int i = 0; i < query_num; i++)
  {
    for (int j = 0; j < index.column_num; j++)
    {
      query_proj[i * index.column_num + j] = 0;
      for (int k = 0; k < index.dim; k++)
      {
        query_proj[i * index.column_num + j] += query[i * index.dim + k] * index.pca_data[k * index.dim + j];
      }
    }
  }
  ///

  int proj_dim = index.column_num;
  float *proj_data = index.trans_data;

  /// precompute squared norms
  float *data_norms;
  cudaMallocHost(&data_norms, index.record_num * query_num * sizeof(float));
  // Calculate norms once and replicate across query_num columns
  float* temp_norms;
  cudaMallocHost(&temp_norms, index.record_num * sizeof(float));
  
  // Calculate norms for each data point
  for (int i = 0; i < index.record_num; i++) {
    temp_norms[i] = 0;
    for (int j = 0; j < proj_dim; j++) {
      temp_norms[i] += proj_data[i * proj_dim + j] * proj_data[i * proj_dim + j];
    }
  }
  
  // Copy to all query columns
  for (int q = 0; q < query_num; q++) {
    for (int i = 0; i < index.record_num; i++) {
      data_norms[q * index.record_num + i] = temp_norms[i];
    }
  }
  
  // Free temporary storage
  cudaFreeHost(temp_norms);
  ///

  /// warmup l2mm_fp16 function using a small dataset
  {
    int warmup_m = 1024;
    int warmup_n = 32;
    int warmup_k = proj_dim;
    
    // Use half precision for FP16 computation
    half *warmup_A, *warmup_B, *warmup_C;
    cudaMalloc(&warmup_A, warmup_m * warmup_k * sizeof(half));
    cudaMalloc(&warmup_B, warmup_n * warmup_k * sizeof(half));
    cudaMalloc(&warmup_C, warmup_m * warmup_n * sizeof(half));
    
    // Run the kernel once to warm up the GPU using FP16 version
    l2mm_fp16(warmup_m, warmup_n, warmup_k, warmup_A, warmup_B, warmup_C, 0);
    
    // Ensure warmup is complete
    cudaDeviceSynchronize();
    
    // Free resources
    cudaFree(warmup_A);
    cudaFree(warmup_B);
    cudaFree(warmup_C);
    
    printf("GPU warmup for FP16 computation completed\n");
  }
  

  /// compute l2 distances
  int m = index.record_num;
  int n = query_num;
  int k = proj_dim;
  /// allocate memory for A, B, C in GPU, and copy data from CPU to GPU
  // A: proj_data, B: query_proj, C: data_norms
  half *A, *B, *C;
  cudaMalloc(&A, m * k * sizeof(half));
  cudaMalloc(&B, n * k * sizeof(half));
  cudaMalloc(&C, m * n * sizeof(half));
  
  // Convert float data to half precision
  half *h_A, *h_B, *h_C;
  cudaMallocHost(&h_A, m * k * sizeof(half));
  cudaMallocHost(&h_B, n * k * sizeof(half));
  cudaMallocHost(&h_C, m * n * sizeof(half));
  
  // Convert proj_data to half
  for (int i = 0; i < m * k; i++) {
    h_A[i] = __float2half(proj_data[i]);
  }
  
  // Convert query_proj to half
  for (int i = 0; i < n * k; i++) {
    h_B[i] = __float2half(query_proj[i]);
  }
  
  // Convert data_norms to half
  for (int i = 0; i < m * n; i++) {
    h_C[i] = __float2half(data_norms[i]);
  }
  
  // Copy the half precision data to GPU
  cudaMemcpy(A, h_A, m * k * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(B, h_B, n * k * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(C, h_C, m * n * sizeof(half), cudaMemcpyHostToDevice);
  
  // Free host half precision buffers
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  
  // Record start event using both CUDA events
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Record start event
  cudaEventRecord(start, 0);
  
  // Use l2mm_fp16 instead of l2mm
  l2mm_fp16(m, n, k, A, B, C, 0);
  
  // Record stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  
  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("l2mm_fp16 kernel execution time: %.3f us\n", milliseconds * 1000.0f);
  
  /// prepare for reduce_min
  auto distances_num = m;
  auto group_num = (distances_num + reduce_group_size - 1) / reduce_group_size;

  int segment_size = reduce_group_size;
  int seg_num_per_query = distances_num / segment_size;
  int segment_num = seg_num_per_query * query_num;

  half* reduced_dists_per_query;
  int* reduced_ids_per_query;
  cudaMalloc(&reduced_dists_per_query, segment_num * sizeof(half));
  cudaMalloc(&reduced_ids_per_query, segment_num * sizeof(int));
  
  // Start timing using chrono
  auto start_time = std::chrono::high_resolution_clock::now();

  half_matrix_reduce(C, reduced_dists_per_query, reduced_ids_per_query, segment_size, segment_num, seg_num_per_query, 0);
  
  // launchComputeGroupMinimaKernel(C_float, distances_num, query_num, reduce_group_size, d_distances_per_query, d_idxs_per_query, 256);
  cudaStreamSynchronize(0);
  
  // End timing and calculate elapsed time
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  printf("Group minima kernel execution time: %.3f us\n", static_cast<float>(duration));
  ///

  segmented_sort_topk_pairs_fp16(reduced_dists_per_query, reduced_ids_per_query, query_num, group_num, phase1_topk);

  

  /// compute the phase1 topk
  half *phase1_distances_d;
  int *phase1_neighbors_d;
  cudaMalloc(&phase1_distances_d, query_num * phase1_topk * sizeof(half));
  cudaMalloc(&phase1_neighbors_d, query_num * phase1_topk * sizeof(int));

  extract_topk(reduced_dists_per_query, reduced_ids_per_query, phase1_distances_d, phase1_neighbors_d, query_num, group_num, phase1_topk, 0, true);


  /// allocate memory for phase1 topk
  half *phase1_distances;
  int *phase1_neighbors;
  cudaMallocHost(&phase1_distances, query_num * phase1_topk * sizeof(half));
  cudaMallocHost(&phase1_neighbors, query_num * phase1_topk * sizeof(int));
  ///

  /// copy phase1 topk to CPU
  cudaMemcpy(phase1_distances, phase1_distances_d, query_num * phase1_topk * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(phase1_neighbors, phase1_neighbors_d, query_num * phase1_topk * sizeof(int), cudaMemcpyDeviceToHost);
  ///

  ThreadPool pool(12);
  std::vector<std::future<int>> results;

  // allocate memory for phase2 topk
  int *phase2_neighbors;
  cudaMallocHost(&phase2_neighbors, query_num * phase2_topk * sizeof(int));

  // Start timing re-ranking phase
  auto rerank_start = std::chrono::high_resolution_clock::now();

  for (int i=0; i<query_num; i++)
  {
    results.emplace_back(
      pool.enqueue(
        re_rank, 
        src_data, 
        src_data_norms,
        query + i * index.dim,
        phase1_neighbors + i * phase1_topk,
        phase1_topk,
        index.record_num,
        index.dim,
        phase2_topk,
        phase2_neighbors + i * phase2_topk
      )
    );
  }

  for (auto && result: results)
  {
    result.get();
  }
  
  // End timing and calculate elapsed time
  auto rerank_end = std::chrono::high_resolution_clock::now();
  auto rerank_duration = std::chrono::duration_cast<std::chrono::microseconds>(rerank_end - rerank_start).count();
  printf("Re-ranking phase execution time: %.3f us\n", static_cast<float>(rerank_duration));

  float total_recall = 0.0f;
  for (size_t i = 0; i < query_num; i++)
  {
    // printf("Query %d \n", i);
    int total_found = 0;
    for (size_t j = 0; j < phase2_topk; j++)
    {
      // printf("j %d, %d\n", j, ground_truth_neighbors[i * phase2_topk + j]);
      for (size_t k = 0; k < phase2_topk; k++)
      {
        // printf("k %d, %d\n", k, phase2_neighbors[i * phase1_topk + k]);
        if (phase2_neighbors[i * phase2_topk + k] == ground_truth_neighbors[i * phase2_topk + j])
        {
          total_found++;
          break;
        }
      }
    }

    float recall = static_cast<float>(total_found) / phase2_topk;
    total_recall += recall;
    // printf("Recall for query %d: %.3f\n", i, recall);
  }
  
  float avg_recall = total_recall / query_num;
  printf("Average recall: %.4f\n", avg_recall);
  
  ///for each query, check how many ground truth neighbours are in the phase1 topk, first iterate all groud truth neighbours, check if it is in phase1 topk, then calculate the recall
  
  // float total_recall_2 = 0.0f;
  // for (size_t i = 0; i < query_num; i++)
  // {
  //   int total_found = 0;
  //   for (size_t j = 0; j < phase2_topk; j++)
  //   {
  //     for (size_t k = 0; k < phase1_topk; k++)
  //     {
  //       if (phase1_neighbors[i * phase1_topk + k] == ground_truth_neighbors[i * phase2_topk + j])
  //       {
  //         total_found++;
  //         break;
  //       }
  //     }
  //   }

  //   float recall = static_cast<float>(total_found) / phase2_topk;
  //   total_recall_2 += recall;
  //   // printf("Recall for query %d: %.3f\n", i, recall);
  // }
  
  // float avg_recall_2 = total_recall_2 / query_num;
  // printf("Average recall: %.4f\n", avg_recall_2);
  

  
  // Destroy the events
  // Clean up events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ///
}