#include "gist.h"
#include "pca.h"

#include "gpu_pca_anns.cuh"
#include "l2mm.cuh"
#include "reduce_min.cuh"
#include "rerank.cuh"
#include "rerank.h"
#include "sq.h"

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/util_type.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/cub.cuh>

#include <cublas_v2.h>

#include <vector>

extern sq_info *p_sq_info;
extern ThreadPool pool;
extern vector< future<int> > results;

// Define a kernel to convert half to float
__global__ void half_to_float_kernel(half* input, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __half2float(input[idx]);
  }
}

// Define a kernel to convert float to half
__global__ void float_to_half_kernel(float* input, half* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __float2half(input[idx]);
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

  // print query, index.pca_data
  // for (int i = 0; i < index.column_num; i++)
  // {
  //   printf("PCA Data %d\n", i);
  //   for (int j = 0; j < index.dim; j++)
  //   {
  //     printf("%.4f ", index.pca_data[i * index.dim + j]);
  //   }
  //   printf("\n");
  // }
  // for (int i = 0; i < query_num; i++)
  // {
  //   printf("Query %d\n", i);
  //   for (int j = 0; j < index.dim; j++)
  //   {
  //     printf("%.4f ", query[i * index.dim + j]);
  //   }
  //   printf("\n");
  // }
  

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

  // print query_proj
  // for (int i = 0; i < query_num; i++)
  // {
  //   printf("Query %d\n", i);
  //   for (int j = 0; j < index.column_num; j++)
  //   {
  //     printf("%.4f ", query_proj[i * index.column_num + j]);
  //   }
  //   printf("\n");
  // }

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

  // print h_A, h_B
  // for (int i = 0; i < 100; i++)
  // {
  //   printf("Data %d\n", i);
  //   for (int j = 0; j < k; j++)
  //   {
  //     printf("%.4f ", __half2float(h_A[i * k + j]));
  //   }
  //   printf("\n");
  // }
  // for (int i = 0; i < n; i++)
  // {
  //   printf("Query %d\n", i);
  //   for (int j = 0; j < k; j++)
  //   {
  //     printf("%.4f ", __half2float(h_B[i * k + j]));
  //   }
  //   printf("\n");
  // }

  // print h_C
  // for (int i = 0; i < n; i++)
  // {
  //   printf("Norm to Query %d\n", i);
  //   for (int j = 0; j < 10; j++)
  //   {
  //     printf("%.4f ", __half2float(h_C[i * m + j]));
  //   }
  //   printf("\n");
  // }
  
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

  // print C
  // half *C_h;
  // cudaMallocHost(&C_h, m * n * sizeof(half));
  // cudaMemcpy(C_h, C, m * n * sizeof(half), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < n; i++)
  // {
  //   printf("Dist to Query %d\n", i);
  //   for (int j = 0; j < 10; j++)
  //   {
  //     printf("%.4f ", __half2float(C_h[i * m + j]));
  //   }
  //   printf("\n");
  // }
  
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

  half_matrix_reduce(
    C, 
    reduced_dists_per_query, 
    reduced_ids_per_query, 
    segment_size, 
    segment_num, 
    seg_num_per_query, 
    0
  );
  
  // launchComputeGroupMinimaKernel(C_float, distances_num, query_num, reduce_group_size, d_distances_per_query, d_idxs_per_query, 256);
  cudaStreamSynchronize(0);

  // print reduced_ids_per_query
  // int *reduced_ids_per_query_h;
  // cudaMallocHost(&reduced_ids_per_query_h, segment_num * sizeof(int));
  // cudaMemcpy(reduced_ids_per_query_h, reduced_ids_per_query, segment_num * sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < query_num; i++)
  // {
  //   printf("Reduced ids for Query %d\n", i);
  //   for (int j = 0; j < seg_num_per_query; j++)
  //   {
  //     printf("%d ", reduced_ids_per_query_h[i * seg_num_per_query + j]);
  //   }
  //   printf("\n");
  // }

  // print reduced_dists_per_query
  // half *reduced_dists_per_query_h;
  // cudaMallocHost(&reduced_dists_per_query_h, segment_num * sizeof(half));
  // cudaMemcpy(reduced_dists_per_query_h, reduced_dists_per_query, segment_num * sizeof(half), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < query_num; i++)
  // {
  //   printf("Reduced dists for Query %d\n", i);
  //   for (int j = 0; j < seg_num_per_query; j++)
  //   {
  //     printf("%.4f ", __half2float(reduced_dists_per_query_h[i * seg_num_per_query + j]));
  //   }
  //   printf("\n");
  // }


  // End timing and calculate elapsed time
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  printf("Group minima kernel execution time: %.3f us\n", static_cast<float>(duration));
  ///

  segmented_sort_topk_pairs_fp16(reduced_dists_per_query, reduced_ids_per_query, query_num, group_num, phase1_topk);

  // print reduced_ids_per_query
  int *reduced_ids_per_query_h;
  cudaMallocHost(&reduced_ids_per_query_h, segment_num * sizeof(int));
  cudaMemcpy(reduced_ids_per_query_h, reduced_ids_per_query, segment_num * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < query_num; i++)
  {
    printf("Reduced ids for Query %d\n", i);
    for (int j = 0; j < seg_num_per_query; j++)
    {
      printf("%d ", reduced_ids_per_query_h[i * seg_num_per_query + j]);
    }
    printf("\n");
  }

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
    // results.emplace_back(
    //   pool.enqueue(
    //     re_rank, 
    //     src_data, 
    //     src_data_norms,
    //     query + i * index.dim,
    //     phase1_neighbors + i * phase1_topk,
    //     phase1_topk,
    //     index.record_num,
    //     index.dim,
    //     phase2_topk,
    //     phase2_neighbors + i * phase2_topk
    //   )
    // );


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


TrihAnnsWorker::TrihAnnsWorker
(
  float *full_dim_pca_data, // each column is a eigen vector(length dim), row-major stored
  float *base_dataset,
  float *pca_dataset,
  int data_num_,
  int max_queries_num_,
  int dim_,
  int pca_dim_,
  int reduce_group_size_,
  int reduce_group_num_,
  int phase1_topk_,
  int phase2_topk_,
  cudaStream_t work_stream_,
  int re_rank_thread_pool_size_
):
gemm_workspace(1024 * 1024 * 1024)
{
  alpha = ElementOutput(-2);
  beta = ElementOutput(1);

  rerank_thread_pool = new ThreadPool(re_rank_thread_pool_size_);

  data_num = data_num_;
  max_queries_num = max_queries_num_;
  dim = dim_;
  pca_dim = pca_dim_;
  reduce_group_size = reduce_group_size_;
  reduce_group_num = reduce_group_num_;
  phase1_topk = phase1_topk_;
  phase2_topk = phase2_topk_;
  work_stream = work_stream_;

  constexpr size_t alignment = 64;
  full_dim_pca_data_h = (float*)aligned_alloc(alignment, dim * dim * sizeof(float));
  base_dataset_h = (float*)aligned_alloc(alignment, data_num * dim * sizeof(float));
  pca_dataset_h = (float*)aligned_alloc(alignment, data_num * pca_dim * sizeof(float));

  cudaMemcpy(full_dim_pca_data_h, full_dim_pca_data, dim * dim * sizeof(float), cudaMemcpyHostToHost);
  cudaMemcpy(base_dataset_h, base_dataset, data_num * dim * sizeof(float), cudaMemcpyHostToHost);
  cudaMemcpy(pca_dataset_h, pca_dataset, data_num * pca_dim * sizeof(float), cudaMemcpyHostToHost);

  /// initialize pca_dim_pca_data_d using full_dim_pca_data_h
  auto pca_dim_pca_data_h = (float*)aligned_alloc(alignment, pca_dim * dim * sizeof(float));
  for (int i = 0; i < pca_dim; i++) // i-th eigen vector
  {
    for (int j = 0; j < dim; j++) // j-th element in i-th eigen vector
    {
      pca_dim_pca_data_h[i * dim + j] = full_dim_pca_data_h[j * dim + i];
    }
  }
  cudaMalloc(&pca_dim_pca_data_d, pca_dim * dim * sizeof(float));
  cudaMemcpy(pca_dim_pca_data_d, pca_dim_pca_data_h, pca_dim * dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaFreeHost(pca_dim_pca_data_h);
  ///

  /// transform pca_dim_pca_data_d to half precision
  cudaMalloc(&half_pca_dim_pca_data_d, pca_dim * dim * sizeof(half));
  float_to_half_kernel<<<(pca_dim * dim + 255) / 256, 256, 0, work_stream>>>(pca_dim_pca_data_d, half_pca_dim_pca_data_d, pca_dim * dim);
  cudaStreamSynchronize(work_stream);
  ///



  /// calculate squared norms for base_dataset_h
  base_dataset_norms_h = (float*)aligned_alloc(alignment, data_num * sizeof(float));
  for (int i = 0; i < data_num; i++)
  {
    base_dataset_norms_h[i] = 0;
    for (int j = 0; j < dim; j++)
    {
      base_dataset_norms_h[i] += base_dataset_h[i * dim + j] * base_dataset_h[i * dim + j];
    }
  }
  ///

  /// copy full_dim_pca_data_h to gpu
  cudaMalloc(&full_dim_pca_data_d, dim * dim * sizeof(float));
  cudaMemcpy(full_dim_pca_data_d, full_dim_pca_data_h, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
  ///

  /// allocate memory for batch_query_d, pca_batch_query_d, half_batch_query_d
  cudaMalloc(&batch_query_d, max_queries_num * dim * sizeof(float));
  cudaMalloc(&pca_batch_query_d, max_queries_num * pca_dim * sizeof(float));
  cudaMalloc(&half_batch_query_d, max_queries_num * dim * sizeof(half));
  ///

  /// allocate memory for alpha0_d, beta0_d
  cudaMalloc(&alpha0_d, sizeof(half));
  cudaMalloc(&beta0_d, sizeof(half));
  auto alpha0 = __float2half(1.0f);
  auto beta0 = __float2half(0.0f);
  cudaMemcpy(alpha0_d, &alpha0, sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(beta0_d, &beta0, sizeof(half), cudaMemcpyHostToDevice);
  ///

  /// calculate squared norms for pca_dataset_h
  float *pca_dataset_norms_h;
  cudaMallocHost(&pca_dataset_norms_h, data_num * max_queries_num * sizeof(float));
  // Calculate norms once and replicate across query_num columns
  float* temp_norms;
  cudaMallocHost(&temp_norms, data_num * sizeof(float));
  
  // Calculate norms for each data point
  for (int i = 0; i < data_num; i++) {
    temp_norms[i] = 0;
    for (int j = 0; j < pca_dim; j++) {
      temp_norms[i] += pca_dataset_h[i * pca_dim + j] * pca_dataset_h[i * pca_dim + j];
    }
  }

  for (int q = 0; q < max_queries_num; q++) {
    for (int i = 0; i < data_num; i++) {
      pca_dataset_norms_h[q * data_num + i] = temp_norms[i];
    }
  }
  cudaFreeHost(temp_norms);
  ///

  /// allocate half_dists_d and half_pca_queries_d
  cudaMalloc(&half_dists_d, data_num * max_queries_num * sizeof(half));
  cudaMalloc(&half_pca_queries_d, max_queries_num * pca_dim * sizeof(half));
  ///
  
  /// transform pca_dataset_h, pca_dataset_norms_h to half precision and copy to half_pca_dataset_d, half_pca_dataset_norms_d
  cudaMalloc(&half_pca_dataset_d, data_num * pca_dim * sizeof(half));
  cudaMalloc(&half_pca_dataset_norms_d, data_num * max_queries_num * sizeof(half));
  half *half_pca_dataset_h, *half_pca_dataset_norms_h;
  cudaMallocHost(&half_pca_dataset_h, data_num * pca_dim * sizeof(half));
  cudaMallocHost(&half_pca_dataset_norms_h, data_num * max_queries_num * sizeof(half));
  for (int i = 0; i < data_num * pca_dim; i++) {
    half_pca_dataset_h[i] = __float2half(pca_dataset_h[i]);
  }
  for (int i = 0; i < data_num * max_queries_num; i++) {
    half_pca_dataset_norms_h[i] = __float2half(pca_dataset_norms_h[i]);
  }
  cudaMemcpy(half_pca_dataset_d, half_pca_dataset_h, data_num * pca_dim * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(half_pca_dataset_norms_d, half_pca_dataset_norms_h, data_num * max_queries_num * sizeof(half), cudaMemcpyHostToDevice);
  cudaFreeHost(half_pca_dataset_h);
  cudaFreeHost(half_pca_dataset_norms_h);
  ///

  /// allocate memory for alpha_d, beta_d, alpha1_d
  cudaMalloc(&alpha_d, sizeof(half));
  cudaMalloc(&beta_d, sizeof(half));
  cudaMalloc(&alpha1_d, sizeof(half));
  auto alpha = __float2half(-2.0f);
  auto beta = __float2half(0.0f);
  auto alpha1 = __float2half(1.0f);
  cudaMemcpy(alpha_d, &alpha, sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(beta_d, &beta, sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(alpha1_d, &alpha1, sizeof(half), cudaMemcpyHostToDevice);
  ///

  /// create cublas handle
  cublasCreate(&handle);
  cublasSetStream(handle, work_stream);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  ///

  /// allocate memory for reduced_dists_per_query_d, reduced_ids_per_query_d
  cudaMalloc(&reduced_dists_per_query_d, reduce_group_num * max_queries_num * sizeof(half));
  cudaMalloc(&reduced_ids_per_query_d, reduce_group_num * max_queries_num * sizeof(int));
  ///

  /// allocate memory for phase1_distances_d, phase1_ids_d
  cudaMalloc(&phase1_distances_d, max_queries_num * phase1_topk * sizeof(half));
  cudaMalloc(&phase1_ids_d, max_queries_num * phase1_topk * sizeof(int));
  ///

  /// initialize segments_offsets_d, temp_storage_d, temp_storage_bytes
  int *segments_offsets_h;
  cudaMallocHost(&segments_offsets_h, (max_queries_num+1) * sizeof(int));
  for (int i = 0; i < max_queries_num+1; i++) {
    segments_offsets_h[i] = i * reduce_group_num;
  }
  cudaMalloc(&segments_offsets_d, (max_queries_num+1) * sizeof(int));
  cudaMemcpy(segments_offsets_d, segments_offsets_h, (max_queries_num+1) * sizeof(int), cudaMemcpyHostToDevice);

  // initialize reduced_dists_per_query_d, reduced_ids_per_query_d to 0
  cudaMemset(reduced_dists_per_query_d, 0, reduce_group_num * max_queries_num * sizeof(half));
  cudaMemset(reduced_ids_per_query_d, 0, reduce_group_num * max_queries_num * sizeof(int));

  temp_storage_bytes = 1024 * 1024 * 1024;

  printf("temp_storage_bytes in construction: %d\n", temp_storage_bytes);

  cudaMalloc(&temp_storage_d, temp_storage_bytes);
  ///

  /// allocate memory for phase1_distances_h, phase1_ids_h and phase2_ids_h
  cudaMallocHost(&phase1_distances_h, max_queries_num * phase1_topk * sizeof(half));
  cudaMallocHost(&phase1_ids_h, max_queries_num * phase1_topk * sizeof(int));
  cudaMallocHost(&phase2_ids_h, max_queries_num * phase2_topk * sizeof(int));
  ///

}

TrihAnnsWorker::TrihAnnsWorker
(
  TrihAnnsWorker &other,
  cudaStream_t work_stream_
):
  alpha(other.alpha),
  beta(other.beta),
  gemm_workspace(1024 * 1024 * 1024),
  full_dim_pca_data_d(other.full_dim_pca_data_d),
  full_dim_pca_data_h(other.full_dim_pca_data_h),
  pca_dim_pca_data_d(other.pca_dim_pca_data_d),
  half_pca_dim_pca_data_d(other.half_pca_dim_pca_data_d),
  alpha0_d(other.alpha0_d),
  beta0_d(other.beta0_d),
  half_pca_dataset_d(other.half_pca_dataset_d),
  half_pca_dataset_norms_d(other.half_pca_dataset_norms_d),
  alpha_d(other.alpha_d),
  beta_d(other.beta_d),
  alpha1_d(other.alpha1_d),
  segments_offsets_d(other.segments_offsets_d),
  temp_storage_bytes(other.temp_storage_bytes),
  base_dataset_h(other.base_dataset_h),
  pca_dataset_h(other.pca_dataset_h),
  base_dataset_norms_h(other.base_dataset_norms_h),
  rerank_thread_pool(other.rerank_thread_pool),
  data_num(other.data_num),
  max_queries_num(other.max_queries_num),
  dim(other.dim),
  pca_dim(other.pca_dim),
  reduce_group_size(other.reduce_group_size),
  reduce_group_num(other.reduce_group_num),
  phase1_topk(other.phase1_topk),
  phase2_topk(other.phase2_topk),
  work_stream(work_stream_)
{
  /// allocate memory for batch_query_d, pca_batch_query_d, half_batch_query_d
  cudaMalloc(&batch_query_d, max_queries_num * dim * sizeof(float));
  cudaMalloc(&pca_batch_query_d, max_queries_num * pca_dim * sizeof(float));
  cudaMalloc(&half_batch_query_d, max_queries_num * dim * sizeof(half));
  ///

  /// allocate half_dists_d and half_pca_queries_d
  cudaMalloc(&half_dists_d, data_num * max_queries_num * sizeof(half));
  cudaMalloc(&half_pca_queries_d, max_queries_num * pca_dim * sizeof(half));
  ///

  /// create cublas handle
  cublasCreate(&handle);
  cublasSetStream(handle, work_stream);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  ///

  /// allocate memory for reduced_dists_per_query_d, reduced_ids_per_query_d
  cudaMalloc(&reduced_dists_per_query_d, reduce_group_num * max_queries_num * sizeof(half));
  cudaMalloc(&reduced_ids_per_query_d, reduce_group_num * max_queries_num * sizeof(int));
  ///

  /// allocate memory for phase1_distances_d, phase1_ids_d
  cudaMalloc(&phase1_distances_d, max_queries_num * phase1_topk * sizeof(half));
  cudaMalloc(&phase1_ids_d, max_queries_num * phase1_topk * sizeof(int));
  ///

  cudaMalloc(&temp_storage_d, temp_storage_bytes);

  /// allocate memory for phase1_distances_h, phase1_ids_h and phase2_ids_h
  cudaMallocHost(&phase1_distances_h, max_queries_num * phase1_topk * sizeof(half));
  cudaMallocHost(&phase1_ids_h, max_queries_num * phase1_topk * sizeof(int));
  cudaMallocHost(&phase2_ids_h, max_queries_num * phase2_topk * sizeof(int));
  ///
}

int* TrihAnnsWorker::batch_query_search
(
  pca_index &index,
  float *batch_query,
  int batch_query_num,
  int *ground_truth_neighbors
)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  /// copy batch_query to gpu and project it and transform to half precision
  cudaMemcpyAsync(batch_query_d, batch_query, batch_query_num * dim * sizeof(float), cudaMemcpyHostToDevice, work_stream);

  // project the query,
  // pca_dim_pca_data_d(dim, pca_dim),  
  // batch_query_d(dim, batch_query_num), 
  // pca_batch_query_d(pca_dim, batch_query_num)

  // covert batch_query to half precision
  float_to_half_kernel<<<(batch_query_num * dim + 255) / 256, 256, 0, work_stream>>>
    (batch_query_d, half_batch_query_d, batch_query_num * dim);
 
  cublasGemmEx(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    pca_dim, batch_query_num, dim,
    alpha0_d,
    half_pca_dim_pca_data_d, CUDA_R_16F, dim,
    half_batch_query_d, CUDA_R_16F, dim,
    beta0_d,
    half_pca_queries_d, CUDA_R_16F, pca_dim,
    CUDA_R_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );

  cutlass::gemm::GemmCoord problem_size(data_num, batch_query_num, pca_dim);

  int split_k_slices = 1;

  cutlass::layout::RowMajor tensor_A_layout(pca_dim);
  cutlass::layout::ColumnMajor tensor_B_layout(pca_dim);
  cutlass::layout::ColumnMajor tensor_C_layout(data_num);
  cutlass::layout::ColumnMajor tensor_D_layout(data_num);

  cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> tensor_A(reinterpret_cast<cutlass::half_t*>(half_pca_dataset_d), tensor_A_layout);
  cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_B(reinterpret_cast<cutlass::half_t*>(half_pca_queries_d), tensor_B_layout);
  cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_C(reinterpret_cast<cutlass::half_t*>(half_pca_dataset_norms_d), tensor_C_layout);
  cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_D(reinterpret_cast<cutlass::half_t*>(half_dists_d), tensor_D_layout);

  typename Gemm::Arguments arguments{
    problem_size,
    tensor_A,
    tensor_B,
    tensor_C,
    tensor_D,
    {alpha, beta},
    split_k_slices
  };

  gemm_op.initialize(arguments, gemm_workspace.get(), work_stream);

  gemm_op(work_stream);

  /// reduce half_dists_d into reduced_dists_per_query_d and reduced_ids_per_query_d
  half_matrix_reduce(
    half_dists_d, 
    reduced_dists_per_query_d, 
    reduced_ids_per_query_d, 
    reduce_group_size, 
    reduce_group_num * batch_query_num, 
    reduce_group_num, 
    work_stream
  );
  ///

  // printf("segmented_sort_topk_pairs_fp16\n");
  /// extract topk from reduced_dists_per_query_d and reduced_ids_per_query_d
  cub::DeviceSegmentedRadixSort::SortPairs(
    nullptr, 
    temp_storage_bytes, 
    reduced_dists_per_query_d, 
    reduced_dists_per_query_d, 
    reduced_ids_per_query_d, 
    reduced_ids_per_query_d, 
    reduce_group_num * batch_query_num, 
    batch_query_num, 
    segments_offsets_d, 
    segments_offsets_d + 1, 
    0, 
    sizeof(float) * 8, 
    work_stream
  );

  cub::DeviceSegmentedRadixSort::SortPairs(
    temp_storage_d, 
    temp_storage_bytes, 
    reduced_dists_per_query_d, 
    reduced_dists_per_query_d, 
    reduced_ids_per_query_d, 
    reduced_ids_per_query_d, 
    reduce_group_num * batch_query_num, 
    batch_query_num, 
    segments_offsets_d, 
    segments_offsets_d + 1, 
    0, 
    sizeof(float) * 8, 
    work_stream
  );

  extract_topk(
    reduced_dists_per_query_d, 
    reduced_ids_per_query_d, 
    phase1_distances_d, 
    phase1_ids_d, 
    batch_query_num, 
    reduce_group_num, 
    phase1_topk, 
    work_stream, 
    false
  );
  ///

  // printf("re-rank\n");
  /// copy phase1_distances_d, phase1_ids_d to phase1_distances_h, phase1_ids_h
  cudaMemcpyAsync(phase1_distances_h, phase1_distances_d, batch_query_num * phase1_topk * sizeof(half), cudaMemcpyDeviceToHost, work_stream);
  cudaMemcpyAsync(phase1_ids_h, phase1_ids_d, batch_query_num * phase1_topk * sizeof(int), cudaMemcpyDeviceToHost, work_stream);
  ///

  cudaStreamSynchronize(work_stream);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  printf("batch_query_search gpu execution time: %.3f us\n", static_cast<float>(duration));

  auto rerank_start = std::chrono::high_resolution_clock::now();
  /// assign rerank tasks
  std::vector<std::future<int>> results;

  for (int i=0; i<batch_query_num; i++)
  {
    // printf("rerank query %d\n", i);
    results.emplace_back(
      rerank_thread_pool->enqueue(
        re_rank2, 
        base_dataset_h, 
        base_dataset_norms_h,
        batch_query + i * dim,
        phase1_ids_h + i * phase1_topk,
        phase1_topk,
        data_num,
        dim,
        phase2_topk,
        phase2_ids_h + i * phase2_topk
      )
    );



    // float *distances = new float[phase1_topk];
    // int *neighbors = new int[phase1_topk];

    // for(int k=0; k<phase1_topk; k++)
    //   distances[k] = __half2float(phase1_distances_h[phase1_topk * i + k]);
    // memcpy(neighbors, phase1_ids_h + i * phase1_topk, sizeof(int) * phase1_topk);

    // results.emplace_back(
    //   rerank_thread_pool->enqueue(
    //     re_rank, 
    //     batch_query + i * index.dim,
    //     std::ref(index),
    //     p_sq_info,
    //     distances,
    //     neighbors,
    //     phase1_topk,
    //     phase2_topk,
    //     phase2_ids_h + i * phase2_topk
    //   )
    // );


  }
  ///

  for (auto && result: results)
  {
    result.get();
  }

  auto rerank_end = std::chrono::high_resolution_clock::now();
  auto rerank_duration = std::chrono::duration_cast<std::chrono::microseconds>(rerank_end - rerank_start).count();
  printf("Re-ranking phase execution time: %.3f us\n", static_cast<float>(rerank_duration));

  float total_recall = 0.0f;
  for (size_t i = 0; i < batch_query_num; i++)
  {
    int total_found = 0;
    for (size_t j = 0; j < phase2_topk; j++)
    {
      for (size_t k = 0; k < phase2_topk; k++)
      {
        // printf("k %d, %d\n", k, phase2_neighbors[i * phase1_topk + k]);
        if (phase2_ids_h[i * phase2_topk + k] == ground_truth_neighbors[i * phase2_topk + j])
        {
          total_found++;
          break;
        }
      }
    }

    float recall = static_cast<float>(total_found) / phase2_topk;
    total_recall += recall;
    
  }
  
  float avg_recall = total_recall / batch_query_num;
  printf("Average recall: %.4f\n", avg_recall);

  return phase1_ids_h;

}