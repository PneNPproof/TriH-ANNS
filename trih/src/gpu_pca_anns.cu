#include "gist.h"
#include "pca.h"

#include "gpu_pca_anns.cuh"
#include "l2mm.cuh"
#include "reduce_min.cuh"
#include "rerank.cuh"

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/util_type.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <vector>

cub::CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

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
  
  // // Convert results back to float for the rest of the pipeline
  // float *C_float;
  // cudaMalloc(&C_float, m * n * sizeof(float));
  
  // // Launch kernel to convert from half to float
  // dim3 block(256);
  // dim3 grid((m * n + block.x - 1) / block.x);
  
  // // Launch the kernel using the triple chevron syntax
  // half_to_float_kernel<<<grid, block>>>(C, C_float, m * n);
  
  // // Free the half precision result
  // cudaFree(C);
  

  /// prepare for reduce_min
  auto distances_num = m;
  auto group_num = (distances_num + reduce_group_size - 1) / reduce_group_size;

  std::vector<cub::DoubleBuffer<float>> distances_per_query(query_num);
  std::vector<cub::DoubleBuffer<int>> idxs_per_query(query_num);

  for (size_t i = 0; i < query_num; i++)
  {
    CubDebugExit(g_allocator.DeviceAllocate((void **)&distances_per_query[i].d_buffers[0], group_num * sizeof(float)));
    CubDebugExit(g_allocator.DeviceAllocate((void **)&distances_per_query[i].d_buffers[1], group_num * sizeof(float)));
    CubDebugExit(g_allocator.DeviceAllocate((void **)&idxs_per_query[i].d_buffers[0], group_num * sizeof(int)));
    CubDebugExit(g_allocator.DeviceAllocate((void **)&idxs_per_query[i].d_buffers[1], group_num * sizeof(int)));
  }
  float **d_distances_per_query;
  int **d_idxs_per_query;
  cudaMalloc(&d_distances_per_query, query_num * sizeof(float *));
  cudaMalloc(&d_idxs_per_query, query_num * sizeof(int *));
  for (size_t i = 0; i < query_num; i++)
  {
    cudaMemcpy(d_distances_per_query + i, &distances_per_query[i].d_buffers[distances_per_query[i].selector], sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idxs_per_query + i, &idxs_per_query[i].d_buffers[idxs_per_query[i].selector], sizeof(int *), cudaMemcpyHostToDevice);
  }

  // float *d_distances_per_query;
  // int *d_idxs_per_query;
  // cudaMalloc(&d_distances_per_query, group_num * query_num * sizeof(float));
  // cudaMalloc(&d_idxs_per_query, group_num * query_num * sizeof(int));

  

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

  // return ;

  /// compute the phase1 topk
  

  // size_t temp_storage_bytes = 0;
  // void* d_temp_storage      = nullptr;
  // CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, distances_per_query[0], idxs_per_query[0], group_num));
  // CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // // start_time = std::chrono::high_resolution_clock::now();
  // cudaEventRecord(start);

  // for (size_t i = 0; i < query_num; i++)
  // {
  //   CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, distances_per_query[i], idxs_per_query[i], group_num));
  // }

  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);

  // float milliseconds1 = 0;
  // cudaEventElapsedTime(&milliseconds1, start, stop);
  // printf("Phase1 topk execution time: %.3f us\n", milliseconds1 * 1000.0f);

  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  // printf("Phase1 topk execution time: %.3f us\n", static_cast<float>(duration));
  ///

  /// allocate memory for phase1 topk
  half *phase1_distances;
  int *phase1_neighbors;
  cudaMallocHost(&phase1_distances, query_num * phase1_topk * sizeof(half));
  cudaMallocHost(&phase1_neighbors, query_num * phase1_topk * sizeof(int));
  ///

  /// copy phase1 topk to CPU
  for (size_t i = 0; i < query_num; i++)
  {
    cudaMemcpy(phase1_distances + i * phase1_topk, reduced_dists_per_query + i * group_num, phase1_topk * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(phase1_neighbors + i * phase1_topk, reduced_ids_per_query + i * group_num, phase1_topk * sizeof(int), cudaMemcpyDeviceToHost);
  }
  ///

  /// initialize host memory for pdis_per_query and pidx_per_query using cudaMallocHost
  // float **pdis_per_query;
  // int **pidx_per_query;
  // cudaMallocHost(&pdis_per_query, query_num * sizeof(float *));
  // cudaMallocHost(&pidx_per_query, query_num * sizeof(int *));
  // for (size_t i = 0; i < query_num; i++)
  // {
  //   pdis_per_query[i] = distances_per_query[i].Current();
  //   pidx_per_query[i] = idxs_per_query[i].Current();
  // }
  // ///

  // /// copy pdis_per_query and pidx_per_query to GPU
  // cudaMemcpy(d_distances_per_query, pdis_per_query, query_num * sizeof(float *), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_idxs_per_query, pidx_per_query, query_num * sizeof(int *), cudaMemcpyHostToDevice);
  // ///

  // /// compute the exact distances
  // start_time = std::chrono::high_resolution_clock::now();
  // compute_exact_distances(pdis_per_query, pidx_per_query, gpu_src_data, gpu_query, query_num, index.record_num, index.dim, phase1_topk, 1024);
  
  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  // printf("Exact distances execution time: %.3f us\n", static_cast<float>(duration));


  ///for each query, check how many ground truth neighbours are in the phase1 topk, first iterate all groud truth neighbours, check if it is in phase1 topk, then calculate the recall
  
  float total_recall = 0.0f;
  for (size_t i = 0; i < query_num; i++)
  {
    int total_found = 0;
    for (size_t j = 0; j < phase2_topk; j++)
    {
      for (size_t k = 0; k < phase1_topk; k++)
      {
        if (phase1_neighbors[i * phase1_topk + k] == ground_truth_neighbors[i * phase2_topk + j])
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
  

  
  // Destroy the events
  // Clean up events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ///
}