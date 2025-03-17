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

  /// warmup l2mm function using a small dataset
  {
    int warmup_m = 1024;
    int warmup_n = 32;
    int warmup_k = proj_dim;
    
    float *warmup_A, *warmup_B, *warmup_C;
    cudaMalloc(&warmup_A, warmup_m * warmup_k * sizeof(float));
    cudaMalloc(&warmup_B, warmup_n * warmup_k * sizeof(float));
    cudaMalloc(&warmup_C, warmup_m * warmup_n * sizeof(float));
    
    // Run the kernel once to warm up the GPU
    l2mm(warmup_m, warmup_n, warmup_k, warmup_A, warmup_B, warmup_C, 0);
    
    // Ensure warmup is complete
    cudaDeviceSynchronize();
    
    // Free resources
    cudaFree(warmup_A);
    cudaFree(warmup_B);
    cudaFree(warmup_C);
    
    printf("GPU warmup completed\n");
  }

  /// compute l2 distances
  int m = index.record_num;
  int n = query_num;
  int k = proj_dim;
  /// allocate memory for A, B, C in GPU, and copy data from CPU to GPU
  // A: proj_data, B: query_proj, C: data_norms
  float *A, *B, *C;
  cudaMalloc(&A, m * k * sizeof(float));
  cudaMalloc(&B, n * k * sizeof(float));
  cudaMalloc(&C, m * n * sizeof(float));
  
  cudaMemcpy(A, proj_data, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B, query_proj, n * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(C, data_norms, m * n * sizeof(float), cudaMemcpyHostToDevice);
  // allocate workspace
  // void *workspace;
  // size_t workspaceSize = (size_t)1024 * 1024 * 1024 * 8;
  // cudaMalloc(&workspace, workspaceSize);
  
  // Record start event using both CUDA events and C++ chrono
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Record start event
  cudaEventRecord(start, 0);
  
  l2mm(m, n, k, A, B, C, 0);
  
  // Record stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  
  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("l2mm kernel execution time: %.3f us\n", milliseconds * 1000.0f);
  
  
  
  
  // copy C to CPU and print the first 10 elements
  // float *l2_distances;
  // cudaMallocHost(&l2_distances, m * n * sizeof(float));
  // printf("Copying C to CPU...\n");
  // cudaMemcpy(l2_distances, C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 10; i++) {
  //   printf("l2_distances[%d]: %.3f\n", i, l2_distances[i]);
  // }
  

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

  // Start timing using chrono
  auto start_time = std::chrono::high_resolution_clock::now();
  
  launchComputeGroupMinimaKernel(C, distances_num, query_num, reduce_group_size, d_distances_per_query, d_idxs_per_query, 256);
  cudaStreamSynchronize(0);
  
  // End timing and calculate elapsed time
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  printf("Group minima kernel execution time: %.3f us\n", static_cast<float>(duration));
  ///

  // return ;

  /// compute the phase1 topk
  

  size_t temp_storage_bytes = 0;
  void* d_temp_storage      = nullptr;
  CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, distances_per_query[0], idxs_per_query[0], group_num));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // start_time = std::chrono::high_resolution_clock::now();
  cudaEventRecord(start);

  for (size_t i = 0; i < query_num; i++)
  {
    CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, distances_per_query[i], idxs_per_query[i], group_num));
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds1 = 0;
  cudaEventElapsedTime(&milliseconds1, start, stop);
  printf("Phase1 topk execution time: %.3f us\n", milliseconds1 * 1000.0f);

  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  // printf("Phase1 topk execution time: %.3f us\n", static_cast<float>(duration));
  ///

  /// allocate memory for phase1 topk
  float *phase1_distances;
  int *phase1_neighbors;
  cudaMallocHost(&phase1_distances, query_num * phase1_topk * sizeof(float));
  cudaMallocHost(&phase1_neighbors, query_num * phase1_topk * sizeof(int));
  ///

  /// copy phase1 topk to CPU
  for (size_t i = 0; i < query_num; i++)
  {
    cudaMemcpy(phase1_distances + i * phase1_topk, distances_per_query[i].Current(), phase1_topk * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(phase1_neighbors + i * phase1_topk, idxs_per_query[i].Current(), phase1_topk * sizeof(int), cudaMemcpyDeviceToHost);
  }
  ///

  /// initialize host memory for pdis_per_query and pidx_per_query using cudaMallocHost
  float **pdis_per_query;
  int **pidx_per_query;
  cudaMallocHost(&pdis_per_query, query_num * sizeof(float *));
  cudaMallocHost(&pidx_per_query, query_num * sizeof(int *));
  for (size_t i = 0; i < query_num; i++)
  {
    pdis_per_query[i] = distances_per_query[i].Current();
    pidx_per_query[i] = idxs_per_query[i].Current();
  }
  ///

  /// copy pdis_per_query and pidx_per_query to GPU
  cudaMemcpy(d_distances_per_query, pdis_per_query, query_num * sizeof(float *), cudaMemcpyHostToDevice);
  cudaMemcpy(d_idxs_per_query, pidx_per_query, query_num * sizeof(int *), cudaMemcpyHostToDevice);
  ///

  /// compute the exact distances
  start_time = std::chrono::high_resolution_clock::now();
  compute_exact_distances(pdis_per_query, pidx_per_query, gpu_src_data, gpu_query, query_num, index.record_num, index.dim, phase1_topk, 1024);
  
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  printf("Exact distances execution time: %.3f us\n", static_cast<float>(duration));


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