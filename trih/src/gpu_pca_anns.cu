#include "pca.h"
#include "gpu_pca_anns.cuh"
#include "l2mm.cuh"
#include "reduce_min.cuh"
#include "rerank.cuh"
#include "rerank.h"
#include "sq.h"
#include "BS_thread_pool.hpp"
#include "utils.h"

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/util_type.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/cub.cuh>
#include <cublas_v2.h>

#include <vector>

std::mutex TrihAnnsWorker::thread_pool_mutex;

extern sq_info *p_sq_info;
extern BS::thread_pool<> *rr_pool;

std::mutex rr_pool_mutex;
// extern BS::thread_pool<> rerank_task_scheduler_pool;

extern int file_ind;

#define CHECK_CUDA_ERROR(err)                                                                 \
  do                                                                                          \
  {                                                                                           \
    if (err != cudaSuccess)                                                                   \
    {                                                                                         \
      fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)

// Function to convert cublasStatus_t to string
const char *cublasGetErrorString(cublasStatus_t status)
{
  switch (status)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  // Add more cases as needed based on your cuBLAS version/usage
  default:
    return "Unknown cuBLAS error";
  }
}
#define CHECK_CUBLAS(status)                                               \
  do                                                                       \
  {                                                                        \
    cublasStatus_t CUB_err = (status);                                     \
    if (CUB_err != CUBLAS_STATUS_SUCCESS)                                  \
    {                                                                      \
      fprintf(stderr, "cuBLAS error in %s:%d: %s (%d)\n",                  \
              __FILE__, __LINE__, cublasGetErrorString(CUB_err), CUB_err); \
      /* You might want to handle the error more gracefully than exit */   \
      /* For example, return an error code or throw an exception */        \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

// Define a kernel to convert half to float
__global__ void half_to_float_kernel(half *input, float *output, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = __half2float(input[idx]);
  }
}

// Define a kernel to convert float to half
__global__ void float_to_half_kernel(float *input, half *output, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = __float2half(input[idx]);
  }
}

TrihAnnsWorker::TrihAnnsWorker(
    pca_index &index,
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
    int re_rank_thread_pool_size_) : gemm_workspace(1024 * 1024 * 1024),
                                     full_dim_pca_data_h(full_dim_pca_data),
                                     base_dataset_h(base_dataset),
                                     pca_dataset_h(pca_dataset),
                                     data_num(data_num_),
                                     max_queries_num(max_queries_num_),
                                     dim(dim_),
                                     pca_dim(pca_dim_),
                                     reduce_group_size(reduce_group_size_),
                                     reduce_group_num(reduce_group_num_),
                                     phase1_topk(phase1_topk_),
                                     phase2_topk(phase2_topk_),
                                     work_stream(work_stream_)
{
  alpha = ElementOutput(-2);
  beta = ElementOutput(1);

  /// copy to alpha_d, beta_d
  CHECK_CUDA_ERROR(cudaMalloc(&alpha_d, sizeof(ElementOutput)));
  CHECK_CUDA_ERROR(cudaMalloc(&beta_d, sizeof(ElementOutput)));
  CHECK_CUDA_ERROR(cudaMemcpy(alpha_d, &alpha, sizeof(ElementOutput), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(beta_d, &beta, sizeof(ElementOutput), cudaMemcpyHostToDevice));
  ///

  // quant_queries_h = static_cast<uint8_t*>(aligned_alloc(64, (dim - pca_dim) * max_queries_num_ * sizeof(uint8_t)));

  // aligned_malloc_host(
  //     (void **)&quant_queries_h,
  //     (dim - pca_dim) * max_queries_num_ * sizeof(uint8_t),
  //     64);

  sq_info_h = std::make_shared<std::vector<sq_info>>();
  sq_info_h->reserve(data_num_);
  for (int i = 0; i < data_num_; i++)
  {
    sq_info_h->emplace_back(dim_);
  }

  gen_sq_info(
      index.trans_data_remain,
      dim - pca_dim,
      8,
      data_num,
      sq_info_h->data(),
      -1);

  constexpr size_t alignment = 64;
  // full_dim_pca_data_h = (float *)aligned_alloc(alignment, dim * dim * sizeof(float));
  // base_dataset_h = (float *)aligned_alloc(alignment, data_num * dim * sizeof(float));
  // pca_dataset_h = (float *)aligned_alloc(alignment, data_num * pca_dim * sizeof(float));

  // CHECK_CUDA_ERROR(cudaMemcpy(full_dim_pca_data_h, full_dim_pca_data, dim * dim * sizeof(float), cudaMemcpyHostToHost));
  // CHECK_CUDA_ERROR(cudaMemcpy(base_dataset_h, base_dataset, data_num * dim * sizeof(float), cudaMemcpyHostToHost));
  // CHECK_CUDA_ERROR(cudaMemcpy(pca_dataset_h, pca_dataset, data_num * pca_dim * sizeof(float), cudaMemcpyHostToHost));

  /// initialize pca_dim_pca_data_d using full_dim_pca_data_h
  auto pca_dim_pca_data_h = (float *)aligned_alloc(alignment, pca_dim * dim * sizeof(float));
  auto remain_dim_pca_data_h = (float *)aligned_alloc(alignment, (dim - pca_dim) * dim * sizeof(float));
  for (int i = 0; i < pca_dim; i++) // i-th eigen vector
  {
    for (int j = 0; j < dim; j++) // j-th element in i-th eigen vector
    {
      pca_dim_pca_data_h[i * dim + j] = full_dim_pca_data_h[j * dim + i];
    }
  }
  for (int i = 0; i < dim - pca_dim; i++) // i-th eigen vector
  {
    for (int j = 0; j < dim; j++) // j-th element in i-th eigen vector
    {
      remain_dim_pca_data_h[i * dim + j] = full_dim_pca_data_h[j * dim + pca_dim + i];
    }
  }
  CHECK_CUDA_ERROR(cudaMalloc(&pca_dim_pca_data_d, pca_dim * dim * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(pca_dim_pca_data_d, pca_dim_pca_data_h, pca_dim * dim * sizeof(float), cudaMemcpyHostToDevice));
  // cudaFreeHost(pca_dim_pca_data_h);
  free(pca_dim_pca_data_h);

  CHECK_CUDA_ERROR(cudaMalloc(&remain_dim_pca_data_d, (dim - pca_dim) * dim * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(remain_dim_pca_data_d, remain_dim_pca_data_h, (dim - pca_dim) * dim * sizeof(float), cudaMemcpyHostToDevice));
  // cudaFreeHost(remain_dim_pca_data_h);
  free(remain_dim_pca_data_h);
  ///

  /// transform pca_dim_pca_data_d to half precision
  CHECK_CUDA_ERROR(cudaMalloc(&half_pca_dim_pca_data_d, pca_dim * dim * sizeof(half)));
  float_to_half_kernel<<<(pca_dim * dim + 255) / 256, 256, 0, work_stream>>>(pca_dim_pca_data_d, half_pca_dim_pca_data_d, pca_dim * dim);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(work_stream));
  ///

  /// calculate squared norms for base_dataset_h
  base_dataset_norms_h = (float *)aligned_alloc(alignment, data_num * sizeof(float));
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
  CHECK_CUDA_ERROR(cudaMalloc(&full_dim_pca_data_d, dim * dim * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(full_dim_pca_data_d, full_dim_pca_data_h, dim * dim * sizeof(float), cudaMemcpyHostToDevice));
  ///

  /// allocate memory for batch_query_d, pca_batch_query_d, half_batch_query_d
  CHECK_CUDA_ERROR(cudaMalloc(&batch_query_d, max_queries_num * dim * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&remain_batch_query_d, max_queries_num * (dim - pca_dim) * sizeof(float)));
  // CHECK_CUDA_ERROR(cudaMallocHost(&remain_batch_query_h, max_queries_num * (dim - pca_dim) * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&pca_batch_query_d, max_queries_num * pca_dim * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&half_batch_query_d, max_queries_num * dim * sizeof(half)));
  ///

  /// Initialize batch_query_d, pca_batch_query_d, half_batch_query_d to zero
  CHECK_CUDA_ERROR(cudaMemset(batch_query_d, 0, max_queries_num * dim * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(pca_batch_query_d, 0, max_queries_num * pca_dim * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(half_batch_query_d, 0, max_queries_num * dim * sizeof(half)));
  ///

  // printf("flag 11\n");

  /// allocate memory for alpha0_d, beta0_d
  CHECK_CUDA_ERROR(cudaMalloc(&alpha0_d, sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&beta0_d, sizeof(half)));
  auto alpha0 = __float2half(1.0f);
  auto beta0 = __float2half(0.0f);
  CHECK_CUDA_ERROR(cudaMemcpy(alpha0_d, &alpha0, sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(beta0_d, &beta0, sizeof(half), cudaMemcpyHostToDevice));
  ///

  /// allocate memory for falpha_d, fbeta_d
  CHECK_CUDA_ERROR(cudaMalloc(&falpha_d, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&fbeta_d, sizeof(float)));
  auto falpha = 1.0f;
  auto fbeta = 0.0f;
  CHECK_CUDA_ERROR(cudaMemcpy(falpha_d, &falpha, sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(fbeta_d, &fbeta, sizeof(float), cudaMemcpyHostToDevice));

  /// allocate half_dists_d and half_pca_queries_d
  CHECK_CUDA_ERROR(cudaMalloc(&half_dists_d, data_num * max_queries_num * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&half_pca_queries_d, max_queries_num * pca_dim * sizeof(half)));
  ///

  /// intialize half_dists_d and half_pca_queries_d to zero
  CHECK_CUDA_ERROR(cudaMemset(half_dists_d, 0, data_num * max_queries_num * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMemset(half_pca_queries_d, 0, max_queries_num * pca_dim * sizeof(half)));
  ///

  /// transform pca_dataset_h, pca_dataset_norms_h to half precision and copy to half_pca_dataset_d, half_pca_dataset_norms_d
  CHECK_CUDA_ERROR(cudaMalloc(&half_pca_dataset_d, data_num * pca_dim * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&half_pca_dataset_norms_d, data_num * max_queries_num * sizeof(half)));
  half *half_pca_dataset_h, *half_pca_dataset_norms_h;
  CHECK_CUDA_ERROR(cudaMallocHost(&half_pca_dataset_h, data_num * pca_dim * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMallocHost(&half_pca_dataset_norms_h, data_num * max_queries_num * sizeof(half)));
  for (size_t i = 0; i < data_num * pca_dim; i++)
  {
    half_pca_dataset_h[i] = __float2half(pca_dataset_h[i]);
  }

  /// calculate squared norms for pca_dataset_h
  float *pca_dataset_norms_h;
  CHECK_CUDA_ERROR(cudaMallocHost(&pca_dataset_norms_h, data_num * max_queries_num * sizeof(float)));
  // printf("flag 2\n");
  // Calculate norms once and replicate across query_num columns
  float *temp_norms;
  CHECK_CUDA_ERROR(cudaMallocHost(&temp_norms, data_num * sizeof(float)));

  // Calculate norms for each data point
  for (int i = 0; i < data_num; i++)
  {
    temp_norms[i] = 0;
    for (int j = 0; j < pca_dim; j++)
    {
      temp_norms[i] += __half2float(half_pca_dataset_h[i * pca_dim + j]) * __half2float(half_pca_dataset_h[i * pca_dim + j]);
      // temp_norms[i] += pca_dataset_h[i * pca_dim + j] * pca_dataset_h[i * pca_dim + j];
    }
  }
  for (int q = 0; q < max_queries_num; q++)
  {
    for (int i = 0; i < data_num; i++)
    {
      pca_dataset_norms_h[q * data_num + i] = temp_norms[i];
    }
  }
  CHECK_CUDA_ERROR(cudaFreeHost(temp_norms));
  ///

  for (size_t i = 0; i < data_num * max_queries_num; i++)
  {
    half_pca_dataset_norms_h[i] = __float2half(pca_dataset_norms_h[i]);
  }
  CHECK_CUDA_ERROR(cudaMemcpy(half_pca_dataset_d, half_pca_dataset_h, data_num * pca_dim * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(half_pca_dataset_norms_d, half_pca_dataset_norms_h, data_num * max_queries_num * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaFreeHost(half_pca_dataset_h));
  CHECK_CUDA_ERROR(cudaFreeHost(half_pca_dataset_norms_h));
  CHECK_CUDA_ERROR(cudaFreeHost(pca_dataset_norms_h));
  ///

  /// create cublas handle
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetStream(handle, work_stream));
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  ///

  /// allocate memory for reduced_dists_per_query_d, reduced_ids_per_query_d
  CHECK_CUDA_ERROR(cudaMalloc(&reduced_dists_per_query_d, reduce_group_num * max_queries_num * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&reduced_ids_per_query_d, reduce_group_num * max_queries_num * sizeof(int)));
  ///

  /// intialize reduced_dists_per_query_d, reduced_ids_per_query_d to 0
  CHECK_CUDA_ERROR(cudaMemset(reduced_dists_per_query_d, 0, reduce_group_num * max_queries_num * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMemset(reduced_ids_per_query_d, 0, reduce_group_num * max_queries_num * sizeof(int)));
  ///

  /// allocate memory for phase1_distances_d, phase1_ids_d
  CHECK_CUDA_ERROR(cudaMalloc(&phase1_distances_d, max_queries_num * phase1_topk * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&phase1_ids_d, max_queries_num * phase1_topk * sizeof(int)));
  ///

  /// initialize phase1_distances_d, phase1_ids_d to 0
  CHECK_CUDA_ERROR(cudaMemset(phase1_distances_d, 0, max_queries_num * phase1_topk * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMemset(phase1_ids_d, 0, max_queries_num * phase1_topk * sizeof(int)));
  ///

  /// initialize segments_offsets_d, temp_storage_d, temp_storage_bytes
  int *segments_offsets_h;
  CHECK_CUDA_ERROR(cudaMallocHost(&segments_offsets_h, (max_queries_num + 1) * sizeof(int)));
  for (int i = 0; i < max_queries_num + 1; i++)
  {
    segments_offsets_h[i] = i * reduce_group_num;
  }
  CHECK_CUDA_ERROR(cudaMalloc(&segments_offsets_d, (max_queries_num + 1) * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(segments_offsets_d, segments_offsets_h, (max_queries_num + 1) * sizeof(int), cudaMemcpyHostToDevice));

  temp_storage_bytes = 1024 * 1024 * 1024;

#ifdef DETAILED_LOG
  printf("temp_storage_bytes in construction: %d\n", temp_storage_bytes);
#endif

  CHECK_CUDA_ERROR(cudaMalloc(&temp_storage_d, temp_storage_bytes));
  ///

  /// allocate memory for phase1_distances_h, phase1_ids_h and phase2_ids_h
  CHECK_CUDA_ERROR(cudaMallocHost(&phase1_distances_h, max_queries_num * phase1_topk * sizeof(half)));
  // cudaMallocHost(&phase1_ids_h, max_queries_num * phase1_topk * sizeof(int));
  // cudaMallocHost(&phase2_ids_h, max_queries_num * phase2_topk * sizeof(int));
  ///
}

TrihAnnsWorker::TrihAnnsWorker(
    TrihAnnsWorker &other,
    cudaStream_t work_stream_) : alpha(other.alpha),
                                 beta(other.beta),
                                 alpha_d(other.alpha_d),
                                 beta_d(other.beta_d),
                                 gemm_workspace(1024 * 1024 * 1024),
                                 full_dim_pca_data_d(other.full_dim_pca_data_d),
                                 full_dim_pca_data_h(other.full_dim_pca_data_h),
                                 pca_dim_pca_data_d(other.pca_dim_pca_data_d),
                                 remain_dim_pca_data_d(other.remain_dim_pca_data_d),
                                 half_pca_dim_pca_data_d(other.half_pca_dim_pca_data_d),
                                 alpha0_d(other.alpha0_d),
                                 beta0_d(other.beta0_d),
                                 half_pca_dataset_d(other.half_pca_dataset_d),
                                 half_pca_dataset_norms_d(other.half_pca_dataset_norms_d),
                                 segments_offsets_d(other.segments_offsets_d),
                                 temp_storage_bytes(other.temp_storage_bytes),
                                 base_dataset_h(other.base_dataset_h),
                                 pca_dataset_h(other.pca_dataset_h),
                                 base_dataset_norms_h(other.base_dataset_norms_h),
                                 rerank_thread_pool(other.rerank_thread_pool),
                                 sq_info_h(other.sq_info_h),
                                 data_num(other.data_num),
                                 max_queries_num(other.max_queries_num),
                                 dim(other.dim),
                                 pca_dim(other.pca_dim),
                                 reduce_group_size(other.reduce_group_size),
                                 reduce_group_num(other.reduce_group_num),
                                 phase1_topk(other.phase1_topk),
                                 phase2_topk(other.phase2_topk),
                                 work_stream(work_stream_),
                                 falpha_d(other.falpha_d),
                                 fbeta_d(other.fbeta_d)
{
  /// allocate memory for batch_query_d, pca_batch_query_d, half_batch_query_d
  cudaMalloc(&batch_query_d, max_queries_num * dim * sizeof(float));
  cudaMalloc(&pca_batch_query_d, max_queries_num * pca_dim * sizeof(float));
  cudaMalloc(&half_batch_query_d, max_queries_num * dim * sizeof(half));

  cudaMalloc(&remain_batch_query_d, max_queries_num * (dim - pca_dim) * sizeof(float));
  // cudaMallocHost(&remain_batch_query_h, max_queries_num * (dim - pca_dim) * sizeof(float));
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
  ///

  // quant_queries_h = static_cast<uint8_t*>(aligned_alloc(64, (dim - pca_dim) * max_queries_num * sizeof(uint8_t)));
  // aligned_malloc_host(
  //     (void **)&quant_queries_h,
  //     (dim - pca_dim) * max_queries_num * sizeof(uint8_t),
  //     64);
}

void rerank_task_scheduler(
    TrihAnnsWorker *worker,
    cudaEvent_t syncEvent,
    float *batch_query,
    int batch_query_num,
    half *phase1_distances_h,
    int *phase1_ids_h,
    int *phase2_ids_h,
    float *remain_batch_query_h,
    uint8_t *quant_queries_h)
{
  cudaEventSynchronize(syncEvent);
  for (int i = 0; i < batch_query_num; i++)
  {
    rr_pool->detach_task(
        [worker, batch_query, i, phase1_distances_h, phase1_ids_h, phase2_ids_h, quant_queries_h, remain_batch_query_h]
        {
          re_rank(
              batch_query + i * worker->dim,
              quant_queries_h + i * (worker->dim - worker->pca_dim),
              remain_batch_query_h + i * (worker->dim - worker->pca_dim),
              worker->dim,
              worker->pca_dim,
              worker->sq_info_h->data(),
              phase1_distances_h + i * worker->phase1_topk,
              phase1_ids_h + i * worker->phase1_topk,
              worker->phase1_topk,
              phase2_ids_h + i * worker->phase2_topk,
              worker->phase2_topk);
        });
  }
  // cudaEventDestroy(syncEvent);
}

void TrihAnnsWorker::batch_query_search(
    float *batch_query,
    int batch_query_num,
    half *phase1_distances_h,
    int *phase1_ids_h,
    int *phase2_ids_h,
    // cudaEvent_t syncEvent,
    float *remain_batch_query_h,
    uint8_t *quant_queries_h,
    bool verbose)
{
  // auto start_time = std::chrono::high_resolution_clock::now();
  /// copy batch_query to gpu and project it and transform to half precision
  cudaMemcpyAsync(batch_query_d, batch_query, batch_query_num * dim * sizeof(float), cudaMemcpyHostToDevice, work_stream);

  // project the query,
  // pca_dim_pca_data_d(dim, pca_dim),
  // batch_query_d(dim, batch_query_num),
  // pca_batch_query_d(pca_dim, batch_query_num)

  // covert batch_query to half precision
  // float_to_half_kernel<<<(batch_query_num * dim + 255) / 256, 256, 0, work_stream>>>
  //   (batch_query_d, half_batch_query_d, batch_query_num * dim);

  // cublasGemmEx(
  //   handle,
  //   CUBLAS_OP_T, CUBLAS_OP_N,
  //   pca_dim, batch_query_num, dim,
  //   alpha0_d,
  //   half_pca_dim_pca_data_d, CUDA_R_16F, dim,
  //   half_batch_query_d, CUDA_R_16F, dim,
  //   beta0_d,
  //   half_pca_queries_d, CUDA_R_16F, pca_dim,
  //   CUDA_R_16F,
  //   CUBLAS_GEMM_DEFAULT_TENSOR_OP
  // );

  cublasGemmEx(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      pca_dim, batch_query_num, dim,
      falpha_d,
      pca_dim_pca_data_d, CUDA_R_32F, dim,
      batch_query_d, CUDA_R_32F, dim,
      fbeta_d,
      pca_batch_query_d, CUDA_R_32F, pca_dim,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  float_to_half_kernel<<<(batch_query_num * pca_dim + 255) / 256, 256, 0, work_stream>>>(pca_batch_query_d, half_pca_queries_d, batch_query_num * pca_dim);

  // remain dim projection
  cublasGemmEx(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      dim - pca_dim, batch_query_num, dim,
      falpha_d,
      remain_dim_pca_data_d, CUDA_R_32F, dim,
      batch_query_d, CUDA_R_32F, dim,
      fbeta_d,
      remain_batch_query_d, CUDA_R_32F, dim - pca_dim,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // printf("flag 1\n");

  cutlass::gemm::GemmCoord problem_size(data_num, batch_query_num, pca_dim);

  int split_k_slices = 1;

  cutlass::layout::RowMajor tensor_A_layout(pca_dim);
  cutlass::layout::ColumnMajor tensor_B_layout(pca_dim);
  cutlass::layout::ColumnMajor tensor_C_layout(data_num);
  cutlass::layout::ColumnMajor tensor_D_layout(data_num);

  cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> tensor_A(reinterpret_cast<cutlass::half_t *>(half_pca_dataset_d), tensor_A_layout);
  cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_B(reinterpret_cast<cutlass::half_t *>(half_pca_queries_d), tensor_B_layout);
  cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_C(reinterpret_cast<cutlass::half_t *>(half_pca_dataset_norms_d), tensor_C_layout);
  cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_D(reinterpret_cast<cutlass::half_t *>(half_dists_d), tensor_D_layout);

  // printf("flag 2\n");

  typename Gemm::Arguments arguments{
      problem_size,
      tensor_A,
      tensor_B,
      tensor_C,
      tensor_D,
      {alpha, beta},
      // {*alpha_d, *beta_d},
      split_k_slices};

  // printf("flag 3\n");

  gemm_op.initialize(arguments, gemm_workspace.get(), work_stream);

  // cudaStreamSynchronize(work_stream);
  // printf("flag 4\n");

  gemm_op(work_stream);

  // cudaStreamSynchronize(work_stream);

  // Debug: write 1000 distance results to a file for each query
  // if (batch_query_num > 0) {
  //   // Allocate host memory to store distances
  //   half* debug_dists_h = nullptr;
  //   const int debug_count = std::min((size_t)100, data_num);
  //   cudaMallocHost(&debug_dists_h, batch_query_num * debug_count * sizeof(half));

  //   // Copy a subset of distances from device to host
  //   for (int q = 0; q < batch_query_num; q++) {
  //     cudaMemcpyAsync(
  //         debug_dists_h + q * debug_count,
  //         half_dists_d + q * data_num + 2340268,
  //         debug_count * sizeof(half),
  //         cudaMemcpyDeviceToHost,
  //         work_stream);
  //   }

  //   cudaStreamSynchronize(work_stream);

  //   // Write to file
  //   char filename[100];
  //   sprintf(filename, "log/dists_debug_%d.txt", file_ind++);
  //   FILE* fp = fopen(filename, "w");
  //   if (fp) {
  //     for (int q = 0; q < batch_query_num; q++) {
  //       fprintf(fp, "Query %d distances:\n", q);
  //       for (int i = 0; i < debug_count; i++) {
  //         fprintf(fp, "%.6f ", __half2float(debug_dists_h[q * debug_count + i]));
  //         if ((i + 1) % 4 == 0) fprintf(fp, "\n");
  //       }
  //       fprintf(fp, "\n\n");
  //     }
  //     fclose(fp);
  //   }

  //   // Free allocated memory
  //   cudaFreeHost(debug_dists_h);
  // }

  // cudaStreamSynchronize(work_stream);
  // printf("flag 5\n");

  /// reduce half_dists_d into reduced_dists_per_query_d and reduced_ids_per_query_d
  // half_matrix_reduce(
  //   half_dists_d,
  //   reduced_dists_per_query_d,
  //   reduced_ids_per_query_d,
  //   reduce_group_size,
  //   reduce_group_num * batch_query_num,
  //   reduce_group_num,
  //   work_stream
  // );

  // printf("half_matrix_reduce_v2\n");

  half_matrix_reduce_v2(
      half_dists_d,
      reduced_dists_per_query_d,
      reduced_ids_per_query_d,
      reduce_group_size,
      data_num,
      batch_query_num,
      8,
      work_stream);
  ///

  // Debug: write reduced distances and IDs to a file
  // cudaStreamSynchronize(work_stream);
  // half *debug_reduced_dists_h = nullptr;
  // int *debug_reduced_ids_h = nullptr;
  // const int total_elements = reduce_group_num * batch_query_num;
  // cudaMallocHost(&debug_reduced_dists_h, total_elements * sizeof(half));
  // cudaMallocHost(&debug_reduced_ids_h, total_elements * sizeof(int));

  // // Copy data from device to host
  // cudaMemcpyAsync(debug_reduced_dists_h, reduced_dists_per_query_d,
  //                 total_elements * sizeof(half), cudaMemcpyDeviceToHost, work_stream);
  // cudaMemcpyAsync(debug_reduced_ids_h, reduced_ids_per_query_d,
  //                 total_elements * sizeof(int), cudaMemcpyDeviceToHost, work_stream);
  // cudaStreamSynchronize(work_stream);

  // // Write to file
  // FILE *fp = fopen("log/reduced_data_debug2.txt", "a");
  // if (fp)
  // {
  //   fprintf(fp, "Query\tGroup\tID\tDistance\n");
  //   for (int q = 0; q < batch_query_num; q++)
  //   {
  //     for (int g = 0; g < reduce_group_num; g++)
  //     {
  //       int idx = q * reduce_group_num + g;
  //       fprintf(fp, "%d\t%d\t%d\t%.6f\n",
  //               q, g, debug_reduced_ids_h[idx], __half2float(debug_reduced_dists_h[idx]));
  //     }
  //     fprintf(fp, "\n");
  //   }
  //   fclose(fp);
  // }

  // // Free allocated memory
  // cudaFreeHost(debug_reduced_dists_h);
  // cudaFreeHost(debug_reduced_ids_h);

  // printf("reduced_dists_per_query_d\n");

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
      work_stream);

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
      work_stream);

  // printf("cub::DeviceSegmentedRadixSort::SortPairs\n");

  extract_topk(
      reduced_dists_per_query_d,
      reduced_ids_per_query_d,
      phase1_distances_d,
      phase1_ids_d,
      batch_query_num,
      reduce_group_num,
      phase1_topk,
      work_stream,
      false);
  ///

  // printf("extract_topk\n");

  /// copy phase1_distances_d, phase1_ids_d to phase1_distances_h, phase1_ids_h
  cudaMemcpyAsync(phase1_distances_h, phase1_distances_d, batch_query_num * phase1_topk * sizeof(half), cudaMemcpyDeviceToHost, work_stream);
  cudaMemcpyAsync(phase1_ids_h, phase1_ids_d, batch_query_num * phase1_topk * sizeof(int), cudaMemcpyDeviceToHost, work_stream);
  ///

  /// copy remain_batch_query_d to remain_batch_query_h
  cudaMemcpyAsync(remain_batch_query_h, remain_batch_query_d, batch_query_num * (dim - pca_dim) * sizeof(float), cudaMemcpyDeviceToHost, work_stream);
  ///

  cudaStreamSynchronize(work_stream);

  // Create syncEvent for rerank task scheduling
  // cudaEvent_t syncEvent;
  // cudaEventCreate(&syncEvent);

  // Start timing with std::chrono
  // auto start_time = std::chrono::high_resolution_clock::now();
  // Record syncEvent for task synchronization
  // cudaEventRecord(syncEvent, work_stream);

  // End timing
  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  // if (verbose) {
  //   printf("GPU operations time: %.3f ms\n", static_cast<float>(duration));
  // }

  // rerank_task_scheduler_pool.detach_task(
  //     [this, syncEvent, batch_query, batch_query_num, phase1_distances_h, phase1_ids_h, phase2_ids_h]
  //     {
  //       rerank_task_scheduler(
  //           this,
  //           syncEvent,
  //           batch_query,
  //           batch_query_num,
  //           phase1_distances_h,
  //           phase1_ids_h,
  //           phase2_ids_h);
  //     });

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  // printf("batch_query_search gpu execution time: %.3f us\n", static_cast<float>(duration));

  // auto rerank_start = std::chrono::high_resolution_clock::now();
  /// assign rerank tasks
  // std::vector<std::future<int>> results;

  // std::cout << "there are " << rr_pool->get_tasks_queued() << " in pool" << std::endl;
  {
    // std::lock_guard<std::mutex> lock(thread_pool_mutex);
    // printf("re-renk task assign\n");

    // rr_pool_mutex.lock();
    for (int i = 0; i < batch_query_num; i++)
    {
      // rr_pool->detach_task(
      //     [this, batch_query, i, phase1_ids_h, phase2_ids_h]
      //     {
      //       re_rank2(
      //           this->base_dataset_h,
      //           this->base_dataset_norms_h,
      //           batch_query + i * this->dim,
      //           // this->phase1_ids_h + i * this->phase1_topk,
      //           phase1_ids_h + i * this->phase1_topk,
      //           this->phase1_topk,
      //           this->data_num,
      //           this->dim,
      //           this->phase2_topk,
      //           phase2_ids_h + i * this->phase2_topk);
      //     });

      rr_pool->detach_task(
          [this, batch_query, i, phase1_distances_h, phase1_ids_h, phase2_ids_h, quant_queries_h, remain_batch_query_h]
          {
            re_rank(
                batch_query + i * this->dim,
                quant_queries_h + i * (this->dim - this->pca_dim),
                remain_batch_query_h + i * (this->dim - this->pca_dim),
                this->dim,
                this->pca_dim,
                this->sq_info_h->data(),
                phase1_distances_h + i * this->phase1_topk,
                phase1_ids_h + i * this->phase1_topk,
                this->phase1_topk,
                phase2_ids_h + i * this->phase2_topk,
                this->phase2_topk);
          });
    }
    // rr_pool->wait();
    // rr_pool_mutex.unlock();
  }
  // auto rerank_end = std::chrono::high_resolution_clock::now();
  // auto rerank_duration = std::chrono::duration_cast<std::chrono::microseconds>(rerank_end - rerank_start).count();
  // printf("Re-ranking phase execution time: %.3f us\n", static_cast<float>(rerank_duration));
}

extern atomic<int> query_batch_counter;

void search_task(
    TrihAnnsWorker *worker,
    float *batch_query,
    int batch_query_num,
    half *phase1_distances_h,
    int *phase1_ids_h,
    int *phase2_ids_h,
    int query_batch_num,
    // , cudaEvent_t* syncEvent
    float *remain_batch_query_h,
    uint8_t *quant_queries_h)
{
  // repeatly call batch_query_search until all queries are processed
  while (true)
  {
    int current_query_batch_num = query_batch_counter.fetch_add(1);
    if (current_query_batch_num >= query_batch_num)
      break;

    // printf("query batch %d\n", current_query_batch_num);
    worker->batch_query_search(
        batch_query + current_query_batch_num * batch_query_num * worker->dim,
        batch_query_num,
        phase1_distances_h + current_query_batch_num * batch_query_num * worker->phase1_topk,
        phase1_ids_h + current_query_batch_num * batch_query_num * worker->phase1_topk,
        phase2_ids_h + current_query_batch_num * batch_query_num * worker->phase2_topk,
        // syncEvent[current_query_batch_num],
        remain_batch_query_h + current_query_batch_num * batch_query_num * (worker->dim - worker->pca_dim),
        quant_queries_h + current_query_batch_num * batch_query_num * (worker->dim - worker->pca_dim),
        current_query_batch_num == 0 ? true : false);
  }
}