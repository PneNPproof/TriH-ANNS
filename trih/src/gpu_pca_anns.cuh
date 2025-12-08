/**
 * @file gpu_pca_anns.cuh
 * @brief GPU-accelerated TriH-ANNS worker class and search interface
 * 
 * This header defines the core GPU-accelerated search infrastructure for TriH-ANNS.
 * The system uses a two-phase approach:
 * 1. Phase 1: Fast GPU-based search in PCA-reduced space using CUTLASS GEMM
 * 2. Phase 2: CPU-based reranking using full-dimensional features
 * 
 * Key optimizations:
 * - CUTLASS library for high-performance tensor operations on modern GPUs
 * - Half-precision (FP16) arithmetic for memory bandwidth efficiency
 * - Asynchronous CPU reranking overlapped with GPU computation
 * - Multi-worker support with independent CUDA streams
 * 
 * Dependencies:
 * - CUDA runtime and cuBLAS for GPU operations
 * - CUTLASS library for optimized tensor operations
 * - Custom thread pool for CPU reranking
 * - L2MM custom kernels for distance computation
 */

#pragma once

#include <mutex>
#include <memory>

#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/device_memory.h"

#include "pca.h"
#include "thread_pool_v2.h"
#include "l2mm.cuh"
#include "sq.h"

/**
 * @brief Simple GPU-based ANNS function interface
 * 
 * Simplified interface for GPU-accelerated nearest neighbor search.
 * This function provides a direct API without the worker class abstraction.
 * 
 * @param query Query vectors array
 * @param query_num Number of queries to process
 * @param src_data Training dataset vectors
 * @param index PCA index structure with transformation matrices
 * @param distances Output distances for nearest neighbors
 * @param neighbors Output nearest neighbor indices
 * @param reduce_group_size Group size for GPU reduction operations
 * @param phase1_topk Number of candidates from phase 1 (PCA search)
 * @param phase2_topk Number of final results after reranking
 * @param ground_truth_neighbors Ground truth for accuracy evaluation (optional)
 * 
 * @note This is a convenience wrapper around the TrihAnnsWorker class
 */
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
);

/**
 * @brief High-performance GPU worker class for TriH-ANNS search
 * 
 * Encapsulates all GPU memory allocations and computational resources needed
 * for efficient batch processing of nearest neighbor queries. Supports:
 * - Multiple concurrent workers with separate CUDA streams
 * - Memory reuse across batches for optimal performance
 * - Asynchronous CPU reranking overlapped with GPU computation
 * - CUTLASS-optimized tensor operations for modern GPU architectures
 */
class TrihAnnsWorker
{
public:

/// CUTLASS GEMM configuration for high-performance tensor operations
  using ElementOutput = cutlass::half_t;        ///< Output element type (FP16)
  using ElementAccumulator = float;             ///< Accumulator type (FP32 for precision)
  
  /**
   * @brief CUTLASS GEMM operator configuration
   * 
   * Optimized for Ampere architecture (SM80+) with:
   * - Input: FP16 matrices (queries and dataset)
   * - Output: FP16 distances
   * - Accumulation: FP32 for numerical stability
   * - Tile sizes: 128x256x64 for optimal memory coalescing
   * - Warp-level tiles: 64x64x64 for tensor core utilization
   * - Instruction shape: 16x8x16 for tensor core efficiency
   */
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor,        // A matrix (queries)
      cutlass::half_t, cutlass::layout::ColumnMajor,     // B matrix (dataset)
      ElementOutput, cutlass::layout::ColumnMajor,       // C matrix (distances)
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 64>,            // Threadblock tile
      cutlass::gemm::GemmShape<64, 64, 64>,              // Warp tile
      cutlass::gemm::GemmShape<16, 8, 16>,               // Instruction shape
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  // Alternative smaller tile configuration (commented out)
  // using Gemm = cutlass::gemm::device::Gemm<...GemmShape<64, 64, 32>...>;

  // CUTLASS GEMM operation parameters
  ElementOutput alpha;           ///< GEMM alpha parameter
  ElementOutput beta;            ///< GEMM beta parameter  
  ElementOutput* alpha_d;        ///< Device memory for alpha
  ElementOutput* beta_d;         ///< Device memory for beta
  Gemm gemm_op;                  ///< CUTLASS GEMM operator instance
  cutlass::device_memory::allocation<uint8_t> gemm_workspace; ///< GEMM workspace memory

/// GPU memory for query projection operations
  float* full_dim_pca_data_d;    ///< Complete PCA transformation matrix on GPU
  float* full_dim_pca_data_h;    ///< Complete PCA transformation matrix on CPU
  float* pca_dim_pca_data_d;     ///< Primary PCA components on GPU
  float* remain_dim_pca_data_d;  ///< Remaining PCA components on GPU
  half* half_pca_dim_pca_data_d; ///< Primary PCA components in FP16
  float* batch_query_d;          ///< Query batch on GPU (FP32)
  half* half_batch_query_d;      ///< Query batch on GPU (FP16)
  float* pca_batch_query_d;      ///< Projected queries in PCA space
  float* remain_batch_query_d;   ///< Projected queries in remaining space
  half* alpha0_d;                ///< GEMM alpha parameter for projection
  half* beta0_d;                 ///< GEMM beta parameter for projection

/// GPU memory for L2 distance matrix computation
  half* half_pca_dataset_d;      ///< PCA-projected dataset in FP16
  half* half_pca_queries_d;      ///< PCA-projected queries in FP16
  half* half_pca_dataset_norms_d;///< Dataset vector norms in FP16
  half* half_dists_d;            ///< Distance matrix output
  cublasHandle_t handle;         ///< cuBLAS handle for BLAS operations

/// GPU memory for distance reduction and top-k selection
  half* reduced_dists_per_query_d; ///< Reduced distances per query
  int* reduced_ids_per_query_d;    ///< Reduced indices per query

/// GPU memory for segmented sorting operations
  half *phase1_distances_d;      ///< Phase 1 distances for sorting
  int *phase1_ids_d;             ///< Phase 1 indices for sorting
  int *segments_offsets_d;       ///< Segment boundaries for sorting
  void *temp_storage_d;          ///< Temporary storage for CUB operations
  size_t temp_storage_bytes;     ///< Size of temporary storage

/// CPU memory and threading for reranking operations
  half *phase1_distances_h;      ///< Phase 1 distances on CPU
  float* base_dataset_h;         ///< Original dataset on CPU for reranking
  float* pca_dataset_h;          ///< PCA dataset on CPU
  float* pca_dataset_d;          ///< PCA dataset on GPU
  float* base_dataset_norms_h;   ///< Dataset norms on CPU
  ThreadPool *rerank_thread_pool;///< Thread pool for CPU reranking
  static std::mutex thread_pool_mutex; ///< Synchronization for thread pool access
  std::shared_ptr<std::vector<sq_info>> sq_info_h; ///< Scalar quantization info

  // Worker configuration parameters
  size_t data_num;               ///< Number of database vectors
  size_t max_queries_num;        ///< Maximum queries per batch
  size_t dim;                    ///< Original feature dimensionality
  size_t pca_dim;                ///< PCA-reduced dimensionality
  size_t reduce_group_size;      ///< Group size for GPU reductions
  size_t reduce_group_num;       ///< Number of reduction groups
  size_t phase1_topk;            ///< Candidates from phase 1
  size_t phase2_topk;            ///< Final results after phase 2

  cudaStream_t work_stream;      ///< CUDA stream for this worker

  // Additional GEMM parameters for flexibility
  float *falpha_d;               ///< Float alpha parameter on GPU
  float *fbeta_d;                ///< Float beta parameter on GPU

  /**
   * @brief Primary constructor for TrihAnnsWorker
   * 
   * Initializes all GPU memory allocations and sets up computational resources.
   * This constructor performs the heavy initialization work including:
   * - GPU memory allocation for all intermediate results
   * - CUTLASS GEMM operator setup
   * - Thread pool initialization for CPU reranking
   * - cuBLAS handle creation
   * 
   * @param index PCA index containing transformation matrices and projected data
   * @param full_dim_pca_data Complete PCA transformation matrix
   * @param base_dataset Original training dataset for reranking
   * @param pca_dataset PCA-projected training dataset
   * @param data_num Number of training vectors
   * @param max_queries_num Maximum queries processed per batch
   * @param dim Original feature dimensionality
   * @param pca_dim PCA-reduced dimensionality
   * @param reduce_group_size Group size for GPU reduction operations
   * @param reduce_group_num Number of reduction groups
   * @param phase1_topk Number of candidates from phase 1
   * @param phase2_topk Number of final results
   * @param work_stream_ CUDA stream for this worker
   * @param re_rank_thread_pool_size_ Number of CPU threads for reranking
   */
  TrihAnnsWorker
  (
    pca_index &index,
    float *full_dim_pca_data,
    float *base_dataset,
    float *pca_dataset,
    int data_num,
    int max_queries_num,
    int dim,
    int pca_dim,
    int reduce_group_size,
    int reduce_group_num,
    int phase1_topk,
    int phase2_topk,
    cudaStream_t work_stream_,
    int re_rank_thread_pool_size_ = 12
  );

  /**
   * @brief Copy constructor for creating additional workers
   * 
   * Creates a new worker that shares GPU memory allocations with an existing worker
   * but uses a different CUDA stream. This enables multiple workers to process
   * different query batches concurrently while sharing the same dataset in GPU memory.
   * 
   * @param other Existing worker to copy GPU allocations from
   * @param work_stream_ CUDA stream for the new worker
   * 
   * @note GPU memory is shared, but each worker maintains independent processing state
   */
  TrihAnnsWorker
  (
    TrihAnnsWorker &other,
    cudaStream_t work_stream_
  );

  /**
   * @brief Execute batch nearest neighbor search
   * 
   * Performs complete two-phase search on a batch of queries:
   * 1. Projects queries to PCA space
   * 2. Computes distances in PCA space using CUTLASS GEMM
   * 3. Selects top-k candidates
   * 4. Launches asynchronous CPU reranking using full dimensions
   * 
   * @param batch_query Input query vectors (batch_query_num x dim)
   * @param batch_query_num Number of queries in batch
   * @param phase1_distances_h Output phase 1 distances (on CPU)
   * @param phase1_ids_h Output phase 1 candidate indices
   * @param phase2_ids_h Output final reranked indices
   * @param remain_batch_query_h Projected queries in remaining dimensions
   * @param quant_queries_h Quantized queries for efficient distance computation
   * @param verbose Enable detailed logging
   * @param use_normal_rerank use normal rerank method or optimized rerank
   * 
   * @note This function returns before reranking completes - use thread pool wait
   * @note All host memory must be pre-allocated by caller
   */
  void batch_query_search(
    float *batch_query,
    int batch_query_num,
    half *phase1_distances_h,
    int *phase1_ids_h,
    int *phase2_ids_h,
    float *remain_batch_query_h,
    uint8_t *quant_queries_h,
    bool verbose,
    bool use_normal_rerank
  );
};

/**
 * @brief Worker thread function for processing query batches
 * 
 * Thread function that coordinates multiple query batches across workers.
 * Each worker processes a subset of query batches using atomic counters
 * for work distribution.
 * 
 * @param worker TrihAnnsWorker instance for GPU processing
 * @param batch_query All query batches
 * @param batch_query_num Queries per batch
 * @param phase1_distances_h Output phase 1 distances for all batches
 * @param phase1_ids_h Output phase 1 indices for all batches
 * @param phase2_ids_h Output final indices for all batches
 * @param query_batch_num Total number of query batches
 * @param remain_batch_query_h Remaining dimension projections for all batches
 * @param quant_queries_h Quantized queries for all batches
 * @param use_normal_rerank use normal rerank method or optimized rerank
 * 
 * @note Uses global atomic counter for work distribution across workers
 * @note Each worker processes multiple batches until all are completed
 */
void search_task(
  TrihAnnsWorker *worker,
  float *batch_query,
  int batch_query_num,
  half *phase1_distances_h,
  int *phase1_ids_h,
  int *phase2_ids_h,
  int query_batch_num,
  float *remain_batch_query_h,
  uint8_t *quant_queries_h,
  bool use_normal_rerank
  );