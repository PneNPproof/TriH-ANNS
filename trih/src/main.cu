/**
 * @file main.cu
 * @brief Main entry point for TriH-ANNS (Triangle Inequality based Hierarchical Approximate Nearest Neighbor Search)
 * 
 * This file implements the primary functionality for both building and searching TriH-ANNS indices.
 * The system uses a two-phase approach:
 * 1. PCA-based dimensionality reduction for initial candidate selection
 * 2. Precise reranking using original high-dimensional features
 * 
 * Dependencies:
 * - CUDA runtime for GPU acceleration
 * - OpenMP for CPU parallelization
 * - HDF5 for data loading
 * - CUTLASS for optimized GPU matrix operations
 * - Custom thread pool for asynchronous reranking
 * 
 * Usage:
 * - Build index: ./trih_anns <dataset> b <index_tag> <column_num> <ratio>
 * - Search: ./trih_anns <dataset> s <index_tag> <batch_size> <phase1_topk> <phase2_topk> <reduce_group_size> <num_workers> <batch_per_worker> <rerank_threads> <use_normal_rerank>
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <float.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <queue>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "BS_thread_pool.hpp"
#include "dataset.h"
#include "pca.h"
#include "shuffle.h"
#include "utils.h"

#include "sq.h"

#include "gpu_pca_anns.cuh"

using namespace std;

// Global thread pool for reranking operations - shared across all workers
BS::thread_pool<> *rr_pool;

// File index for multi-file operations (appears to be legacy code)
int file_ind = 2;

// Atomic counter for tracking query batch processing across multiple workers
atomic<int> query_batch_counter(0);

/**
 * @brief Main entry point for TriH-ANNS system
 * 
 * Handles both index building and search operations based on command line arguments.
 * The system supports two main modes:
 * 1. Build mode ('b'): Creates PCA-based index from training data
 * 2. Search mode ('s'): Performs approximate nearest neighbor search using built index
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments array
 *        For build: argv[1]=dataset, argv[2]='b', argv[3]=index_tag, argv[4]=column_num, argv[5]=ratio
 *        For search: argv[1]=dataset, argv[2]='s', argv[3]=index_tag, argv[4]=batch_size, 
 *                   argv[5]=phase1_topk, argv[6]=phase2_topk, argv[7]=reduce_group_size,
 *                   argv[8]=num_workers, argv[9]=batch_per_worker, argv[10]=rerank_threads, argv[11]=use_rerank
 * @return 0 on success, non-zero on error
 */
int main(int argc, char *argv[]) {
    // Default data and index directories - TODO: Make these configurable
    const char *data_dir = "/home/leizhou/lunwen_trih/dataset/dataset_shuffle";
    const char *index_dir = "/home/leizhou/lunwen_trih/TriH-add-rerank-11-3/index";

    char filename[100];

#ifdef DETAILED_LOG
    std::cout << "Loading data ..." << std::endl;
#endif

    // Load dataset (supports both HDF5 and binary vector formats)
    trih::data gdata;
    sprintf(filename, "%s/%s", data_dir, argv[1]);
    gdata = trih::load_data(filename);

    // Construct index filename based on provided tag
    char *index_tag = argv[3];
    sprintf(filename, "%s/proj_%s.bin", index_dir, index_tag);

    // check topK method 
    bool use_normal_topK = false;
    if(argc > 11)
    {
        use_normal_topK = atoi(argv[11]) == 1 ? true:false;
    }

    if (argv[2][0] == 'b') { 
        // BUILD INDEX MODE
        // Command format: ./pca data_filename b index_tag column_num(128) ratio(0.85)
        
        // Parse PCA parameters
        int column_num = atoi(argv[4]); // Number of PCA components to retain (0 = use ratio instead)
        float ratio = atof(argv[5]);    // Ratio of variance to preserve (used when column_num = 0)

        std::cout << "Generating index ..." << std::endl;

        ofstream ofs(filename, std::ios::binary);

        // TODO: The following shuffle code is commented out - consider if needed for performance
        // Apply data shuffling for better memory access patterns
        // myshuffle2(gdata.train, gdata.dim, gdata.train_point_count, (int *)gdata.neighbors, 
        //           gdata.test_point_count * gdata.neighbors_per_test);
        // sprintf(filename, "%s/shuffled_%s.bin", index_dir, index_tag);
        // save_shuffled_data(filename, gdata);

        // Build PCA-based index and save to file
        trih::build_index(gdata.train, gdata.dim, gdata.train_point_count,
                          column_num, ratio, ofs);
        ofs.close();
        
    } else if (argv[2][0] == 's') { 
        // SEARCH MODE
        // Command format: ./pca data_filename s index_tag query_batch_size phase1_topk phase2_topk
        //                reduce_group_size num_workers batch_num_per_worker rerank_thread_pool_size

#ifdef DETAILED_LOG
        std::cout << "Loading index ..." << std::endl;
#endif

        // Load previously built PCA index
        ifstream ifs(filename, std::ios::binary);

        // TODO: Load shuffled data if using shuffled indices
        // sprintf(filename, "%s/shuffled_%s.bin", index_dir, index_tag);
        // load_shuffled_data(filename, gdata);

        pca_index index;
        load_pca_index(ifs, index);

        // Ensure data sizes are multiples of 8 for efficient SIMD operations
        // This is a performance optimization for vectorized operations
        auto residual = gdata.train_point_count % 8;
        gdata.train_point_count -= residual;
        index.record_num -= residual;

#ifdef DETAILED_LOG
        std::cout << "\tindex " << ": " << std::endl;
        std::cout << "\tdim=" << index.dim << std::endl;
        std::cout << "\trecord_num=" << index.record_num << std::endl;
        std::cout << "\tcolumn_num=" << index.column_num << std::endl;
        std::cout << "\tratio=" << index.ratio << std::endl;
#endif

        ifs.close();

        // Start timing for search infrastructure setup
        auto start_time = std::chrono::high_resolution_clock::now();

        // Parse search parameters
        int query_batch_size = atoi(argv[4]);        // Number of queries processed together
        int phase1_topk = atoi(argv[5]);             // Top-k candidates from PCA phase
        int phase2_topk = atoi(argv[6]);             // Final top-k results after reranking
        int rerank_thread_pool_size = atoi(argv[10]); // Threads for CPU reranking
        int reduce_group_size = atoi(argv[7]);       // Group size for GPU reduction operations
        
        // Calculate number of reduction groups needed
        int reduce_group_num = (gdata.train_point_count + (reduce_group_size - 1)) / reduce_group_size;

        // Initialize global thread pool for reranking operations
        rr_pool = new BS::thread_pool<>(rerank_thread_pool_size);

        // Create multiple GPU workers for parallel processing
        int num_workers = atoi(argv[8]);
        int max_queries_num = 1000; // Maximum queries per batch - TODO: Make configurable

        // Create CUDA streams for each worker to enable concurrent GPU operations
        vector<cudaStream_t> streams(num_workers);
        vector<TrihAnnsWorker *> workers(num_workers);
        for (int i = 0; i < num_workers; i++) {
            cudaStreamCreate(&streams[i]);
        }

        // Create primary worker with full initialization
        TrihAnnsWorker worker(index, index.pca_data, gdata.train,
                              index.trans_data, index.record_num,
                              max_queries_num, index.dim, index.column_num,
                              reduce_group_size, reduce_group_num, phase1_topk,
                              phase2_topk, streams[0], rerank_thread_pool_size);
        workers[0] = &worker;

#ifdef DETAILED_LOG
        printf("create first worker done\n");
#endif

        // Create additional workers by copying from the primary worker
        // This shares GPU memory allocations while using separate streams
        for (int i = 1; i < num_workers; i++) {
            workers[i] = new TrihAnnsWorker(worker, streams[i]);
        }

#ifdef DETAILED_LOG
        printf("create workers done\n");
#endif

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

#ifdef DETAILED_LOG
        std::cout << "search build time: " << duration.count() << " ms" << std::endl;
#endif

        // Prepare query batches for processing
        int batch_num_per_worker = atoi(argv[9]);
        int query_batch_num = num_workers * batch_num_per_worker;

        // Allocate aligned memory for query batches and ground truth
        float *batch_query;
        aligned_malloc_host(
            (void **)&batch_query,
            query_batch_num * query_batch_size * gdata.dim * sizeof(float), 64);
        int *batch_query_groundtruth = (int *)aligned_alloc(
            64, query_batch_num * query_batch_size * gdata.neighbors_per_test * sizeof(int));

        // Prepare query batches from test data
        prepare_query_batch(gdata, query_batch_size, num_workers,
                            batch_num_per_worker, batch_query,
                            batch_query_groundtruth);

#ifdef DETAILED_LOG
        printf("prepare query batch done\n");
#endif

        // Warm up GPU kernels with dummy queries
        // This eliminates cold start overhead from timing measurements
        float *warmup_batch_query = (float *)aligned_alloc(
            64, query_batch_size * index.dim * sizeof(float));
        for (int i = 0; i < query_batch_size * index.dim; i++) {
            // Initialize with zeros instead of random for deterministic warmup
            warmup_batch_query[i] = 0;
        }
        
        // Allocate memory for search results
        half *candidate_topk_dists_1;
        int *candidate_topk_ids_1;
        aligned_malloc_host((void **)&candidate_topk_dists_1,
                            phase1_topk * query_batch_size * sizeof(half), 64);
        aligned_malloc_host((void **)&candidate_topk_ids_1,
                            phase1_topk * query_batch_size * sizeof(int), 64);
        int *topk_ids_1 = (int *)malloc(phase2_topk * query_batch_size * sizeof(int));

        // Allocate memory for intermediate processing
        float *remain_batch_query_1;
        uint8_t *quant_queries_1;
        aligned_malloc_host((void **)&remain_batch_query_1,
                            query_batch_size * (index.dim - index.column_num) * sizeof(float), 64);
        aligned_malloc_host((void **)&quant_queries_1,
                            query_batch_size * (index.dim - index.column_num) * sizeof(uint8_t), 64);

        // Perform warmup search to initialize GPU state
        workers[0]->batch_query_search(
            warmup_batch_query,
            query_batch_size,
            candidate_topk_dists_1,
            candidate_topk_ids_1,
            topk_ids_1,
            remain_batch_query_1,
            quant_queries_1,
            false,
            false
        ); // No synchronization needed for warmup

        // Wait for warmup reranking to complete
        rr_pool->wait();

#ifdef DETAILED_LOG
        printf("warm up done\n");
#endif

        // Allocate memory for actual benchmark queries
        half *candidate_topk_dists;
        int *candidate_topk_ids;
        aligned_malloc_host((void **)&candidate_topk_dists,
                            phase1_topk * query_batch_size * query_batch_num * sizeof(half), 64);
        aligned_malloc_host((void **)&candidate_topk_ids,
                            phase1_topk * query_batch_size * query_batch_num * sizeof(int), 64);

        int *topk_ids = (int *)malloc(phase2_topk * query_batch_size * query_batch_num * sizeof(int));

        // Allocate memory for remaining dimensions processing
        float *remain_batch_query;
        uint8_t *quant_queries;
        aligned_malloc_host((void **)&remain_batch_query,
                            query_batch_num * query_batch_size * 
                            (index.dim - index.column_num) * sizeof(float), 64);
        aligned_malloc_host((void **)&quant_queries,
                            query_batch_num * query_batch_size * 
                            (index.dim - index.column_num) * sizeof(uint8_t), 64);

        // Launch worker threads for parallel search processing
        std::vector<std::thread> workers_threads;

        auto multi_worker_search_begin = std::chrono::high_resolution_clock::now();

        // Start search task on each worker thread
        for (int i = 0; i < num_workers; i++) {
            workers_threads.emplace_back(
                search_task, workers[i], batch_query, query_batch_size,
                candidate_topk_dists, candidate_topk_ids, topk_ids,
                query_batch_num, remain_batch_query, quant_queries, use_normal_topK);
        }

        // Wait for all GPU processing to complete
        for (auto &&worker_thread : workers_threads) {
            worker_thread.join();
        }

        auto multi_worker_search_end_1 = std::chrono::high_resolution_clock::now();
        auto multi_worker_search_duration_1 =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                multi_worker_search_end_1 - multi_worker_search_begin);

#ifdef DETAILED_LOG
        printf("multi worker gpu calculation done, duration = %ld ms, qps = %zu, wait for re-rank\n",
               multi_worker_search_duration_1.count(),
               (size_t)(query_batch_num * query_batch_size) * 1000 /
                   multi_worker_search_duration_1.count());
#endif

        // Wait for all reranking operations to complete
        rr_pool->wait();

#ifdef DETAILED_LOG
        printf("re-rank done\n");
#endif

        auto multi_worker_search_end = std::chrono::high_resolution_clock::now();
        auto multi_worker_search_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                multi_worker_search_end - multi_worker_search_begin);

        // Calculate and report queries per second (QPS)
        size_t qps = (size_t)(query_batch_num * query_batch_size) * 1000 /
                     multi_worker_search_duration.count();

#ifdef DETAILED_LOG
        printf("multi worker search done, qps = %zu\n", qps);
#endif

        printf("qps = %zu\n", qps);

        // Calculate and report recall performance
        trih::cal_recall(batch_query_groundtruth, topk_ids, phase2_topk,
                         gdata.neighbors_per_test,
                         query_batch_num * query_batch_size);

        // Clean up CUDA resources
        for (int i = 0; i < num_workers; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    return 0;
}
