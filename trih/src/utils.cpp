#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <unordered_set> // For efficient lookups and uniqueness
#include <numeric>       // Potentially for iota or other utilities (not strictly needed here)
#include <stdexcept>     // For error handling (optional, could use cerr)
#include <iomanip>       // For setting precision in output
#include <cstddef> // For size_t
#include <cstdint> // For uintptr_t

#include <cuda_runtime.h>

#include "pca.h"
#include "data.h"
#include "search.h"
#include "shuffle.h"

using namespace std;

namespace trih
{

// 保存shuffle数据
void save_shuffled_data(const char *filename, data &gdata)
{
    ofstream ofs(filename, std::ios::binary);

    ofs.write((const char *)gdata.train, sizeof(float) * gdata.dim * gdata.train_point_count);
    ofs.write((const char *)gdata.neighbors, sizeof(uint32_t) * gdata.neighbors_per_test * gdata.test_point_count);

    ofs.close();
}

// 用shuffle的数据覆盖gdata数据
void load_shuffled_data(const char *filename, data &gdata)
{
    ifstream ifs(filename, std::ios::binary);

    ifs.read((char *)gdata.train, sizeof(float) * gdata.dim * gdata.train_point_count);
    ifs.read((char *)gdata.neighbors, sizeof(uint32_t) * gdata.neighbors_per_test * gdata.test_point_count);

    ifs.close();
}

// 构建索引，保存为文件
// base 数据，dimension 原始数据维度，size 原始数据大小，column_num PCA 特征向量个数，ratio PCA 累积特征值比率
void build_index(float *base, int dimension, int size, int &column_num, float &ratio, ofstream &ofs)
{

    std::cout << "Starting to build PCA index..." << std::endl;

    save_pca_index(base, dimension, size, ratio, column_num, ofs);

    std::cout << "PCA index is ready." << std::endl;
}

/**
 * @brief Calculates the average recall@k over a batch of queries, handling duplicates
 *        and ground truth arrays potentially larger than k.
 *
 * Recall@k for a single query is defined as the number of *unique* relevant items
 * (from the true top-k ground truth) found within the top-k retrieved items,
 * divided by the number of *unique* items in the true top-k ground truth set.
 * Duplicate IDs within either the ground truth or the retrieved list for a single
 * query are handled such that a specific relevant ID only contributes once to the
 * hit count for that query.
 *
 * This function calculates the recall@k for each query in the batch and
 * then computes the average recall across the entire batch.
 *
 * @param gt Pointer to the flattened 2D array of ground truth neighbor IDs.
 *           Layout: [batch_query_num][gt_neighbors_per_query].
 *           Total size: batch_query_num * gt_neighbors_per_query.
 *           Contains the ground truth neighbors for each query.
 *           Only the *first* 'topk' neighbors for each query are considered relevant for Recall@k.
 * @param topk_ids Pointer to the flattened 2D array of retrieved top-k IDs.
 *                 Layout: [batch_query_num][topk]. Total size: batch_query_num * topk.
 *                 Contains the 'topk' IDs returned by the search system for each query.
 * @param topk The number of neighbors considered for recall (the 'k' in Recall@k).
 *             This defines how many retrieved items are checked and how many items
 *             from the beginning of the ground truth list are considered relevant.
 * @param gt_neighbors_per_query The actual number of neighbors stored per query in the 'gt' array.
 *                               Must be >= 'topk'.
 * @param batch_query_num The number of queries processed in this batch.
 */
void cal_recall(const int *gt, const int *topk_ids, int topk, int gt_neighbors_per_query, int batch_query_num)
{

    std::ofstream query_recall_file("log/query_recalls.txt");

    // --- Input Validation ---
    if (gt == nullptr)
    {
        std::cerr << "Error: Ground truth pointer (gt) is NULL." << std::endl;
        // Or throw std::invalid_argument("Ground truth pointer (gt) is NULL.");
        return;
    }
    if (topk_ids == nullptr)
    {
        std::cerr << "Error: Top-k IDs pointer (topk_ids) is NULL." << std::endl;
        // Or throw std::invalid_argument("Top-k IDs pointer (topk_ids) is NULL.");
        return;
    }
    if (batch_query_num <= 0)
    {
        std::cerr << "Warning: batch_query_num is non-positive (" << batch_query_num
                  << "). Average recall is 0." << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average Recall@" << topk << ": " << 0.0 << std::endl;
        return;
    }
     if (topk <= 0)
    {
        // Recall@0 or less doesn't make sense.
        std::cerr << "Warning: topk is non-positive (" << topk
                  << "). Cannot calculate recall. Average recall is 0." << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average Recall@" << topk << ": " << 0.0 << std::endl;
        return;
    }
    if (gt_neighbors_per_query < topk) {
        std::cerr << "Error: gt_neighbors_per_query (" << gt_neighbors_per_query
                  << ") must be greater than or equal to topk (" << topk << ")." << std::endl;
        // Or throw std::invalid_argument("gt_neighbors_per_query must be >= topk.");
        return; // Cannot determine the true top-k from the provided gt data
    }
     if (gt_neighbors_per_query <= 0) {
         std::cerr << "Warning: gt_neighbors_per_query is non-positive (" << gt_neighbors_per_query
                   << "). Assuming gt array is unusable. Recall is 0." << std::endl;
         std::cout << std::fixed << std::setprecision(6);
         std::cout << "Average Recall@" << topk << ": " << 0.0 << std::endl;
         return;
     }


    // --- Calculation ---
    long long total_hits = 0; // Use long long for safety against large batch/k
    // Optional: To calculate recall more strictly (sum of individual recalls / num_queries)
    double total_individual_recall_sum = 0.0;
    int valid_queries_for_strict_recall = 0; // Count queries with non-empty unique GT sets

    // Iterate through each query in the batch
    for (int i = 0; i < batch_query_num; ++i)
    {
        // Calculate the starting offset for the current query's data
        // Use long long for intermediate offset calculation to prevent overflow
        long long gt_offset = static_cast<long long>(i) * gt_neighbors_per_query;
        long long topk_ids_offset = static_cast<long long>(i) * topk;

        // Pointers to the start of the current query's data sections
        const int *current_gt_base_ptr = gt + gt_offset;
        const int *current_topk_ids_ptr = topk_ids + topk_ids_offset;

        // --- Optimized Hit Calculation using unordered_set ---

        // 1. Store the *unique true top-k* ground truth IDs in a hash set.
        //    We only look at the first 'topk' elements from the ground truth list
        //    for this specific query, even if more are available (`gt_neighbors_per_query`).
        std::unordered_set<int> true_topk_gt_set;
        true_topk_gt_set.reserve(topk); // Reserve space based on k
        for (int l = 0; l < topk; ++l)
        {
            true_topk_gt_set.insert(current_gt_base_ptr[l]); // Insert only the first 'topk' GT IDs
        }

        // The number of unique relevant items for *this* query (denominator for individual recall)
        size_t unique_gt_count_for_query = true_topk_gt_set.size();

        // if (unique_gt_count_for_query == 0 && topk > 0) {
        //     // Handle case where the top-k ground truth was empty or all duplicates of one item
        //     // if k=1 and gt=[5], unique=1. if k=2 and gt=[5, 5], unique=1. if k=0, unique=0.
        //     // Depending on definition, recall might be 0 or undefined.
        //     // Let's assume 0 hits if GT is effectively empty for recall purposes.
        //     continue; // Skip to next query if calculating strict average recall
        // }


        // 2. Iterate through the `topk` retrieved IDs and check against the true top-k GT set.
        //    Use another set to count only *unique* hits found in the retrieved list.
        std::unordered_set<int> unique_hits_found;
        unique_hits_found.reserve(topk); // Max possible unique hits is topk

        int current_query_hits = 0;
        for (int j = 0; j < topk; ++j)
        {
            int retrieved_id = current_topk_ids_ptr[j];

            // Check if the retrieved ID is relevant (present in the true top-k GT set)
            // true_topk_gt_set.count() is O(1) on average
            if (true_topk_gt_set.count(retrieved_id))
            {
                // Check if we haven't already counted this specific relevant ID *for this query*
                // unique_hits_found.insert().second returns true if insertion happened (was new)
                // O(1) average time for insert
                if (unique_hits_found.insert(retrieved_id).second)
                {
                    current_query_hits++;
                }
            }
        }
        total_hits += current_query_hits;
        // if (i==0)
        // printf("Query %d: Recall = %lf\n", i, current_query_hits / static_cast<double>(topk));
        // Write recall for this query to a file
        
        
        // query_recall_file << "Query " << i << ": Recall = " 
        //     << (current_query_hits / static_cast<double>(topk)) << std::endl;
            
        

        // Optional: Calculate individual recall and add to sum for strict average
        if (unique_gt_count_for_query > 0) {
            total_individual_recall_sum += static_cast<double>(current_query_hits) / unique_gt_count_for_query;
            valid_queries_for_strict_recall++;
        }

    } // End loop over queries

    query_recall_file.close();

    // --- Calculate and Print Average Recall ---

    // Definition 1: (Total Unique Hits across all queries) / (Total possible unique slots)
    // The denominator here is commonly batch_query_num * topk, assuming k is the target number
    // of relevant items per query.
    double average_recall = 0.0;
    if (batch_query_num > 0 && topk > 0) { // Avoid division by zero
         // Using the common definition: total hits / (num_queries * k)
        average_recall = static_cast<double>(total_hits) / (static_cast<double>(batch_query_num) * topk);
    }

    // Definition 2 (Optional, more strict): Average of individual query recalls
    double strict_average_recall = 0.0;
    if (valid_queries_for_strict_recall > 0) {
        strict_average_recall = total_individual_recall_sum / valid_queries_for_strict_recall;
    }

    std::cout << std::fixed << std::setprecision(6); // Set output precision
    // std::cout << "Average Recall@" << topk << ": " << average_recall << std::endl;
    // If using the strict definition:
    std::cout << "Strict Average Recall@" << topk << ": " << strict_average_recall << std::endl;

}

/**
 * @brief Prepares batches of query points and their ground truth nearest neighbors.
 *
 * This function populates pre-allocated buffers with query vectors and their
 * corresponding ground truth neighbor indices, drawing from the provided dataset.
 * If the number of required queries exceeds the available test points, the
 * test points are replicated cyclically.
 *
 * @param gdata The dataset containing test points and ground truth neighbors.
 * @param query_batch_size The number of queries in a single batch.
 * @param worker_num The number of parallel workers.
 * @param batch_num_per_worker The number of batches each worker will process.
 * @param batch_query Output buffer to store the batched query vectors (flattened).
 *                    Must be pre-allocated with size:
 *                    worker_num * batch_num_per_worker * query_batch_size * gdata.dim * sizeof(float).
 * @param batch_query_groundtruth Output buffer to store the batched ground truth neighbor indices (flattened).
 *                                Indices are stored as ints.
 *                                Must be pre-allocated with size:
 *                                worker_num * batch_num_per_worker * query_batch_size * gdata.neighbors_per_test * sizeof(int).
 */
void prepare_query_batch(data &gdata, int query_batch_size, int worker_num, int batch_num_per_worker, float *batch_query, int *batch_query_groundtruth) { // <-- Changed type here

    // --- Input Validation (Optional but Recommended) ---
    if (gdata.test == nullptr || gdata.neighbors == nullptr || batch_query == nullptr || batch_query_groundtruth == nullptr) {
        // Handle error: Null pointers provided
        throw std::runtime_error("Null pointer passed to prepare_query_batch");
        // Or: fprintf(stderr, "Error: Null pointer passed to prepare_query_batch\n"); return;
    }
    if (gdata.dim == 0 || gdata.test_point_count == 0 || gdata.neighbors_per_test == 0) {
         // Handle error: Invalid data dimensions or counts
         throw std::runtime_error("Invalid data dimensions or counts in gdata");
         // Or: fprintf(stderr, "Error: Invalid data dimensions or counts in gdata\n"); return;
    }
     if (query_batch_size <= 0 || worker_num <= 0 || batch_num_per_worker <= 0) {
        // Handle error: Invalid batching parameters
        throw std::runtime_error("Invalid batching parameters");
        // Or: fprintf(stderr, "Error: Invalid batching parameters\n"); return;
    }


    // --- Calculate Total Queries Needed ---
    // Use uint64_t to prevent potential overflow if numbers are large
    uint64_t total_queries_needed = (uint64_t)worker_num * batch_num_per_worker * query_batch_size;

    uint32_t dim = gdata.dim;
    uint32_t available_test_points = gdata.test_point_count;
    uint32_t k = gdata.neighbors_per_test; // Number of neighbors per query

    // --- Fill the Output Buffers ---
    for (uint64_t i = 0; i < total_queries_needed; ++i) {
        // Determine the index of the source query vector in gdata.test
        // Use modulo operator to handle replication if needed
        uint32_t source_query_index = i % available_test_points;

        // --- 1. Copy Query Vector ---
        // Calculate source pointer in gdata.test
        float *query_src_ptr = gdata.test + (uint64_t)source_query_index * dim;
        // Calculate destination pointer in batch_query
        float *query_dest_ptr = batch_query + i * dim;
        // Copy the vector data (dim * float values)
        memcpy(query_dest_ptr, query_src_ptr, dim * sizeof(float));

        // --- 2. Copy Ground Truth Neighbors ---
        // Calculate source pointer in gdata.neighbors (uint32_t*)
        uint32_t *gt_src_ptr = gdata.neighbors + (uint64_t)source_query_index * k;
        // Calculate destination pointer in batch_query_groundtruth (int*)
        int *gt_dest_ptr = batch_query_groundtruth + i * k; // <-- Changed pointer type here
        // Copy ground truth indices one by one, casting uint32_t to int
        for (uint32_t j = 0; j < k; ++j) {
            // Explicitly cast the uint32_t index to int for storage
            gt_dest_ptr[j] = (int)(gt_src_ptr[j]); // <-- Changed cast and destination type here
            // Note: Be aware of potential value truncation if a uint32_t index
            // exceeds the maximum value representable by a signed int (INT_MAX).
            // This is unlikely for typical dataset sizes but possible.
        }
    }
}

}

// Helper macro for CUDA error checking
#define checkCudaErrors(call)                                                  \
    do                                                                         \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            fprintf(stderr, "CUDA Error at %s %d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err), err);                             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


/**
 * @brief Allocates pinned host memory with a specified alignment.
 *
 * Uses cudaMallocHost for pinning and manual pointer arithmetic for alignment.
 * Stores the original pointer just before the aligned pointer for later freeing.
 *
 * @param ptr_out Address of the pointer that will receive the aligned memory address.
 * @param size The number of bytes to allocate.
 * @param alignment The desired alignment boundary (must be a power of 2).
 * @return cudaError_t Returns cudaSuccess on success, or a CUDA error code on failure.
 */
cudaError_t aligned_malloc_host(void** ptr_out, size_t size, size_t alignment) {
    if (!ptr_out) {
        return cudaErrorInvalidValue; // Output pointer cannot be null
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
         return cudaErrorInvalidValue; // Alignment must be a power of 2
    }
    if (size == 0) {
        *ptr_out = nullptr;
        return cudaSuccess; // Allocating 0 bytes is often handled this way
    }

    // 1. Calculate total allocation size needed
    // Need space for the requested size, potential padding for alignment,
    // and space to store the original raw pointer (sizeof(void*)).
    size_t total_size = size + alignment - 1 + sizeof(void*);
    void* raw_ptr = nullptr;

    // 2. Allocate raw pinned memory
    cudaError_t err = cudaMallocHost(&raw_ptr, total_size);
    if (err != cudaSuccess) {
        *ptr_out = nullptr;
        return err; // Allocation failed
    }

    // 3. Calculate the aligned pointer within the raw block
    // Add space for the original pointer storage *before* aligning
    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_ptr);
    uintptr_t aligned_addr = (raw_addr + sizeof(void*) + alignment - 1) & ~(alignment - 1);

    // 4. Store the original raw pointer just before the aligned address
    void** original_ptr_storage = reinterpret_cast<void**>(aligned_addr - sizeof(void*));
    *original_ptr_storage = raw_ptr;

    // 5. Set the output pointer to the aligned address
    *ptr_out = reinterpret_cast<void*>(aligned_addr);

    return cudaSuccess;
}