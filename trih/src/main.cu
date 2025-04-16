#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <thread>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <float.h>
#include <atomic>
#include <thread>

#include <cuda_runtime.h>

#include "pca.h"
#include "shuffle.h"
#include "utils.h"
#include "dataset.h"
#include "BS_thread_pool.hpp"

#include "sq.h"

#include "gpu_pca_anns.cuh"

// #define SECTION_NUM 8000
// #define TOP_K 300

using namespace std;


// 全局变量
// BS::thread_pool rr_pool(12);
BS::thread_pool<>* rr_pool;
// BS::thread_pool rerank_task_scheduler_pool(4);
int file_ind = 2;
atomic<int> query_batch_counter(0);

int main(int argc, char *argv[])
{

    // const char *data_dir = "/home/yshen/ann-benchmarks/data";
    // const char *index_dir = "/home/wangzhe/TriH-ANNS/index";
    const char *data_dir = "/Trih/dataset/dataset_shuffle";
    const char *index_dir = "/paper_experiment/index";

    char filename[100];

    std::cout << "Loading data ..." << std::endl;
    trih::data gdata;
    sprintf(filename, "%s/%s", data_dir, argv[1]);
    gdata = trih::load_data(filename);

    char *index_tag = argv[3];
    sprintf(filename, "%s/proj_%s.bin", index_dir, index_tag);

    if (argv[2][0] == 'b')
    {                                   // build index,           command: ./pca data_filename b index_tag column_num(128) ratio(0.85)
        int column_num = atoi(argv[4]); // 当 列 数量不为 0 ，以输出 列为主，不考虑 ratio
        float ratio = atof(argv[5]);    // 当 列 数量为 0， 以ratio为主，比如 0.85

        std::cout << "Generating index ..." << std::endl;

        ofstream ofs(filename, std::ios::binary);

        // 对数据进行shuffle
        // myshuffle2(gdata.train, gdata.dim, gdata.train_point_count, (int *)gdata.neighbors, gdata.test_point_count * gdata.neighbors_per_test);
        // sprintf(filename, "%s/shuffled_%s.bin", index_dir, index_tag);
        // save_shuffled_data(filename, gdata);

        trih::build_index(gdata.train, gdata.dim, gdata.train_point_count, column_num, ratio, ofs);
        ofs.close();
    }
    else if (argv[2][0] == 's')
    { // search,  command: ./pca data_filename s index_tag query_batch_size phase1_topk phase2_topk reduce_group_size num_workers batch_num_per_worker rerank_thread_pool_size

        std::cout << "Loading index ..." << std::endl;

        ifstream ifs(filename, std::ios::binary);

        // load shuffled data, 在load 数据的基础上，替换
        // sprintf(filename, "%s/shuffled_%s.bin", index_dir, index_tag);
        // load_shuffled_data(filename, gdata);

        pca_index index;
        load_pca_index(ifs, index);

        /// Make gdata.train_point_count and index.record_num a multiple of 4.
        auto residual = gdata.train_point_count % 8;
        gdata.train_point_count -= residual;
        index.record_num -= residual;
        ///

        std::cout << "\tindex " << ": " << std::endl;
        std::cout << "\tdim=" << index.dim << std::endl;
        std::cout << "\trecord_num=" << index.record_num << std::endl;
        std::cout << "\tcolumn_num=" << index.column_num << std::endl;
        std::cout << "\tratio=" << index.ratio << std::endl;

        ifs.close();

        int query_batch_size = atoi(argv[4]);
        int phase1_topk = atoi(argv[5]); // 每个section 获取 top1之后，top_k 指定筛选多少
        int phase2_topk = atoi(argv[6]);
        int rerank_thread_pool_size = atoi(argv[10]);
        int reduce_group_size = atoi(argv[7]);
        int reduce_group_num = (gdata.train_point_count + (reduce_group_size - 1)) / reduce_group_size;

        rr_pool = new BS::thread_pool<>(rerank_thread_pool_size);
        // file_ind = atoi(argv[11]);

        /// create multi worker
        int num_workers = atoi(argv[8]);
        int max_queries_num = 1000;

        vector<cudaStream_t> streams(num_workers);
        vector<TrihAnnsWorker *> workers(num_workers);
        for (int i = 0; i < num_workers; i++)
        {
            cudaStreamCreate(&streams[i]);
        }

        TrihAnnsWorker worker(
            index,
            index.pca_data,
            gdata.train,
            index.trans_data,
            index.record_num,
            max_queries_num,
            index.dim,
            index.column_num,
            reduce_group_size,
            reduce_group_num,
            phase1_topk,
            phase2_topk,
            streams[0],
            rerank_thread_pool_size);
        workers[0] = &worker;
        
        printf("create first worker done\n");

        for (int i = 1; i < num_workers; i++)
        {
            workers[i] = new TrihAnnsWorker(worker, streams[i]);
        }

        printf("create workers done\n");
        ///

        /// prepare query batch
        int batch_num_per_worker = atoi(argv[9]);
        int query_batch_num = num_workers * batch_num_per_worker;

        // float *batch_query = (float *)aligned_alloc(64, query_batch_num * query_batch_size * gdata.dim * sizeof(float));
        float *batch_query;
        aligned_malloc_host((void **)&batch_query, query_batch_num * query_batch_size * gdata.dim * sizeof(float), 64);
        int *batch_query_groundtruth = (int *)aligned_alloc(64, query_batch_num * query_batch_size * gdata.neighbors_per_test * sizeof(int));

        prepare_query_batch(
            gdata,
            query_batch_size,
            num_workers,
            batch_num_per_worker,
            batch_query,
            batch_query_groundtruth);

        printf("prepare query batch done\n");
        ///

        /// randomly generate batch_query for warm up and do warm up
        float *warmup_batch_query = (float *)aligned_alloc(64, query_batch_size * index.dim * sizeof(float));
        for (int i = 0; i < query_batch_size * index.dim; i++)
        {
            // warmup_batch_query[i] = rand() / (float)RAND_MAX;
            warmup_batch_query[i] = 0;
        }
        half *candidate_topk_dists_1 = (half *)malloc(phase1_topk * query_batch_size * sizeof(half));
        int *candidate_topk_ids_1 = (int *)malloc(phase1_topk * query_batch_size * sizeof(int));
        int *topk_ids_1 = (int *)malloc(phase2_topk * query_batch_size * sizeof(int));
        // cudaEvent_t syncEvent;
        // cudaEventCreate(&syncEvent);

        workers[0]->batch_query_search(
            warmup_batch_query,
            query_batch_size,
            candidate_topk_dists_1,
            candidate_topk_ids_1,
            topk_ids_1,
            // syncEvent,
            false);
        // rerank_task_scheduler_pool.wait();
        rr_pool->wait();
        printf("warm up done\n");
        ///

        /// using multi_worker to process multiple query batches
        half *candidate_topk_dists = (half *)malloc(phase1_topk * query_batch_size * query_batch_num * sizeof(half));
        int *candidate_topk_ids = (int *)malloc(phase1_topk * query_batch_size * query_batch_num * sizeof(int));
        // cudaEvent_t* syncEvents = new cudaEvent_t[query_batch_num];

        // for (int i = 0; i < query_batch_num; i++) {
        //     cudaEventCreate(&syncEvents[i]);
        // }

        int *topk_ids = (int *)malloc(phase2_topk * query_batch_size * query_batch_num * sizeof(int));
        std::vector<std::thread> workers_threads;

        auto multi_worker_search_begin = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_workers; i++)
        {
            workers_threads.emplace_back(
                search_task,
                workers[i],
                batch_query,
                query_batch_size,
                candidate_topk_dists,
                candidate_topk_ids,
                topk_ids,
                query_batch_num
                // , syncEvents
            );
        }

        for (auto &&worker_thread : workers_threads)
        {
            worker_thread.join();
        }

        auto multi_worker_search_end_1 = std::chrono::high_resolution_clock::now();
        auto multi_worker_search_duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(multi_worker_search_end_1 - multi_worker_search_begin);
        printf("multi worker gpu calculation done, duration = %lld ms, qps = %zu, wait for re-rank\n", 
               multi_worker_search_duration_1.count(), 
               (size_t)(query_batch_num * query_batch_size) * 1000 / multi_worker_search_duration_1.count());

        // rr_pool.wait();
        // rerank_task_scheduler_pool.wait();
        rr_pool->wait();
        printf("re-rank done\n");

        auto multi_worker_search_end = std::chrono::high_resolution_clock::now();
        auto multi_worker_search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(multi_worker_search_end - multi_worker_search_begin);

        size_t qps = (size_t)(query_batch_num * query_batch_size) * 1000 / multi_worker_search_duration.count();
        printf("multi worker search done, qps = %zu\n", qps);

        trih::cal_recall(batch_query_groundtruth, topk_ids, phase2_topk, gdata.neighbors_per_test, query_batch_num * query_batch_size);
        ///

        /// free memory
        for (int i = 0; i < num_workers; i++)
        {
            cudaStreamDestroy(streams[i]);
        }
        ///
    }

    return 0;
}
