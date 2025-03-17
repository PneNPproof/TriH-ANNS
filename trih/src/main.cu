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

#include "gist.h"
#include "pca.h"
#include "search.h"
#include "shuffle.h"
#include "utils.h"

#include "gpu_pca_anns.cuh"

#define SECTION_NUM 8000
#define TOP_K 300

using namespace std;

// write a GPU warm up kernel
__global__ void warmup(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// write a host func to call the warm up kernel
void gpu_warmup() {
    int n = 1024;
    int *a, *b, *c;
    cudaMalloc(&a, n * sizeof(int));
    cudaMalloc(&b, n * sizeof(int));
    cudaMalloc(&c, n * sizeof(int));
    
    // Call the kernel with appropriate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    warmup<<<numBlocks, blockSize>>>(a, b, c, n);
    
    // Synchronize to ensure the kernel completes
    cudaDeviceSynchronize();
    
    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    
    // Reset any errors
    cudaGetLastError();
}



int main(int argc, char *argv[]) {
            
    gist gdata;

    std::cout << "Loading gist data ..." << std::endl;
    gdata = load_gist("/home/wangzhe/faiss_cpu/GIST960_L2/gist-960-euclidean.hdf5");

    char *index_tag = argv[2];
    char filename[100];
    sprintf(filename, "proj_%s.bin", index_tag);
    
    if(strcmp(argv[1], "b") == 0) { //build index,           command: ./pca b index_tag column_num(64) ratio(0.85) column_num2(896) ratio2(0.85)
        if(argc != 7) {
            cout << "Wrong parameters!" << endl;
            return 0;
        }

        int column_num = atoi(argv[3]); //当 列 数量不为 0 ，以输出 列为主，不考虑 ratio 
        float ratio = atof(argv[4]); //当 列 数量为 0， 以ratio为主，比如 0.85
        int column_num2 = atoi(argv[5]);
        float ratio2 = atof(argv[6]); //当 列 数量为 0， 以ratio2为主，比如 0.85
        
        ofstream ofs(filename, std::ios::binary);

        //对数据进行shuffle
        myshuffle2(gdata.train, 960, 1000000, (int*)gdata.neighbors, 1000*100);

        sprintf(filename, "shuffled_gist_%s.bin", index_tag);
        save_shuffled_gist(filename, gdata);

        build_index(gdata.train, 960, 1000000, column_num, ratio, column_num2, ratio2, ofs);

        ofs.close();
    }
    else if(strcmp(argv[1], "s") == 0) { // search,          command: ./pca s index_tag section_num(8000) top_k(300) 

        if(argc != 5) {
            cout << "Wrong parameters!" << endl;
            return 0;
        }

        std::cout << "Loading index ..." << std::endl;
        ifstream ifs(filename, std::ios::binary);

        //load shuffled gist, 在load gist数据的基础上，替换
        sprintf(filename, "shuffled_gist_%s.bin", index_tag);
        load_shuffled_gist(filename, gdata);

        pca_index index;

        load_pca_index(ifs, index);
        std::cout << "index " << ": " << std::endl;
        std::cout << "\tdim=" << index.dim << std::endl;
        std::cout << "\trecord_num=" << index.record_num << std::endl;
        std::cout << "\tcolumn_num=" << index.column_num << std::endl;
        std::cout << "\tratio=" << index.ratio << std::endl;
        std::cout << "\tcolumn_num2=" << index.column_num2 << std::endl;
        std::cout << "\tratio2=" << index.ratio2 << std::endl;

        ifs.close();



        int test_batch_size = atoi(argv[3]);
        int top_k = atoi(argv[4]);//每个section 获取 top1之后，top_k 指定筛选多少

        // recall_test(gdata, index, section_num, top_k);
        // recall_test_sq(gdata, index, section_num, top_k);
        
        gpu_warmup();
        gpu_anns(gdata.test, test_batch_size, gdata.train, index, gdata.distances, (int *)gdata.neighbors, 125, top_k, 100, (int *)gdata.neighbors);

    }

    return 0;
}

