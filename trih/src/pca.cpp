#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>

#include "pca.h"
#include "shuffle.h"

#include "pca.cuh"

#define MAX_SAMPLE 1000000

using namespace std;

void PCA(const float* src, const int N0, const int D, float &ratio, int &d, float*& pca_data) {
    
    int N = N0;
    if(N > MAX_SAMPLE) N = MAX_SAMPLE;

    float *data = new float[N*D]{0};

    memcpy(data, src, sizeof(float)*N*D);

    float *mean = new float[D]{0};
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) 
            mean[j] += data[i*D+j];
    }
    for(int j=0; j<D; j++) 
        mean[j] /= N;

    Eigen::MatrixXf matrix(N, D);
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) {
            data[i*D+j] -= mean[j];
            matrix(i,j) = data[i*D+j];
        }
    }

    Eigen::MatrixXf cov_matrix = (matrix.transpose() * matrix) / float(matrix.rows() -1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov_matrix);

    Eigen::VectorXf eigen_values = solver.eigenvalues().real();

    Eigen::MatrixXf eigen_vectors = solver.eigenvectors().real();

    std::vector<EigenPair> eigen_pairs(D);
    for(int i=0; i<D; i++) {
        eigen_pairs[i].value = eigen_values[i];
        eigen_pairs[i].vec = std::vector<float>(D);
        for(int j=0; j<D; j++)
            eigen_pairs[i].vec[j] = eigen_vectors(j, i);
    }

    std::sort(eigen_pairs.begin(), eigen_pairs.end(), compareEigenPairs);


    float sum = 0;
    for(int i=0; i<D; i++) 
        sum += eigen_pairs[i].value;

    if(d <= 0) {

        if(ratio <= 0) {
            cout << "Wrong parameter: ratio" << endl;
            return;
        } 

        float total = 0;
        d = 0;
        for(; d<D; d++) {
            total += eigen_pairs[d].value;
            if(total >= sum*ratio) {
                ratio = total/sum;
                break;
            }
        }
        d+=1;
    }
    else {
        float total = 0;
        for(int i=0; i<d; i++) 
            total += eigen_pairs[i].value;
        ratio = total/sum;
    }

    pca_data = new float[D*D];
    for(int i=0; i<D; i++) {
        for(int j=0; j<D; j++)
            pca_data[i*D+j] = eigen_pairs[j].vec[i];
    }

    delete [] data;
    delete [] mean;
}

//为一组数据求PCA，并且保存特征向量，以及投影向量
void save_pca_index(float *data, int dim, int record_num, //原始数据
            float &ratio, int &column_num, //PCA 参数
            ofstream &ofs) {

    cout << "Start PCA ..." << endl;
    float *pca_data;
    float *trans_data = new float[record_num*column_num]{0};
    float *trans_data_remain = new float[record_num*(dim-column_num)]{0};
    // PCA(data, record_num, dim, ratio, column_num, pca_data); 
    PCA_CUDA(data, record_num, dim, ratio, column_num, pca_data);
    
    // transpose pca_data
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            std::swap(pca_data[i * dim + j], pca_data[j * dim + i]);
        }
    }

    try {
        projectPCA_CUDA(data, pca_data, trans_data, trans_data_remain, record_num, dim, column_num);
    } catch (const std::exception& e) {
         std::cerr << "!!! CUDA Projection Failed: " << e.what() << std::endl;
         // Cleanup memory before exiting
         delete[] data;
         delete[] pca_data;
         delete[] trans_data;
         delete[] trans_data_remain; // Safe even if nullptr
         return ; // Indicate error
    }

    cout << "PCA projection complete." << endl;
    
    
//     cout << "Start PCA projecting..." << endl;
    
// #pragma omp parallel for
//     for(int r=0; r<record_num; r++) {
//         for(int c=0; c<column_num; c++) {
//             for(int d=0; d<dim; d++)
//                 trans_data[r*column_num+c] += data[r*dim+d] * pca_data[d*dim+c];
//         }
//     }

//     cout << "Start PCA remain projecting..." << endl;
    
// #pragma omp parallel for
//     for(int r=0; r<record_num; r++) {
//         for(int c=0; c<dim-column_num; c++) {
//             for(int d=0; d<dim; d++)
//                 trans_data_remain[r*(dim-column_num)+c] += data[r*dim+d] * pca_data[d*dim+column_num+c];
//         }
//     }

    
    //保存 基础信息
    ofs.write((const char *)&dim, sizeof(int)); //数据维度，960
    ofs.write((const char *)&record_num, sizeof(int)); //数据向量个数
    ofs.write((const char *)&column_num, sizeof(int)); //特征向量个数，128
    ofs.write((const char *)&ratio, sizeof(float)); //比率
    ofs.write((const char *)pca_data, sizeof(float)*dim*dim); //保存所有特征向量，dim个特征向量

    ofs.write((const char *)trans_data, sizeof(float)*record_num*column_num);
    ofs.write((const char *)trans_data_remain, sizeof(float)*record_num*(dim-column_num));

    delete [] trans_data;
    delete [] trans_data_remain;
    delete [] pca_data;


    //yshen：优于 针对 trans_data_remain，如果再进行PCA，结果与 trans_data_remain 基本一致，即使shuffle，空间坐标基本一致，所以不再进行 PCA 2
}

void load_pca_index(ifstream &ifs, pca_index &index) {

    // Read dimensions and parameters
    if (!ifs.read((char *)&index.dim, sizeof(int)) || 
        !ifs.read((char *)&index.record_num, sizeof(int)) ||
        !ifs.read((char *)&index.column_num, sizeof(int)) ||
        !ifs.read((char *)&index.ratio, sizeof(float))) {
        std::cerr << "Error: Failed to read PCA index header from file." << std::endl;
        return;
    }

    // Validate the dimensions
    if (index.dim <= 0 || index.record_num <= 0 || index.column_num <= 0 || 
        index.column_num > index.dim || index.ratio <= 0 || index.ratio > 1) {
        std::cerr << "Error: Invalid PCA index parameters in file." << std::endl;
        return;
    }

    // Allocate memory for PCA data
    index.pca_data = static_cast<float*>(aligned_alloc(64, sizeof(float)*index.dim*index.dim));
    if (!index.pca_data) {
        std::cerr << "Error: Failed to allocate memory for PCA data." << std::endl;
        return;
    }

    // Read PCA data
    if (!ifs.read((char *)index.pca_data, sizeof(float)*index.dim*index.dim)) {
        std::cerr << "Error: Failed to read PCA data from file." << std::endl;
        free(index.pca_data);
        return;
    }

    // Allocate memory for transformed data
    index.trans_data = static_cast<float*>(aligned_alloc(64, sizeof(float)*index.record_num*index.column_num));
    if (!index.trans_data) {
        std::cerr << "Error: Failed to allocate memory for transformed data." << std::endl;
        free(index.pca_data);
        return;
    }

    // Read transformed data
    if (!ifs.read((char *)index.trans_data, sizeof(float)*index.record_num*index.column_num)) {
        std::cerr << "Error: Failed to read transformed data from file." << std::endl;
        free(index.pca_data);
        free(index.trans_data);
        return;
    }

    if(index.dim <= index.column_num) {
        index.trans_data_remain = nullptr;
        return;
    }
        
    // Allocate memory for remaining data
    index.trans_data_remain = static_cast<float*>(aligned_alloc(64, sizeof(float)*(index.record_num*(index.dim-index.column_num))));
    if (!index.trans_data_remain) {
        std::cerr << "Error: Failed to allocate memory for remaining data." << std::endl;
        free(index.pca_data);
        free(index.trans_data);
        return;
    }

    // Read remaining data
    if (!ifs.read((char *)index.trans_data_remain, sizeof(float)*index.record_num*(index.dim-index.column_num))) {
        std::cerr << "Error: Failed to read remaining data from file." << std::endl;
        free(index.pca_data);
        free(index.trans_data);
        free(index.trans_data_remain);
        return;
    }
}


//获取 query 在PCA上的投影
void queryProject(const float *query, pca_index &index, float *trans_data, float *trans_data_remain) {

    memset(trans_data, 0, sizeof(float)*index.column_num); 
    memset(trans_data_remain, 0, sizeof(float)*(index.dim-index.column_num)); 

    for(int c=0; c<index.column_num; c++) {
        for(int d=0; d<index.dim; d++) 
            trans_data[c] += query[d] * index.pca_data[d*index.dim+c];
    }

    for(int c=0; c<index.dim-index.column_num; c++) {
        for(int d=0; d<index.dim; d++) 
            trans_data_remain[c] += query[d] * index.pca_data[d*index.dim + index.column_num + c];
    }
}

//获取 query 在PCA上的主投影
void queryProjectMain(const float *query, pca_index &index, float *trans_data) {

    memset(trans_data, 0, sizeof(float)*index.column_num); 

    for(int c=0; c<index.column_num; c++) {
        for(int d=0; d<index.dim; d++) 
            trans_data[c] += query[d] * index.pca_data[d*index.dim+c];

        //yshen
        // trans_data[c] = query[c]; //不进行投影，用于全维度的时候，验证不投影情况下的recall，注意：column_num == dim
    }
}

//获取 query 在PCA上的remain部分的投影
void queryProjectRest(const float *query, pca_index &index, float *trans_data_remain) {

    memset(trans_data_remain, 0, sizeof(float)*(index.dim-index.column_num)); 

    for(int c=0; c<index.dim-index.column_num; c++) {
        for(int d=0; d<index.dim; d++) 
            trans_data_remain[c] += query[d] * index.pca_data[d*index.dim + index.column_num + c];
    }
}




void queryProjectRest2(const float *query, pca_index &index, float *trans_data_remain) {
    const int total_c = index.dim - index.column_num;
    memset(trans_data_remain, 0, sizeof(float) * total_c);

    const float *pca_data = index.pca_data;
    const int column_num = index.column_num;
    const int dim = index.dim;

    // AVX512 向量宽度（16 floats）
    const int vec_width = 16;
    const int num_blocks = total_c / vec_width;
    const int remainder = total_c % vec_width;

    // 分块处理主循环（每次处理16个c）
    for (int c_block = 0; c_block < num_blocks; c_block++) {
        const int c_start = c_block * vec_width;
        __m512 acc = _mm512_setzero_ps();  // 累加器初始化为0

        // 遍历所有行d
        for (int d = 0; d < dim; d++) {
            // 加载 query[d] 并广播到向量寄存器
            float qd = query[d];
            __m512 vec_query = _mm512_set1_ps(qd);

            // 加载 pca_data 中行d、列[column_num + c_start]的连续16个元素
            const float *pca_ptr = pca_data + d * dim + column_num + c_start;
            __m512 vec_pca = _mm512_loadu_ps(pca_ptr);

            // 乘积累加：acc += vec_query * vec_pca
            acc = _mm512_fmadd_ps(vec_query, vec_pca, acc);
        }

        // 存储结果到 trans_data_remain[c_start...c_start+15]
        _mm512_storeu_ps(trans_data_remain + c_start, acc);
    }

    // 处理剩余不足16个的c（掩码操作）
    if (remainder > 0) {
        const int c_start = num_blocks * vec_width;
        const __mmask16 mask = (1 << remainder) - 1;  // 生成掩码（例如余数5 → 0b0000000000011111）
        __m512 acc = _mm512_setzero_ps();

        for (int d = 0; d < dim; d++) {
            float qd = query[d];
            __m512 vec_query = _mm512_set1_ps(qd);

            // 使用掩码加载剩余元素
            const float *pca_ptr = pca_data + d * dim + column_num + c_start;
            __m512 vec_pca = _mm512_maskz_loadu_ps(mask, pca_ptr);

            // 掩码乘积累加
            acc = _mm512_mask_fmadd_ps(vec_query, mask, vec_pca, acc);
        }

        // 掩码存储结果
        _mm512_mask_storeu_ps(trans_data_remain + c_start, mask, acc);
    }
}
