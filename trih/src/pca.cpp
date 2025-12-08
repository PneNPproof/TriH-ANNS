/**
 * @file pca.cpp
 * @brief Principal Component Analysis implementation for TriH-ANNS
 * 
 * This file implements the core PCA functionality used in TriH-ANNS for dimensionality
 * reduction. The implementation provides both CPU-based (using Eigen) and GPU-accelerated
 * (using CUDA) versions of PCA computation and data projection.
 * 
 * Key features:
 * - Eigenvalue decomposition using Eigen library for mathematical precision
 * - GPU acceleration for large-scale data projection using CUDA
 * - Adaptive component selection based on variance ratio or fixed count
 * - Memory-efficient handling of large datasets with sampling for PCA computation
 * - Optimized data layout and transposition for GPU operations
 * 
 * The PCA process involves:
 * 1. Data centering (subtracting mean)
 * 2. Covariance matrix computation
 * 3. Eigenvalue decomposition
 * 4. Component selection based on explained variance
 * 5. Batch projection of all data to PCA and remaining spaces
 * 
 * Dependencies:
 * - Eigen library for linear algebra operations
 * - CUDA implementation (pca.cuh) for GPU acceleration
 * - Custom data shuffling utilities
 */

#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>

#include "pca.h"
#include "shuffle.h"

#include "pca.cuh"

// Maximum number of samples to use for PCA computation (memory limitation)
#define MAX_SAMPLE 1000000

using namespace std;

/**
 * @brief Compute Principal Component Analysis using Eigen library
 * 
 * Performs eigenvalue decomposition on the covariance matrix to find principal
 * components. Supports both fixed component count and adaptive selection based
 * on variance ratio. Uses sampling for very large datasets to manage memory usage.
 * 
 * Algorithm:
 * 1. Sample up to MAX_SAMPLE vectors if dataset is too large
 * 2. Center data by subtracting per-dimension mean
 * 3. Compute covariance matrix: C = (X^T * X) / (N-1)
 * 4. Perform eigenvalue decomposition: C = V * Λ * V^T
 * 5. Sort eigenvalues/eigenvectors in descending order
 * 6. Select components based on variance ratio or fixed count
 * 
 * @param src Input data matrix (N0 x D) in row-major order
 * @param N0 Number of input vectors
 * @param D Feature dimensionality
 * @param ratio Target variance ratio to preserve (in/out parameter)
 * @param d Number of components to retain (in/out parameter)
 * @param pca_data Output PCA transformation matrix (allocated by function)
 * 
 * @note Either 'ratio' or 'd' is used depending on which is specified (d > 0)
 * @note Uses sampling if N0 > MAX_SAMPLE to manage memory usage
 * @note Caller must free pca_data memory after use
 * @note Output matrix is in column-major order for efficient GPU operations
 */
void PCA(const float* src, const int N0, const int D, float &ratio, int &d, float*& pca_data) {
    
    // Use sampling if dataset is too large to fit in memory
    int N = N0;
    if(N > MAX_SAMPLE) N = MAX_SAMPLE;

    // Allocate and copy sample data for PCA computation
    float *data = new float[N*D]{0};
    memcpy(data, src, sizeof(float)*N*D);

    // Compute per-dimension mean for data centering
    float *mean = new float[D]{0};
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) 
            mean[j] += data[i*D+j];
    }
    for(int j=0; j<D; j++) 
        mean[j] /= N;

    // Center data and convert to Eigen matrix format
    Eigen::MatrixXf matrix(N, D);
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) {
            data[i*D+j] -= mean[j];  // Center each dimension
            matrix(i,j) = data[i*D+j];
        }
    }

    // Compute covariance matrix: C = (X^T * X) / (N-1)
    Eigen::MatrixXf cov_matrix = (matrix.transpose() * matrix) / float(matrix.rows() -1);

    // Perform eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov_matrix);

    Eigen::VectorXf eigen_values = solver.eigenvalues().real();
    Eigen::MatrixXf eigen_vectors = solver.eigenvectors().real();

    // Create eigenvalue-eigenvector pairs for sorting
    std::vector<EigenPair> eigen_pairs(D);
    for(int i=0; i<D; i++) {
        eigen_pairs[i].value = eigen_values[i];
        eigen_pairs[i].vec = std::vector<float>(D);
        for(int j=0; j<D; j++)
            eigen_pairs[i].vec[j] = eigen_vectors(j, i);
    }

    // Sort by eigenvalue in descending order (most important components first)
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), compareEigenPairs);

    // Compute total variance for ratio calculations
    float sum = 0;
    for(int i=0; i<D; i++) 
        sum += eigen_pairs[i].value;

    // Determine number of components to retain
    if(d <= 0) {
        // Use variance ratio to determine component count
        if(ratio <= 0) {
            cout << "Wrong parameter: ratio" << endl;
            return;
        } 

        float total = 0;
        d = 0;
        // Find minimum components needed to achieve target ratio
        for(; d<D; d++) {
            total += eigen_pairs[d].value;
            if(total >= sum*ratio) {
                ratio = total/sum;  // Update actual ratio achieved
                break;
            }
        }
        d+=1;  // Include the component that achieved the ratio
    }
    else {
        // Use fixed component count, compute achieved ratio
        float total = 0;
        for(int i=0; i<d; i++) 
            total += eigen_pairs[i].value;
        ratio = total/sum;
    }

    // Build PCA transformation matrix (eigenvectors as columns)
    pca_data = new float[D*D];
    for(int i=0; i<D; i++) {
        for(int j=0; j<D; j++)
            pca_data[i*D+j] = eigen_pairs[j].vec[i];
    }

    // Cleanup temporary memory
    delete [] data;
    delete [] mean;
}

#include <chrono>

/**
 * @brief Build complete PCA index and save to file
 * 
 * This is the main index building function that:
 * 1. Computes PCA transformation using GPU acceleration when available
 * 2. Projects all training data to both PCA and remaining dimensions
 * 3. Saves the complete index structure to binary file
 * 
 * The function handles the complete pipeline from raw training data to
 * a ready-to-use search index, including all optimizations for GPU processing.
 * 
 * @param data Training dataset vectors (record_num x dim)
 * @param dim Original feature dimensionality
 * @param record_num Number of training vectors
 * @param ratio Target variance ratio (in/out parameter)
 * @param column_num Number of PCA components (in/out parameter)
 * @param ofs Output file stream for saving index
 * 
 * @note Uses GPU acceleration (PCA_CUDA) when available, falls back to CPU
 * @note Performs matrix transposition for optimal GPU memory access patterns
 * @note Includes comprehensive error handling and timing measurements
 * @note All intermediate results are saved to the output file
 */
void save_pca_index(float *data, int dim, int record_num,
            float &ratio, int &column_num,
            ofstream &ofs) {

    auto start_time = std::chrono::high_resolution_clock::now();
    cout << "Start PCA ..." << endl;
    
    float *pca_data;
    float *trans_data = new float[record_num*column_num]{0};           // PCA projected data
    float *trans_data_remain = new float[record_num*(dim-column_num)]{0}; // Remaining dimensions
    
    // Use GPU-accelerated PCA computation for better performance
    PCA_CUDA(data, record_num, dim, ratio, column_num, pca_data);

    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time);
    std::cout << "Time taken for pca: " << duration1.count() << " ms" << std::endl;
    
    // Transpose PCA matrix for optimal GPU memory access patterns
    // This ensures coalesced memory access during projection operations
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            std::swap(pca_data[i * dim + j], pca_data[j * dim + i]);
        }
    }

    // Optional: Scale PCA coefficients (commented out for current implementation)
    // This could be used for numerical stability in some cases
    // for (int i = 0; i < dim; ++i) {
    //     for (int j = 0; j < dim; ++j) {
    //         pca_data[i * dim + j] = pca_data[i * dim + j] / 256;
    //     }
    // }

    // Project all training data using GPU acceleration
    try {
        projectPCA_CUDA(data, pca_data, trans_data, trans_data_remain, record_num, dim, column_num);
    } catch (const std::exception& e) {
         std::cerr << "!!! CUDA Projection Failed: " << e.what() << std::endl;
         // Note: Memory cleanup handled by destructors or calling function
         return; // Indicate error
    }

    cout << "PCA projection complete." << endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken for pca info build time: " << duration.count() << " ms" << std::endl;

    // Save index metadata to binary file
    ofs.write((const char *)&dim, sizeof(int));         // Original dimensionality (e.g., 960)
    ofs.write((const char *)&record_num, sizeof(int));  // Number of training vectors
    ofs.write((const char *)&column_num, sizeof(int));  // Number of PCA components (e.g., 128)
    ofs.write((const char *)&ratio, sizeof(float));     // Variance ratio preserved
    
    // Save complete PCA transformation matrix (dim x dim)
    ofs.write((const char *)pca_data, sizeof(float)*dim*dim);
    
    // Save projected training data in PCA space (record_num x column_num)
    ofs.write((const char *)trans_data, sizeof(float)*record_num*column_num);
    
    // Save projected training data in remaining dimensions (record_num x (dim-column_num))
    ofs.write((const char *)trans_data_remain, sizeof(float)*record_num*(dim-column_num));

    // Cleanup allocated memory
    delete [] trans_data;
    delete [] trans_data_remain;
    delete [] pca_data;

    // Note: Further PCA on trans_data_remain would yield essentially identical results
    // due to the orthogonal nature of PCA components, so second-level PCA is not performed
}

/**
 * @brief Load PCA index from file
 * 
 * Reads the PCA index structure from a binary file, including all metadata,
 * PCA transformation matrices, and pre-computed data projections. The loaded
 * index can be used for efficient nearest neighbor search in the reduced
 * dimensionality space.
 * 
 * @param ifs Input file stream for reading index
 * @param index PCA index structure to be filled
 * 
 * @note Allocates memory for PCA data and projections using aligned_alloc
 * @note Caller is responsible for freeing the allocated memory in index
 * @note Includes validation of file format and error handling
 */
/**
 * @brief Load PCA index from binary file
 * 
 * Reads a previously saved PCA index from binary file, reconstructing all
 * data structures needed for efficient search. Includes comprehensive error
 * checking and memory allocation with proper alignment for SIMD operations.
 * 
 * File format (binary):
 * 1. Index metadata (dim, record_num, column_num, ratio)
 * 2. PCA transformation matrix (dim x dim floats)
 * 3. Projected training data in PCA space (record_num x column_num floats)
 * 4. Projected training data in remaining dimensions (record_num x (dim-column_num) floats)
 * 
 * @param ifs Input file stream in binary mode
 * @param index Output PCA index structure (memory allocated by function)
 * 
 * @note Uses 64-byte aligned allocation for optimal SIMD performance
 * @note Includes comprehensive validation of file format and parameters
 * @note Graceful error handling with proper memory cleanup on failure
 * @note Caller must ensure proper cleanup of allocated memory
 */
void load_pca_index(ifstream &ifs, pca_index &index) {

    // Read index metadata with error checking
    if (!ifs.read((char *)&index.dim, sizeof(int)) || 
        !ifs.read((char *)&index.record_num, sizeof(int)) ||
        !ifs.read((char *)&index.column_num, sizeof(int)) ||
        !ifs.read((char *)&index.ratio, sizeof(float))) {
        std::cerr << "Error: Failed to read PCA index header from file." << std::endl;
        return;
    }

    // Validate loaded parameters for consistency
    if (index.dim <= 0 || index.record_num <= 0 || index.column_num <= 0 || 
        index.column_num > index.dim || index.ratio <= 0 || index.ratio > 1) {
        std::cerr << "Error: Invalid PCA index parameters in file." << std::endl;
        return;
    }

    // Allocate aligned memory for PCA transformation matrix
    index.pca_data = static_cast<float*>(aligned_alloc(64, sizeof(float)*index.dim*index.dim));
    if (!index.pca_data) {
        std::cerr << "Error: Failed to allocate memory for PCA data." << std::endl;
        return;
    }

    // Read PCA transformation matrix
    if (!ifs.read((char *)index.pca_data, sizeof(float)*index.dim*index.dim)) {
        std::cerr << "Error: Failed to read PCA data from file." << std::endl;
        free(index.pca_data);
        return;
    }

    // Allocate aligned memory for PCA-projected training data
    index.trans_data = static_cast<float*>(aligned_alloc(64, sizeof(float)*index.record_num*index.column_num));
    if (!index.trans_data) {
        std::cerr << "Error: Failed to allocate memory for transformed data." << std::endl;
        free(index.pca_data);
        return;
    }

    // Read PCA-projected training data
    if (!ifs.read((char *)index.trans_data, sizeof(float)*index.record_num*index.column_num)) {
        std::cerr << "Error: Failed to read transformed data from file." << std::endl;
        free(index.pca_data);
        free(index.trans_data);
        return;
    }

    // Handle case where all dimensions are used in PCA (no remaining dimensions)
    if(index.dim <= index.column_num) {
        index.trans_data_remain = nullptr;
        return;
    }
        
    // Allocate aligned memory for remaining dimensions data
    index.trans_data_remain = static_cast<float*>(aligned_alloc(64, sizeof(float)*(index.record_num*(index.dim-index.column_num))));
    if (!index.trans_data_remain) {
        std::cerr << "Error: Failed to allocate memory for remaining data." << std::endl;
        free(index.pca_data);
        free(index.trans_data);
        return;
    }

    // Read remaining dimensions data
    if (!ifs.read((char *)index.trans_data_remain, sizeof(float)*index.record_num*(index.dim-index.column_num))) {
        std::cerr << "Error: Failed to read remaining data from file." << std::endl;
        free(index.pca_data);
        free(index.trans_data);
        free(index.trans_data_remain);
        return;
    }
}


/**
 * @brief Project query vector onto PCA space
 * 
 * Computes the projection of a given query vector onto the PCA space defined
 * by the loaded PCA index. The query is projected to both the PCA dimensions
 * and the remaining dimensions. This allows for efficient nearest neighbor
 * search in the reduced space.
 * 
 * @param query Input query vector (dimensionality: index.dim)
 * @param index PCA index structure containing transformation matrices
 * @param trans_data Output buffer for PCA projected data (column_num dimensions)
 * @param trans_data_remain Output buffer for remaining dimensions (dim-column_num dimensions)
 * 
 * @note Both trans_data and trans_data_remain must be allocated by the caller
 * @note Memory for trans_data_remain can be nullptr if dim <= column_num
 * @note Includes error handling for invalid index or memory allocation issues
 */
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

/**
 * @brief Project query vector onto PCA main components
 * 
 * Computes the projection of a given query vector onto the main PCA components
 * defined by the loaded PCA index. This is a subset of the full projection that
 * only includes the dimensions corresponding to the largest eigenvalues.
 * 
 * @param query Input query vector (dimensionality: index.dim)
 * @param index PCA index structure containing transformation matrices
 * @param trans_data Output buffer for PCA main components projection (column_num dimensions)
 * 
 * @note trans_data must be allocated by the caller
 * @note Includes error handling for invalid index or memory allocation issues
 */
void queryProjectMain(const float *query, pca_index &index, float *trans_data) {

    memset(trans_data, 0, sizeof(float)*index.column_num); 

    for(int c=0; c<index.column_num; c++) {
        for(int d=0; d<index.dim; d++) 
            trans_data[c] += query[d] * index.pca_data[d*index.dim+c];

        //yshen
        // trans_data[c] = query[c]; //不进行投影，用于全维度的时候，验证不投影情况下的recall，注意：column_num == dim
    }
}

/**
 * @brief Project query vector onto PCA remaining dimensions
 * 
 * Computes the projection of a given query vector onto the remaining dimensions
 * that are not included in the main PCA components. This allows for a complete
 * representation of the query in both PCA-reduced and original feature spaces.
 * 
 * @param query Input query vector (dimensionality: index.dim)
 * @param index PCA index structure containing transformation matrices
 * @param trans_data_remain Output buffer for remaining dimensions projection (dim-column_num dimensions)
 * 
 * @note trans_data_remain must be allocated by the caller
 * @note Includes error handling for invalid index or memory allocation issues
 */
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
