#ifndef PCA_CUH
#define PCA_CUH

#include <cuda_runtime.h>

// Host function declarations
void PCA_CUDA(const float *src, const int N0, const int D, float &ratio, int &d, float *&pca_data_host);

// Forward declaration of the function
void projectPCA_CUDA(const float *h_data,        // Host input data
                     const float *h_pca_data,    // Host transposed PCA matrix
                     float *h_trans_data,        // Host output projected data
                     float *h_trans_data_remain, // Host output remaining projected data
                     int record_num,
                     int dim,
                     int column_num);

void projectPCA_CUDA_cuBLAS(const float *h_data,        // Host input data (row-major, record_num x dim)
                            const float *h_pca_data,    // Host PCA matrix (col-major components, dim x dim)
                            float *h_trans_data,        // Host output projected data (row-major, record_num x column_num)
                            float *h_trans_data_remain, // Host output remaining projected data (row-major, record_num x (dim-col))
                            int record_num,
                            int dim,
                            int column_num);

#endif // PCA_CUH