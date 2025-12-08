/**
 * @file pca.h
 * @brief Principal Component Analysis (PCA) interface for TriH-ANNS dimensionality reduction
 * 
 * This header defines the core PCA functionality used in TriH-ANNS for:
 * 1. Index building: Computing PCA transformation from training data
 * 2. Query processing: Projecting queries into PCA space for efficient search
 * 3. Two-phase search: Separating dimensions into primary (PCA) and remaining components
 * 
 * The PCA approach enables efficient GPU-based nearest neighbor search by:
 * - Reducing dimensionality for initial candidate selection
 * - Preserving most important variance in lower dimensions
 * - Enabling precise reranking using full dimensional data
 * 
 * Dependencies:
 * - Eigen library for matrix operations and eigenvalue computation
 * - Standard C++ containers for data management
 */

#pragma once

#ifndef __PCA_H__
#define __PCA_H__

#include <vector>
#include <fstream>

using namespace std;

/**
 * @brief Eigenvalue-eigenvector pair for PCA computation
 * 
 * Stores eigenvalue and corresponding eigenvector from covariance matrix decomposition.
 * Used for sorting and selecting principal components based on explained variance.
 */
struct EigenPair {
    float value;            ///< Eigenvalue (variance explained by this component)
    vector<float> vec;      ///< Eigenvector (principal component direction)
};

/**
 * @brief Complete PCA index structure for TriH-ANNS
 * 
 * Contains all information needed for PCA-based search:
 * - Original data metadata
 * - PCA transformation matrix
 * - Projected training data in both primary and remaining dimensions
 * 
 * Memory management: Uses RAII with destructor cleanup
 */
struct pca_index {
    // Original dataset metadata
    int dim;                    ///< Original feature dimensionality (e.g., 960 for GIST)
    int record_num;             ///< Number of training vectors in the index
    
    // PCA transformation parameters
    float *pca_data;           ///< Full PCA transformation matrix (dim x dim)
    int column_num;            ///< Number of retained principal components (e.g., 128)
    float ratio;               ///< Cumulative variance ratio preserved by selected components
    
    // Projected training data
    float *trans_data;         ///< Training data projected to PCA space (record_num x column_num)
    float *trans_data_remain;  ///< Training data in remaining dimensions (record_num x (dim-column_num))

    /**
     * @brief Default constructor - initializes pointers to null
     */
    pca_index() {
        pca_data = nullptr;
        trans_data = nullptr;
        trans_data_remain = nullptr;
    }

    /**
     * @brief Destructor - releases allocated memory
     * 
     * Ensures proper cleanup of large memory allocations to prevent leaks.
     * Uses null checks to handle partially initialized objects safely.
     */
    ~pca_index() {   
        if(pca_data != nullptr) {
            delete [] pca_data;
            pca_data = nullptr;
        }

        if(trans_data != nullptr) {
            delete [] trans_data;
            trans_data = nullptr;
        }

        if(trans_data_remain != nullptr) {
            delete [] trans_data_remain;
            trans_data_remain = nullptr;
        }
    }
};

/**
 * @brief Comparison function for sorting eigenvalue-eigenvector pairs
 * 
 * @param a First eigenvalue-eigenvector pair
 * @param b Second eigenvalue-eigenvector pair
 * @return true if a.value > b.value (descending order by eigenvalue)
 * 
 * Used to sort principal components by explained variance in descending order.
 */
inline bool compareEigenPairs(const EigenPair &a, const EigenPair &b){
    return a.value > b.value;
}

/**
 * @brief Compute PCA transformation from training data
 * 
 * Performs eigenvalue decomposition of covariance matrix to find principal components.
 * Selects components based on either fixed count or variance ratio threshold.
 * 
 * @param src Training data matrix (N0 x D)
 * @param N0 Number of training vectors
 * @param D Original dimensionality
 * @param ratio Target variance ratio (used if d=0)
 * @param d Target number of components (if >0, overrides ratio)
 * @param pca_data Output PCA transformation matrix (allocated by function)
 * 
 * @note Caller must free pca_data memory
 * @note Uses Eigen library for efficient matrix operations
 */
void PCA(const float* src, const int N0, const int D, float &ratio, int &d, float*& pca_data);

/**
 * @brief Build and save complete PCA index to file
 * 
 * Computes PCA transformation and projects all training data to both
 * primary (PCA) and remaining dimensions. Saves complete index to binary file.
 * 
 * @param data Training data vectors
 * @param dim Original dimensionality
 * @param record_num Number of training vectors
 * @param ratio Target variance ratio
 * @param column_num Target number of PCA components
 * @param ofs Output file stream (binary mode)
 * 
 * @note File format: metadata + PCA matrix + projected data
 * @note Both ratio and column_num are in/out parameters
 */
void save_pca_index(float *data, int dim, int record_num,
            float &ratio, int &column_num,
            ofstream &ofs);

/**
 * @brief Load previously saved PCA index from file
 * 
 * @param ifs Input file stream (binary mode)
 * @param index Output PCA index structure (memory allocated by function)
 * 
 * @note Must match file format created by save_pca_index
 * @note Caller responsible for cleaning up index memory
 */
void load_pca_index(ifstream &ifs, pca_index &index);

/**
 * @brief Project query vector to both PCA and remaining dimensions
 * 
 * Complete query projection for two-phase search approach.
 * 
 * @param query Input query vector (dim dimensions)
 * @param index PCA index containing transformation
 * @param trans_data Output projected query in PCA space (column_num dimensions)
 * @param trans_data_remain Output projected query in remaining space ((dim-column_num) dimensions)
 */
void queryProject(const float *query, pca_index &index, float *trans_data, float *trans_data_remain);

/**
 * @brief Project query vector to PCA dimensions only
 * 
 * @param query Input query vector
 * @param index PCA index containing transformation
 * @param trans_data Output projected query in PCA space
 */
void queryProjectMain(const float *query, pca_index &index, float *trans_data);

/**
 * @brief Project query vector to remaining dimensions only
 * 
 * @param query Input query vector
 * @param index PCA index containing transformation
 * @param trans_data_remain Output projected query in remaining space
 */
void queryProjectRest(const float *query, pca_index &index, float *trans_data_remain);

/**
 * @brief Alternative implementation for remaining dimensions projection
 * 
 * @param query Input query vector
 * @param index PCA index containing transformation
 * @param trans_data_remain Output projected query in remaining space
 * 
 * @note May use different algorithm or optimization compared to queryProjectRest
 */
void queryProjectRest2(const float *query, pca_index &index, float *trans_data_remain);

#endif