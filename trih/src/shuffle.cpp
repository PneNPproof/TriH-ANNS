/**
 * @file shuffle.cpp
 * @brief Implementation of vector shuffling algorithms for dataset randomization
 * 
 * This file provides functions to randomly shuffle vector datasets while
 * maintaining consistency with associated metadata (such as ground truth
 * neighbor indices). Shuffling is commonly used in machine learning to
 * eliminate ordering bias and improve training/evaluation robustness.
 * 
 * Features:
 * - Fisher-Yates shuffle algorithm for uniform randomization
 * - Simultaneous shuffling of vectors and ID arrays
 * - Advanced shuffling with neighbor index correction
 * - Memory-efficient in-place operations
 */

#include <random>
#include <cstring>
#include <iostream>
#include <vector>

/**
 * @brief Basic Fisher-Yates shuffle for vectors with ID tracking
 * 
 * Performs an in-place random shuffle of a vector dataset using the
 * Fisher-Yates algorithm. Simultaneously updates an ID array to track
 * the new positions of the original vectors.
 * 
 * Algorithm:
 * 1. For each position i from size-1 down to 1
 * 2. Choose random position ex in [0, i-1]
 * 3. Swap vectors at positions i and ex
 * 4. Update corresponding ID mappings
 * 
 * @param vectors Array of vectors to shuffle (size × dim elements)
 * @param dim Dimension of each vector
 * @param size Number of vectors to shuffle
 * @param ids Array tracking vector positions (updated during shuffle)
 * 
 * @note Uses system time as random seed
 * @note Memory layout: vectors stored as consecutive float arrays
 */
void myshuffle(float *vectors, int dim, int size, int *ids) {
    int ex;
    float vec_tmp[dim];  // Temporary buffer for vector swapping
    int tmp;

    srand(time(NULL));   // Initialize random seed

    // Fisher-Yates shuffle algorithm
    for(int i=size-1; i>1; i--){
        ex = rand() % i; // Random index in [0, i-1]

        // Swap vectors at positions i and ex
        memcpy(vec_tmp, vectors+i*dim, sizeof(float)*dim);
        memcpy(vectors+i*dim, vectors+ex*dim, sizeof(float)*dim);
        memcpy(vectors+ex*dim, vec_tmp, sizeof(float)*dim);

        // Update ID tracking array
        tmp = ids[i];
        ids[i] = ids[ex];
        ids[ex] = tmp;
    }
}

/**
 * @brief Advanced shuffle with neighbor index correction
 * 
 * Performs vector shuffling while maintaining consistency with an external
 * array (typically ground truth neighbor indices). Uses efficient mapping
 * tables to track position changes and update all references in parallel.
 * 
 * Key improvements over basic shuffle:
 * - Maintains bidirectional location mapping
 * - Parallel correction of external references
 * - Efficient tracking of position changes
 * 
 * @param vectors Array of vectors to shuffle (size × dim elements)
 * @param dim Dimension of each vector
 * @param size Number of vectors to shuffle
 * @param others External array to update (e.g., neighbor indices)
 * @param others_size Size of the external array
 * 
 * @note Uses OpenMP for parallel processing of external array updates
 * @note Memory complexity: O(size) for mapping tables
 */
void myshuffle2(float *vectors, int dim, int size, int *others, int others_size) {
    int ex;
    float vec_tmp[dim];  // Temporary buffer for vector swapping

    srand(time(NULL));   // Initialize random seed

    // Create bidirectional mapping tables
    std::vector<int> loc_2_org_loc(size);   // Current location -> original location
    for(int i=0; i<size; i++) loc_2_org_loc[i] = i;
    
    std::vector<int> org_loc_2_loc(size);   // Original location -> current location
    for(int i=0; i<size; i++) org_loc_2_loc[i] = i;

    // Fisher-Yates shuffle with mapping table updates
    for(int i=size-1; i>1; i--){
        ex = rand() % i; // Random index in [0, i-1]

        // Update mapping tables for the swap
        auto org_loc_1 = loc_2_org_loc[i];
        auto org_loc_2 = loc_2_org_loc[ex];
        loc_2_org_loc[i] = org_loc_2;
        loc_2_org_loc[ex] = org_loc_1;
        org_loc_2_loc[org_loc_1] = ex;
        org_loc_2_loc[org_loc_2] = i;

        // Swap vectors at positions i and ex
        memcpy(vec_tmp, vectors+i*dim, sizeof(float)*dim);
        memcpy(vectors+i*dim, vectors+ex*dim, sizeof(float)*dim);
        memcpy(vectors+ex*dim, vec_tmp, sizeof(float)*dim);
    }

    // Update external array using mapping table (parallel processing)
    #pragma omp parallel for
    for(int k=0; k<others_size; k++) {
        auto org_loc = others[k];           // Original location referenced
        others[k] = org_loc_2_loc[org_loc]; // Update to new location
    }
}