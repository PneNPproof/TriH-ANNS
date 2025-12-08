/**
 * @file search.cpp
 * @brief Implementation of search algorithms and distance computation functions for TriH-ANNS
 * 
 * This file implements the core search functionality including:
 * - Euclidean distance calculation
 * - Batch distance computation with OpenMP parallelization
 * - Top-k candidate selection using sectioned approach
 * - Distance compensation algorithms for scalar quantization
 * - Quantization and dequantization utilities
 * 
 * The implementation focuses on optimizing search performance through:
 * - Parallel distance computation
 * - Memory-efficient top-k selection
 * - Various distance compensation strategies for quantized data
 */

#include <iostream>
#include <omp.h>
#include <float.h>
#include <cstring>

#include "gist.h"
#include "pca.h"
#include "search.h"

/**
 * @brief Compute Euclidean distance between two points
 * 
 * Calculates the squared L2 distance between two n-dimensional points.
 * The result is the squared distance (not the actual Euclidean distance)
 * for computational efficiency.
 * 
 * @param a First point coordinates
 * @param b Second point coordinates  
 * @param dimension Number of dimensions
 * @return Squared Euclidean distance between the points
 */
/**
 * @brief Compute Euclidean distance between two points
 * 
 * Calculates the squared L2 distance between two n-dimensional points.
 * The result is the squared distance (not the actual Euclidean distance)
 * for computational efficiency.
 * 
 * @param a First point coordinates
 * @param b Second point coordinates  
 * @param dimension Number of dimensions
 * @return Squared Euclidean distance between the points
 */
float euclideanDistance(const float a[], const float b[], int dimension) {
    float sum = 0.0;
    // Compute sum of squared differences for each dimension
    for (int i = 0; i < dimension; ++i) {
        sum += (a[i] - b[i])*(a[i] - b[i]);
    }

    return sum;
}

/**
 * @brief Compute distances from a query point to all base points
 * 
 * Calculates the Euclidean distance from a single query point to all points
 * in the base dataset. Uses OpenMP for parallel computation to improve performance.
 * 
 * @param query Query point coordinates
 * @param base Base dataset points (stored as consecutive float arrays)
 * @param dimension Number of dimensions per point
 * @param size Number of points in the base dataset
 * @param distances Output array to store computed distances
 */
void computeDistances(const float *query, const float *base, int dimension, int size, float distances[]) {

#pragma omp parallel for
    for(int i=0; i<size; i++) {
        // Compute distance from query to i-th base point
        distances[i] = euclideanDistance(query, base+i*dimension, dimension);
    }
}

//计算query到ids中下标对应的点的距离, size 为 ids 中点的数量，主要用于re-rank
// void computeDistances(const float *query, const float *base, int dimension, int *ids, int size, float distances[]) {

// #pragma omp parallel for
//     for(int i=0; i<size; i++) {
//         distances[i] = euclideanDistance(query, base+ids[i]*dimension, dimension);
//     }
// }

/**
 * @brief Select top-k nearest neighbors using sectioned approach
 * 
 * Divides the distance array into sections and finds the minimum distance
 * in each section, then selects the top-k smallest distances overall.
 * This sectioned approach helps reduce memory access patterns and can
 * improve cache performance for large datasets.
 * 
 * Algorithm:
 * 1. Divide distances array into sectionNum sections
 * 2. Find minimum distance and its index in each section
 * 3. Select top-k minimum distances from the section minimums
 * 
 * @param distances Array of computed distances
 * @param size Total number of distances
 * @param sectionNum Number of sections to divide the distances into
 * @param topk Number of top candidates to select
 * @param ids Output array storing indices of top-k nearest neighbors
 */
void computeTopkDistances(const float *distances, int size, int sectionNum, int topk, int *ids) {
    float min_dist;
    int min_id = -1;

    // Arrays to store minimum distance and index for each section
    float min_distances[sectionNum];
    int min_ids[sectionNum];

    // Find top-1 distance in each section
    int unit = size/sectionNum; // Number of elements per section
    for(int i=0; i<sectionNum; i++) {
        // Find minimum distance in current section
        min_dist = FLT_MAX;
        int start = i*unit;
        int end = start + unit;
        if(i == sectionNum - 1)
            end += size % unit; // Add remaining elements to last section

        // Scan current section for minimum distance
        for(int k=start; k<end; k++){
            if(min_dist > distances[k]) {
                min_dist = distances[k];
                min_id = k;
            }
        }

        min_distances[i] = min_dist;
        min_ids[i] = min_id;
    }

    // Select top-k from section minimums
    for(int i=0; i<topk; i++) {
        min_dist = FLT_MAX;
        // Find the section with current minimum distance
        for(int k=0; k<sectionNum; k++) {
            if(min_dist > min_distances[k]) {
                min_dist = min_distances[k];
                min_id = k; // Record which section has the minimum
            }
        }

        ids[i] = min_ids[min_id]; // Store global index of minimum

        min_distances[min_id] = FLT_MAX; // Mark as processed
    }
}

/**
 * @brief Distance compensation with asymmetric scalar quantization (Version 1)
 * 
 * Computes an approximation of the squared Euclidean distance using scalar
 * quantization on the data vector. This is used for distance compensation
 * when only partial dimensions are available in high precision.
 * 
 * Formula: ||x-q||² ≈ ||x||² - 2⟨q,sq(x)⟩ + ||q||²
 * where sq(x) is the scalar quantized version of x
 * 
 * @param query Query vector coordinates
 * @param x Data vector coordinates
 * @param dim Number of dimensions to process
 * @param sq_intervals Number of quantization intervals (0 = no quantization)
 * @return Approximated squared distance with quantization compensation
 */
float distance_compensation(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    // Compute ||x||² (can be precomputed for efficiency)
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;
    
    // Find min/max values for quantization range
    float max = FLT_MIN, min = FLT_MAX;
    for(int i=0; i<dim; i++) {
        if(x[i] > max) max = x[i];
        if(x[i] < min) min = x[i];
    }

    float scale = (max - min) / sq_intervals;

    // Asymmetric quantization with min as zero point
    float zero_point = min;

    // Apply scalar quantization to x
    float sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((x[i] - zero_point)/scale) * scale; 
    }

    // Compute -2⟨q,sq(x)⟩ term
    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*query[i] * sq[i];
    }

    dist -= tmp;

    return dist;
}


/**
 * @brief Distance compensation with integer arithmetic (Version 2)
 * 
 * Similar to distance_compensation but uses integer arithmetic for improved
 * numerical stability and potential performance gains. Scales floating point
 * values to integers before quantization.
 * 
 * @param query Query vector coordinates
 * @param x Data vector coordinates
 * @param dim Number of dimensions to process
 * @param sq_intervals Number of quantization intervals
 * @return Approximated squared distance with integer-based quantization
 */
float distance_compensation2(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    // Compute ||x||² (can be precomputed)
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;

    // Convert to integer representation for stability
    int xx[dim], qq[dim];
    for(int i=0; i<dim; i++) {
        xx[i] = x[i] * 100000;
        qq[i] = query[i] * 100000;
    }

    // Find min/max in integer space
    int max = -999999, min = 999999;
    for(int i=0; i<dim; i++) {
        if(xx[i] > max) max = xx[i];
        if(xx[i] < min) min = xx[i];
    }

    int unit = (max - min) / sq_intervals;

    // Asymmetric quantization in integer space
    int zero_point = min;

    int sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((xx[i] - zero_point)/unit) * unit + unit/2; 
    }

    // Compute inner product and scale back to float
    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*qq[i] * sq[i];
    }

    dist -= tmp/100000/100000;
    
    return dist;
}

/**
 * @brief Distance compensation with high-precision integer arithmetic (Version 3)
 * 
 * Uses 64-bit integers and higher scaling factor for maximum numerical
 * precision in quantization calculations. Includes safety checks for
 * division by zero.
 * 
 * @param query Query vector coordinates
 * @param x Data vector coordinates
 * @param dim Number of dimensions to process
 * @param sq_intervals Number of quantization intervals
 * @return Approximated squared distance with high-precision quantization
 */
float distance_compensation3(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    // Compute ||x||² (can be precomputed)
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;

    // Convert to high-precision integer representation
    long long xx[dim], qq[dim];
    for(int i=0; i<dim; i++) {
        xx[i] = x[i] * 1e7;
        qq[i] = query[i] * 1e7;
    }

    // Find range in integer space
    long long max = -99999999, min = 99999999;
    for(int i=0; i<dim; i++) {
        if(xx[i] > max) max = xx[i];
        if(xx[i] < min) min = xx[i];
    }

    long long unit = (max - min) / sq_intervals;
    if(unit == 0) unit = 1; // Prevent division by zero

    // Asymmetric quantization
    long long zero_point = min;

    long long sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((xx[i] - zero_point)/unit) * unit + unit/2; 
    }

    // Compute inner product with proper scaling
    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*qq[i] * sq[i];
    }

    dist -= tmp/1e14;

    return dist;
}


/**
 * @brief Distance compensation with rounded quantization (Version 3.1)
 * 
 * Improved version of distance compensation that uses proper rounding
 * in the quantization process for better accuracy. Includes commented
 * alternative implementation using grouping optimization.
 * 
 * @param query Query vector coordinates
 * @param x Data vector coordinates
 * @param dim Number of dimensions to process
 * @param sq_intervals Number of quantization intervals
 * @return Approximated squared distance with rounded quantization
 */
float distance_compensation3_1(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    // Compute ||x||² (can be precomputed)
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;
    
    // Find value range for quantization
    float max = FLT_MIN, min = FLT_MAX;
    for(int i=0; i<dim; i++) {
        if(x[i] > max) max = x[i];
        if(x[i] < min) min = x[i];
    }

    float scale = (max - min) / sq_intervals;

    // Asymmetric quantization with proper rounding
    float zero_point = min;

    // Method 1: Direct quantization with rounding
    float sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((x[i] - zero_point)/scale+0.5) * scale; 
    }

    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*query[i] * sq[i];
    }

    dist -= tmp;

    // Method 2: Optimized computation by grouping similar quantized values
    // This commented code shows an alternative approach that groups
    // dimensions with the same quantized value to reduce computation
    /*
    int sq_ind[960]={0};
    for(int i=0; i<dim; i++) {
        sq_ind[i] = (int)((x[i] - zero_point)/scale + 0.5); 
    }

    for(int i=0; i<dim; i++) {
        float cum_query = 0;
        if(sq_ind[i] == -1) continue;

        int cur_scales = sq_ind[i];
        sq_ind[i] = -1;
        
        cum_query += query[i];

        for(int k=i+1; k<dim; k++) {
            if(cur_scales == sq_ind[k]) {
                sq_ind[k] = -1;
                cum_query += query[k];
            }
        }

        dist -= 2*cum_query*(zero_point + cur_scales * scale);
    }
    */

    return dist;
}


/**
 * @brief Distance compensation using pre-quantized data (Version 4)
 * 
 * Computes distance compensation when the base data is already quantized
 * to uint8_t format. This version quantizes the query on-the-fly and
 * computes the distance in quantized space.
 * 
 * @param quant_base Pre-quantized base dataset
 * @param idx Index of the base point to compare with
 * @param query Query vector coordinates (full precision)
 * @param dim Number of dimensions for compensation
 * @param scale Quantization scale factor
 * @param zp Zero point for quantization
 * @param b Number of quantization bits (levels)
 * @return Scaled distance in original space
 */
float distance_compensation4(uint8_t *quant_base, int idx, float *query, int dim, float scale, float zp, int b) {
    float dist = 0;

    // Quantize query vector using same parameters as base
    uint8_t *quant_query = new uint8_t[dim];
    quant_point(query, quant_query, scale, zp, dim, b);

    // Compute distance in quantized space
    uint8_t *p = quant_base+idx*dim;
    for (int i = 0; i < dim; i++) {
        dist += (p[i] - quant_query[i]) * (p[i] - quant_query[i]);
    }

    // Scale back to original space for distance compensation
    return dist*scale*scale;

    // Alternative implementations (commented):
    // 1. Direct dequantization approach
    // 2. Inner product computation
    // Both methods are preserved for reference
}


/**
 * @brief Exact distance computation (Version 5)
 * 
 * Computes the exact Euclidean distance without any quantization.
 * This serves as a baseline or fallback method when quantization
 * is not needed or for accuracy comparison.
 * 
 * @param base Base dataset points
 * @param idx Index of the base point to compare with
 * @param query Query vector coordinates
 * @param dim Number of dimensions
 * @param scale Unused (for interface compatibility)
 * @param zp Unused (for interface compatibility)
 * @param b Unused (for interface compatibility)
 * @return Exact squared Euclidean distance
 */
float distance_compensation5(float *base, int idx, float *query, int dim, float scale, float zp, int b) {
    float dist = 0;

    // Compute exact squared Euclidean distance
    float *p = base+idx*dim;
    for(int i=0; i<dim; i++) {
        dist += (p[i]-query[i])*(p[i]-query[i]);
    }

    return dist;
}


/**
 * @brief Clip a value to quantization range
 * 
 * Ensures the quantized value falls within the specified range [start, end].
 * Used as a helper function for quantization operations.
 * 
 * @param x Input floating point value
 * @param scale Quantization scale factor
 * @param zp Zero point offset
 * @param start Minimum allowed quantized value
 * @param end Maximum allowed quantized value
 * @return Clipped quantized value as uint8_t
 */
uint8_t clip(float x, float scale, float zp, int start, int end)
{
    float y = x / scale + zp;
    if (y < start)
        return start;
    if (y > end)
        return end;
    return (uint8_t)y;
}

/**
 * @brief Quantize a floating point vector to uint8_t
 * 
 * Converts a vector of floating point values to quantized uint8_t values
 * using the provided scale, zero point, and bit width parameters.
 * 
 * @param point Input floating point vector
 * @param quant_point Output quantized vector
 * @param scale Quantization scale factor
 * @param zp Zero point offset
 * @param dimension Number of dimensions to quantize
 * @param b Number of quantization levels (2^bits)
 */
void quant_point(const float *point, uint8_t *quant_point, float scale, float zp, int dimension, int b)
{
    for (int i = 0; i < dimension; i++)
    {
        quant_point[i] = clip(point[i], scale, zp, 0, b - 1);
    }
}

/**
 * @brief Dequantize a uint8_t vector back to floating point
 * 
 * Converts quantized uint8_t values back to floating point representation
 * using the inverse quantization formula: value = (quantized - zp) * scale
 * 
 * @param quant_point Input quantized vector
 * @param point Output floating point vector
 * @param scale Quantization scale factor
 * @param zp Zero point offset
 * @param dimension Number of dimensions to dequantize
 */
void dequant_point(uint8_t *quant_point, float *point, float scale, float zp, int dimension)
{
    for (int i = 0; i < dimension; i++)
    {
        point[i] = (quant_point[i] - zp) * scale;
    }
}