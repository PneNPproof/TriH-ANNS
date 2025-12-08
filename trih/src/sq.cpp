/**
 * @file sq.cpp
 * @brief Implementation of scalar quantization algorithms for TriH-ANNS
 * 
 * This file implements scalar quantization (SQ) functionality used to compress
 * high-dimensional vectors while preserving distance relationships. Scalar 
 * quantization reduces memory usage and can accelerate similarity computations
 * through efficient integer arithmetic.
 * 
 * Key features:
 * - 8-bit scalar quantization with configurable bit width
 * - Min-max normalization for optimal quantization range
 * - Precomputed quantization information for fast distance compensation
 * - Support for partial quantization (useful for hybrid approaches)
 * - OpenMP parallelization for large-scale quantization
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>  // For FLT_MAX and FLT_MIN constants
#include <iostream>

#include "sq.h"

/**
 * @brief Scalar quantization using min-max normalization
 * 
 * Quantizes a floating-point vector to integer representation using uniform
 * quantization based on the min-max range. This is an asymmetric quantization
 * scheme that maps the value range [min, max] to [0, 2^bits - 1].
 * 
 * Quantization formula: q = round((x - min) / scale)
 * where scale = (max - min) / (2^bits - 1)
 * 
 * @param vector Input floating-point vector to quantize
 * @param quantized_vector Output quantized vector (8-bit unsigned integers)
 * @param dim Dimension of the input vector
 * @param bits Number of quantization bits (typically 8)
 * @param min_val Output parameter for minimum value found in vector
 * @param max_val Output parameter for maximum value found in vector
 * @param scale Output parameter for computed quantization scale factor
 */
void scalar_quantize(float *vector, uint8_t *quantized_vector, int dim, int bits, float *min_val, float *max_val, float *scale) {
    // Find the minimum and maximum values in the vector
    *max_val = -FLT_MAX;
    *min_val = FLT_MAX;
    for (int i = 0; i < dim; i++) {
        if (vector[i] < *min_val) *min_val = vector[i];
        if (vector[i] > *max_val) *max_val = vector[i];
    }

    // Compute quantization scale factor
    *scale = (*max_val - *min_val) / ((1 << bits) - 1);

    // Quantize each vector element with rounding
    for (int i = 0; i < dim; i++) {
        float normalized_value = (vector[i] - *min_val) / *scale;
        quantized_vector[i] = (uint8_t)(normalized_value + 0.5f); // Round to nearest integer
    }
}

/**
 * @brief Dequantize a vector back to floating-point representation
 * 
 * Converts quantized integer values back to floating-point using the
 * inverse quantization formula: x = q * scale + min_val
 * 
 * @param quantized_vector Input quantized vector (8-bit unsigned integers)
 * @param dequantized_vector Output floating-point vector
 * @param dim Dimension of the vectors
 * @param min_val Minimum value used during quantization
 * @param scale Scale factor used during quantization
 */
void scalar_dequantize(uint8_t *quantized_vector, float *dequantized_vector, int dim, float min_val, float scale) {
    for (int i = 0; i < dim; i++) {
        // Apply inverse quantization formula
        dequantized_vector[i] = (float)quantized_vector[i] * scale + min_val;
    }
}

/**
 * @brief Generate precomputed scalar quantization information for dataset
 * 
 * Precomputes quantization parameters and derived values for an entire dataset
 * to enable fast distance compensation during search. This function processes
 * all vectors in parallel and stores essential information for efficient
 * similarity computation.
 * 
 * For each vector, computes:
 * - Quantized representation (8-bit values)
 * - Squared L2 norm (||x||²)
 * - Sum of quantized values
 * - Partial statistics for hybrid quantization (if cut > 0)
 * 
 * @param data Input dataset (vectors stored consecutively)
 * @param dim Dimension of each vector
 * @param bits Number of quantization bits (1-8 supported)
 * @param size Number of vectors in the dataset
 * @param p_info Output array of sq_info structures
 * @param cut Dimension cut point for partial quantization (0 = full quantization)
 */
void gen_sq_info(float *data, int dim, int bits, int size, sq_info *p_info, int cut) {
    float *x;

    // Validate bit width
    if(bits < 1 || bits > 8) {
        std::cout << "Error: Bit width must be between 1 and 8!" << std::endl;
        return;
    }

    // Process all vectors in parallel
    #pragma omp parallel for
    for(int i=0; i<size; i++) {
        x = data + i*dim; // Pointer to i-th vector

        // Initialize precomputed values
        p_info[i].xx = 0;           // ||x||²
        p_info[i].xx_part[0] = 0;   // Partial squared norm (first part)
        p_info[i].xx_part[1] = 0;   // Partial squared norm (second part)
        
        p_info[i].sum_quant_x_part[0] = 0; // Sum of quantized values (first part)
        p_info[i].sum_quant_x_part[1] = 0; // Sum of quantized values (second part)

        // Compute squared L2 norm and partial norms
        for(int d=0; d<dim; d++) {
            p_info[i].xx += x[d]*x[d];
            
            // Store partial norm at cut point for hybrid approaches
            if(cut > 0 && cut < dim && d == cut-1) 
                p_info[i].xx_part[0] = p_info[i].xx;
        }
        p_info[i].xx_part[1] = p_info[i].xx - p_info[i].xx_part[0];

        // Quantize the vector and store quantization parameters
        scalar_quantize(x, p_info[i].quant_x_uint8, dim, bits, &p_info[i].min, &p_info[i].max, &p_info[i].scale);

        // Compute sum of quantized values and partial sums
        p_info[i].sum_quant_x = 0;
        for(int d=0; d<dim; d++) {
            p_info[i].sum_quant_x += p_info[i].quant_x_uint8[d];

            // Store partial sum at cut point
            if(cut > 0 && cut < dim && d == cut-1) 
                p_info[i].sum_quant_x_part[0] = p_info[i].sum_quant_x;
        }
    
        // Compute second part of partial sum
        if(cut > 0 && cut < dim)
            p_info[i].sum_quant_x_part[1] = p_info[i].sum_quant_x - p_info[i].sum_quant_x_part[0];
    }
}
