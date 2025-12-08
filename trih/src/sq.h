/**
 * @file sq.h
 * @brief Scalar quantization support for memory-efficient approximate search
 * 
 * This header provides scalar quantization functionality to reduce memory footprint
 * during the reranking phase of TriH-ANNS. Scalar quantization converts floating-point
 * vectors to 8-bit integers while preserving relative distances for efficient
 * approximate distance computation.
 * 
 * Key features:
 * - Per-vector adaptive quantization parameters
 * - Support for two-stage processing with dimension cutoff
 * - Precomputed statistics for fast distance computation
 * - 64-byte aligned memory allocation for SIMD efficiency
 * 
 * The quantization scheme uses linear mapping: q = (x - min) / scale
 * where scale = (max - min) / (2^bits - 1)
 */

#ifndef __SQ__
#define __SQ__

#define DIM 960  ///< Default dimensionality for GIST dataset

/**
 * @brief Dimension cutoff point for two-stage reranking
 * 
 * When set to -1, disables two-stage processing and uses full dimensionality.
 * When set to positive value, splits processing into primary and remaining dimensions.
 */
#define CUT -1 

#include <stdint.h> 

/**
 * @brief Scalar quantization information for a single vector
 * 
 * Contains all precomputed statistics needed for efficient quantized distance
 * computation without requiring dequantization. This enables fast approximate
 * distance calculation in the reranking phase.
 */
struct sq_info {
    // Single-stage quantization parameters
    float xx;                    ///< Original vector squared norm (||x||²)
    float min;                   ///< Minimum value in vector (for quantization)
    float max;                   ///< Maximum value in vector (for quantization)
    float scale;                 ///< Quantization scale factor
    uint32_t sum_quant_x;        ///< Sum of quantized values
    uint32_t sum_quant_xx;       ///< Sum of squared quantized values

    // Two-stage quantization parameters (when CUT >= 0)
    float xx_part[2];            ///< Squared norms for each part [0:CUT), [CUT:DIM)
    uint32_t sum_quant_x_part[2]; ///< Sum of quantized values for each part

    uint8_t *quant_x_uint8;      ///< Quantized vector data (8-bit per dimension)

    /**
     * @brief Constructor - allocates aligned memory for quantized vector
     * 
     * @param dim Vector dimensionality
     * 
     * @note Uses 64-byte aligned allocation for optimal SIMD performance
     */
    sq_info(int dim) {
        xx = 0;
        sum_quant_x = sum_quant_xx = 0;

        // Allocate aligned memory for efficient SIMD operations
        quant_x_uint8 = static_cast<uint8_t*>(aligned_alloc(64, dim));
    }

    /**
     * @brief Destructor - frees allocated memory
     */
    ~sq_info() {
        if(quant_x_uint8) free(quant_x_uint8);
    }
};

/**
 * @brief Scalar quantization of floating-point vector
 * 
 * Converts floating-point vector to 8-bit quantized representation using
 * linear quantization with computed min/max range.
 * 
 * @param vector Input floating-point vector
 * @param quantized_vector Output quantized vector (8-bit per element)
 * @param dim Vector dimensionality
 * @param bits Quantization bit width (typically 8)
 * @param min_val Output: minimum value found in vector
 * @param max_val Output: maximum value found in vector  
 * @param scale Output: computed quantization scale factor
 * 
 * @note Quantization formula: q = round((x - min) / scale)
 * @note Scale = (max - min) / (2^bits - 1)
 */
void scalar_quantize(float *vector, uint8_t *quantized_vector, int dim, int bits, float *min_val, float *max_val, float *scale);

/**
 * @brief Scalar dequantization back to floating-point
 * 
 * Converts quantized vector back to floating-point representation.
 * Primarily used for verification and debugging purposes.
 * 
 * @param quantized_vector Input quantized vector
 * @param dequantized_vector Output floating-point vector
 * @param dim Vector dimensionality
 * @param min_val Minimum value used in quantization
 * @param scale Scale factor used in quantization
 * 
 * @note Dequantization formula: x = q * scale + min
 */
void scalar_dequantize(uint8_t *quantized_vector, float *dequantized_vector, int dim, float min_val, float scale);

/**
 * @brief Generate complete scalar quantization information for dataset
 * 
 * Processes entire dataset to generate quantization parameters and precomputed
 * statistics for all vectors. This enables efficient quantized distance
 * computation during search.
 * 
 * @param data Input dataset vectors (size x dim)
 * @param dim Vector dimensionality
 * @param bits Quantization bit width
 * @param size Number of vectors in dataset
 * @param p_info Output array of quantization info structures
 * @param cut Dimension cutoff for two-stage processing (-1 = disabled)
 * 
 * @note Precomputes all statistics needed for fast distance calculation
 * @note Memory for p_info array must be pre-allocated by caller
 */
void gen_sq_info(float *data, int dim, int bits, int size, sq_info *p_info, int cut);

#endif