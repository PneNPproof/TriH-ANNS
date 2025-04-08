#ifndef __SQ__
#define __SQ__


#define DIM 960

#define CUT -1 //re_rank 2阶段处理维度的cut点, -1 表示不用2阶段


#include <stdint.h> 

struct sq_info {//单个向量的信息
    float xx;
    float min;
    float max;
    float scale;
    uint32_t sum_quant_x;
    uint32_t sum_quant_xx;

    float xx_part[2];
    uint32_t sum_quant_x_part[2];

    uint8_t *quant_x_uint8;

    sq_info(int dim) {
        xx = 0;
        sum_quant_x = sum_quant_xx = 0;

        quant_x_uint8 = static_cast<uint8_t*>(aligned_alloc(64, dim)); //new uint8_t[DIM];
    }

    ~sq_info() {
        if(quant_x_uint8) free(quant_x_uint8);
    }
};




// 标量量化函数
void scalar_quantize(float *vector, uint8_t *quantized_vector, int dim, int bits, float *min_val, float *max_val, float *scale);

// 反量化函数
void scalar_dequantize(uint8_t *quantized_vector, float *dequantized_vector, int dim, float min_val, float scale);


void gen_sq_info(float *data, int dim, int bits, int size, sq_info *p_info, int cut);
#endif