#include <stdio.h>
#include <stdlib.h>
#include <float.h>  // 用于 FLT_MAX 和 FLT_MIN

#include <iostream>

#include "sq.h"

// 所有数据量化到 0 - 255 内，min_val 限定为 0

// 标量量化函数
//根据最小值最大值进行SQ
void scalar_quantize(float *vector, uint8_t *quantized_vector, int dim, int bits, float *min_val, float *max_val, float *scale) {
    // 找到向量中的最小值和最大值
    *max_val = -FLT_MAX;
    *min_val = FLT_MAX;
    for (int i = 0; i < dim; i++) {
        if (vector[i] < *min_val) *min_val = vector[i];
        if (vector[i] > *max_val) *max_val = vector[i];
    }

    // 计算量化的比例因子
    *scale = (*max_val - *min_val) / ((1 << bits) - 1);

    // 对向量进行量化
    for (int i = 0; i < dim; i++) {
        float normalized_value = (vector[i] - *min_val) / *scale;
        quantized_vector[i] = (uint8_t)(normalized_value + 0.5f); // 四舍五入
    }
}

// 反量化函数
void scalar_dequantize(uint8_t *quantized_vector, float *dequantized_vector, int dim, float min_val, float scale) {
    for (int i = 0; i < dim; i++) {
        dequantized_vector[i] = (float)quantized_vector[i] * scale + min_val;

        // dequantized_vector[i] = (float)quantized_vector[i] * scale;
    }
}



//用于构建SQ量化的预计算
void gen_sq_info(float *data, int dim, int bits, int size, sq_info *p_info, int cut) {
    float *x;

    if(bits < 1 || bits > 8) {
        std::cout << "Wrong bits!" << std::endl;
        return;
    }

    for(int i=0; i<size; i++) {

        x = data + i*dim;

        p_info[i].xx = 0;
        p_info[i].xx_part[0] = 0;
        p_info[i].xx_part[1] = 0;

        p_info[i].sum_quant_x_part[0] = 0;
        p_info[i].sum_quant_x_part[1] = 0;

        for(int d=0; d<dim; d++) {
            p_info[i].xx += x[d]*x[d]; //获取 sum
            
            if(cut > 0 && cut < dim && d == cut-1) 
                p_info[i].xx_part[0] = p_info[i].xx;
        }
        p_info[i].xx_part[1] = p_info[i].xx - p_info[i].xx_part[0];

        scalar_quantize(x, p_info[i].quant_x_uint8, dim, bits, &p_info[i].min, &p_info[i].max, &p_info[i].scale);

        p_info[i].sum_quant_x = 0;
        for(int d=0; d<dim; d++) {
            p_info[i].sum_quant_x += p_info[i].quant_x_uint8[d];

            if(cut > 0 && cut < dim && d == cut-1) 
                p_info[i].sum_quant_x_part[0] = p_info[i].sum_quant_x;
        }
    
        if(cut > 0 && cut < dim)
            p_info[i].sum_quant_x_part[1] = p_info[i].sum_quant_x - p_info[i].sum_quant_x_part[0];
    }
}
