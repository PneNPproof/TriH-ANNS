#include <iostream>
#include <omp.h>
#include <float.h>
#include <cstring>

#include "gist.h"
#include "pca.h"
#include "search.h"

//计算2点之间的l2距离
float euclideanDistance(const float a[], const float b[], int dimension) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += (a[i] - b[i])*(a[i] - b[i]);
    }

    return sum;
}


//计算query到所有点的距离, size 为 base中点的数量
void computeDistances(const float *query, const float *base, int dimension, int size, float distances[]) {

#pragma omp parallel for
    for(int i=0; i<size; i++) {
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

//从求出的distances中，分sectionNum个段，每个段选择最小距离，之后再从所有最小距离中，选择topk个最小者，返回他们在distances的下标
//size 为 distances 数组大小； sectionNum 为分段的每段大小；topk 为求取的最终最近距离的个数； ids 为最近距离对应的distances下标
void computeTopkDistances(const float *distances, int size, int sectionNum, int topk, int *ids) {
    float min_dist;
    int min_id = -1;

    float min_distances[sectionNum];
    int min_ids[sectionNum];

    //每个section求 top1 距离
    int unit = size/sectionNum; //每个section的数量
    for(int i=0; i<sectionNum; i++) {
        //求每个section内的最小值
        min_dist = FLT_MAX;
        int start = i*unit;
        int end = start + unit;
        if(i == sectionNum - 1)
            end += size % unit; //加上最后剩余的

        for(int k=start; k<end; k++){
            if(min_dist > distances[k]) {
                min_dist = distances[k];
                min_id = k;
            }
        }

        min_distances[i] = min_dist;
        min_ids[i] = min_id;
    }

    //min_distances中获取topk
    for(int i=0; i<topk; i++) {
        min_dist = FLT_MAX;
        for(int k=0; k<sectionNum; k++) {
            if(min_dist > min_distances[k]) {
                min_dist = min_distances[k];
                min_id = k; //记录第k个sectionNum的top1是最小的
            }
        }

        ids[i] = min_ids[min_id]; //第min_id个min_ids，为全局的下标

        min_distances[min_id] = FLT_MAX;//标记为最大
    }
}

//在dim范围内计算query与x的近视距离，dim小于query/x的维度
float distance_compensation(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    //1. 可以预计算
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;
    
    float max = FLT_MIN, min = FLT_MAX;
    for(int i=0; i<dim; i++) {
        if(x[i] > max) max = x[i];
        if(x[i] < min) min = x[i];
    }

    float scale = (max - min) / sq_intervals;

    //非对称量化，应该称为近视量化
    float zero_point = min;

    // uint8_t sq[dim];
    // quant_point(x, sq, scale, zero_point, dim, sq_intervals);

    float sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((x[i] - zero_point)/scale) * scale; 
    }

    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*query[i] * sq[i]; //*sq[i]
    }

    dist -= tmp;

    return dist;

}


//在dim范围内计算query与x的近视距离，dim小于query/x的维度
float distance_compensation2(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    //1. 可以预计算
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;

    //2. query的维度内的内积，由于距离补偿这部分相同，不用计算

    //3. -2 * x_i * q_i 部分，对 x_i 进行SQ
    // SQ 过程可以预计算

    //将数据转换成为 int
    int xx[dim], qq[dim];
    for(int i=0; i<dim; i++) {
        xx[i] = x[i] * 100000;
        qq[i] = query[i] * 100000;
    }

    //SQ
    int max = -999999, min = 999999;
    for(int i=0; i<dim; i++) {
        if(xx[i] > max) max = xx[i];
        if(xx[i] < min) min = xx[i];
    }

    int unit = (max - min) / sq_intervals;

    //非对称量化，应该称为近视量化
    int zero_point = min;

    int sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((xx[i] - zero_point)/unit) * unit + unit/2; 
    }

    //x[i] 被限定到 (max-min)/unit 个值，可以将 dim 个乘法 替换成 先加 后乘
    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*qq[i] * sq[i];
    }

    dist -= tmp/100000/100000;
    
    return dist;

}

//在dim范围内计算query与x的近视距离，dim小于query/x的维度
float distance_compensation3(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    //1. 可以预计算
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;

    //2. query的维度内的内积，由于距离补偿这部分相同，不用计算

    //3. -2 * x_i * q_i 部分，对 x_i 进行SQ
    // SQ 过程可以预计算
    
    //将数据转换成为 int
    long long xx[dim], qq[dim];
    for(int i=0; i<dim; i++) {
        xx[i] = x[i] * 1e7;
        qq[i] = query[i] * 1e7;
    }

    //SQ
    long long max = -99999999, min = 99999999;
    for(int i=0; i<dim; i++) {
        if(xx[i] > max) max = xx[i];
        if(xx[i] < min) min = xx[i];
    }

    long long unit = (max - min) / sq_intervals;
    if(unit == 0) unit = 1;

    //非对称量化，应该称为近视量化
    long long zero_point = min;

    long long sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((xx[i] - zero_point)/unit) * unit + unit/2; 
    }

    //x[i] 被限定到 (max-min)/unit 个值，可以将 dim 个乘法 替换成 先加 后乘
    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*qq[i] * sq[i]; //*sq[i]
    }

    dist -= tmp/1e14;

    // 原始计算
    // dist = 0;
    // for(int i=0; i<dim; i++) {
    //     dist += (x[i]-query[i])*(x[i]-query[i]);
    // }

    return dist;

}


float distance_compensation3_1(float *query, float *x, int dim, int sq_intervals) {
    float dist = 0;
    
    //1. 可以预计算
    for(int i=0; i<dim; i++)
        dist += x[i] * x[i];

    if(sq_intervals == 0)
        return dist;
    
    float max = FLT_MIN, min = FLT_MAX;
    for(int i=0; i<dim; i++) {
        if(x[i] > max) max = x[i];
        if(x[i] < min) min = x[i];
    }

    float scale = (max - min) / sq_intervals;

    //非对称量化，应该称为近视量化
    float zero_point = min;

    //1. 方法1:直接获取sq，进行计算
    float sq[dim];
    for(int i=0; i<dim; i++) {
        sq[i] = zero_point + (int)((x[i] - zero_point)/scale+0.5) * scale; 
    }

    double tmp = 0;
    for(int i=0; i<dim; i++) {
        tmp += 2*query[i] * sq[i]; //*sq[i]
    }

    dist -= tmp;


    //2. 合并同类项的方式计算
    // int sq_ind[960]={0};
    // for(int i=0; i<dim; i++) {
    //     sq_ind[i] = (int)((x[i] - zero_point)/scale + 0.5); 
    // }

    // for(int i=0; i<dim; i++) {
        
    //     float cum_query = 0;
    //     if(sq_ind[i] == -1) continue;

    //     int cur_scales = sq_ind[i];
    //     sq_ind[i] = -1;
        
    //     cum_query += query[i];

    //     for(int k=i+1; k<dim; k++) {
    //         if(cur_scales == sq_ind[k]) {
    //             sq_ind[k] = -1;

    //             cum_query += query[k];
    //         }
    //     }

    //     dist -= 2*cum_query*(zero_point + cur_scales * scale);
    // }

    return dist;
}


//quant_base，dim 都是主维度之后需要补偿的维度数量，query，scale，zp也是主维度之后的
float distance_compensation4(uint8_t *quant_base, int idx, float *query, int dim, float scale, float zp, int b) {
    float dist = 0;

    uint8_t *quant_query = new uint8_t[dim];
    quant_point(query, quant_query, scale, zp, dim, b);

    uint8_t *p = quant_base+idx*dim;
    for (int i = 0; i < dim; i++) {
        dist += (p[i] - quant_query[i]) * (p[i] - quant_query[i]);
    }

    // return dist; 
    return dist*scale*scale; //距离补偿需要dequant

    // float point[dim];
    // dequant_point(quant_base+idx*dim, point, scale, zp, dim);
    // for(int i=0; i<dim; i++)
    //     dist += point[i]*point[i];
    
    // auto quant_point = quant_base + idx * dim;
    // float inner_prod = 0;
    // for(int i=0; i<dim; i++)
    //     inner_prod += 2 * quant_point[i] * quant_query[i];
    
    // dist -= inner_prod * scale * scale;

    // return dist;

    // 原始计算
    // dist = 0;
    // for(int i=0; i<dim; i++) {
    //     dist += (x[i]-query[i])*(x[i]-query[i]);
    // }

}


//quant_base，dim 都是主维度之后需要补偿的维度数量，query，scale，zp也是主维度之后的
float distance_compensation5(float *base, int idx, float *query, int dim, float scale, float zp, int b) {
    float dist = 0;

    // 原始计算
    float *p = base+idx*dim;
    for(int i=0; i<dim; i++) {
        dist += (p[i]-query[i])*(p[i]-query[i]);
    }

    return dist;
}

//搜索 PCA index，不涉及re-rank
//在p_index的所有index（index_num个index）中，进行搜索，每个index 计算距离，所有距离汇总
//汇总距离分成 secrtion_num段，每段取 top1，后排序取 topk
void search_index(float *query, float *base, pca_index &index, int section_num, int topk, int *ids, int final_topk, int *final_topk_ids0, int *final_topk_ids, int *final_topk_ids_sq, int *final_topk_ids_shuffle) {

    float *distances = new float[index.record_num];

    //求距离，并放入 distances，从而得到query到所有点的PCA距离
    
    //1. 求query在当前index的投影
    float q_proj[index.column_num], q_proj_remain[index.dim - index.column_num], q_proj2[index.column_num2], q_proj2_shuffle[index.column_num2];
    queryProject(query, index, q_proj, q_proj_remain, q_proj2, q_proj2_shuffle);

    //2. 求query投影PCA1到index内所有点的距离，保存到dist数组
    computeDistances(q_proj, index.trans_data, index.column_num, index.record_num, distances);

    //针对distances，获取topk，如300，的ids，注意：ids内的id已经是原始数据的id
    computeTopkDistances(distances, index.record_num, section_num, topk, ids);

    float *distances2 = new float[index.record_num];
    memcpy(distances2, distances, sizeof(float)*index.record_num);

    float *distances3 = new float[index.record_num];
    memcpy(distances3, distances, sizeof(float)*index.record_num);

    //re-rank with pca2
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        distances[ids[i]] += euclideanDistance(q_proj2, index.trans_data2+ids[i]*index.column_num2, index.column_num2);
    }

    // std::cout << "*******" << std::endl;
    // for(int i=0; i<topk; i++) {
    //     float tmp = euclideanDistance(q_proj2, index.trans_data2+ids[i]*index.column_num2, index.column_num2);
    //     std::cout << tmp << "/" << ids[i] << " ";
        
    // }
    // std::cout << std::endl << std::endl;

    float min_dist;
    int min_id = -1;
    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances[ids[k]]) {
                min_dist = distances[ids[k]];
                min_id = k;
            }
        }

        final_topk_ids[i] = ids[min_id];

        distances[ids[min_id]] = FLT_MAX;
    }

    delete [] distances;


    //re-rank with pca2_shuffle
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        int ind = 0;
        for(; ind<index.record_num; ind++) {
            if(index.ids_shuffle[ind] == ids[i])
                break;
        }
        distances2[ids[i]] += euclideanDistance(q_proj2_shuffle, index.trans_data2_shuffle + ind * index.column_num2, index.column_num2);
    }


    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances2[ids[k]]) {
                min_dist = distances2[ids[k]];
                min_id = k;
            }
        }

        final_topk_ids_shuffle[i] = ids[min_id];

        distances2[ids[min_id]] = FLT_MAX;
    }

    delete [] distances2;

    //re-rank with sq compensation
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        distances3[ids[i]] += distance_compensation3(q_proj2, index.trans_data2+ids[i]*index.column_num2, 894, 64); //不能用全部的剩余维度 960-64=896，否则recall反而降低，溢出相关？128 - 832就可以，64 - 896 不行。
    }

    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances3[ids[k]]) {
                min_dist = distances3[ids[k]];
                min_id = k;
            }
        }

        final_topk_ids_sq[i] = ids[min_id];

        distances3[ids[min_id]] = FLT_MAX;
    }

    delete [] distances3;


    //re-rank 0: 用原始向量计算精确距离
    float distances0[topk];
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        distances0[i] = euclideanDistance(query, base+ids[i]*index.dim, index.dim);
    }

    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances0[k]) {
                min_dist = distances0[k];
                min_id = k;
            }
        }

        final_topk_ids0[i] = ids[min_id];

        distances0[min_id] = FLT_MAX;
    }
}

//增加了全向量单一维度的sq
void search_index(float *query, float *base, pca_index &index, int section_num, int topk, int *ids, int final_topk, int *final_topk_ids0, int *final_topk_ids, int *final_topk_ids_sq, int *final_topk_ids_shuffle,
    uint8_t *quant_base, float scale, float zp, int b, int *final_topk_ids_sq_all_vec) {
    
    float *distances = new float[index.record_num];

    //求距离，并放入 distances，从而得到query到所有点的PCA距离

    //1. 求query在当前index的投影
    float q_proj[index.column_num], q_proj_remain[index.dim - index.column_num], q_proj2[index.column_num2], q_proj2_shuffle[index.column_num2];
    queryProject(query, index, q_proj, q_proj_remain, q_proj2, q_proj2_shuffle);

    //2. 求query投影PCA1到index内所有点的距离，保存到dist数组
    computeDistances(q_proj, index.trans_data, index.column_num, index.record_num, distances);

    //针对distances，获取topk，如300，的ids，注意：ids内的id已经是原始数据的id
    computeTopkDistances(distances, index.record_num, section_num, topk, ids);

    float *distances2 = new float[index.record_num];
    memcpy(distances2, distances, sizeof(float)*index.record_num);

    float *distances3 = new float[index.record_num];
    memcpy(distances3, distances, sizeof(float)*index.record_num);

    float *distances4 = new float[index.record_num];
    memcpy(distances4, distances, sizeof(float)*index.record_num);

    //re-rank with pca2
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        distances[ids[i]] += euclideanDistance(q_proj2, index.trans_data2+ids[i]*index.column_num2, index.column_num2);
    }

    // std::cout << "*******" << std::endl;
    // for(int i=0; i<topk; i++) {
    //     float tmp = euclideanDistance(q_proj2, index.trans_data2+ids[i]*index.column_num2, index.column_num2);
    //     std::cout << tmp << "/" << ids[i] << " ";
        
    // }
    // std::cout << std::endl << std::endl;

    float min_dist;
    int min_id = -1;
    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances[ids[k]]) {
                min_dist = distances[ids[k]];
                min_id = k;
            }
        }

        final_topk_ids[i] = ids[min_id];

        distances[ids[min_id]] = FLT_MAX;
    }

    delete [] distances;


    //re-rank with pca2_shuffle
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        int ind = 0;
        for(; ind<index.record_num; ind++) {
            if(index.ids_shuffle[ind] == ids[i])
                break;
        }
        distances2[ids[i]] += euclideanDistance(q_proj2_shuffle, index.trans_data2_shuffle + ind * index.column_num2, index.column_num2);
    }


    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances2[ids[k]]) {
                min_dist = distances2[ids[k]];
                min_id = k;
            }
        }

        final_topk_ids_shuffle[i] = ids[min_id];

        distances2[ids[min_id]] = FLT_MAX;
    }

    delete [] distances2;




    //re-rank with sq compensation
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        distances3[ids[i]] += distance_compensation3_1(q_proj2, index.trans_data2+ids[i]*index.column_num2, 896, 64); //不能用全部的剩余维度 960-64=896，否则recall反而降低，溢出相关？128 - 832就可以，64 - 896 不行。
    }

    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances3[ids[k]]) {
                min_dist = distances3[ids[k]];
                min_id = k;
            }
        }

        final_topk_ids_sq[i] = ids[min_id];

        distances3[ids[min_id]] = FLT_MAX;
    }

    delete [] distances3;





    //re-rank with sq compensation (all vectors)
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        distances4[ids[i]] += distance_compensation4(quant_base, ids[i], q_proj_remain, 896, scale, zp, b);//remain维度，recall 64

        // distances4[ids[i]] = distance_compensation4(quant_base, ids[i], query, 960, scale, zp, b); //全部维度，recall 94.5

        // distances4[ids[i]] += distance_compensation5(index.trans_data_remain, ids[i], q_proj_remain, 896, scale, zp, b);
    }

    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances4[ids[k]]) {
                min_dist = distances4[ids[k]];
                min_id = k;
            }
        }

        final_topk_ids_sq_all_vec[i] = ids[min_id];

        distances4[ids[min_id]] = FLT_MAX;
    }

    delete [] distances4;






    //re-rank 0: 用原始向量计算精确距离
    float distances0[topk];
    #pragma omp parallel for
    for(int i=0; i<topk; i++) {
        distances0[i] = euclideanDistance(query, base+ids[i]*index.dim, index.dim);
    }

    for(int i=0; i<final_topk; i++) {//挑选最终的 final_topk，如100 
        min_dist = FLT_MAX;

        for(int k=0; k<topk; k++) {
            if(min_dist > distances0[k]) {
                min_dist = distances0[k];
                min_id = k;
            }
        }

        final_topk_ids0[i] = ids[min_id];

        distances0[min_id] = FLT_MAX;
    }
}

uint8_t clip(float x, float scale, float zp, int start, int end)
{
    float y = x / scale + zp;
    if (y < start)
        return start;
    if (y > end)
        return end;
    return (uint8_t)y;
}


void quant_point(const float *point, uint8_t *quant_point, float scale, float zp, int dimension, int b)
{
    for (int i = 0; i < dimension; i++)
    {
        quant_point[i] = clip(point[i], scale, zp, 0, b - 1);
    }
}

void dequant_point(uint8_t *quant_point, float *point, float scale, float zp, int dimension)
{
    for (int i = 0; i < dimension; i++)
    {
        point[i] = (quant_point[i] - zp) * scale;
    }
}