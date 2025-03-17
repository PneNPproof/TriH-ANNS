#ifndef __SEARCH_H__
#define __SEARCH_H__

#include "pca.h"

float euclideanDistance(const float a[], const float b[], int dimension);
void computeDistances(const float *query, const float *base, int dimension, int size, float distances[]);
void computeTopkDistances(const float *distances, int size, int sectionNum, int topk, int *ids);
void search_index(float *query, float *base, pca_index &index, int section_num, int topk, int *ids, int final_topk, int *final_topk_ids0, int *final_topk_ids, int *final_topk_ids_sq, int *final_topk_ids_shuffle);
void search_index(float *query, float *base, pca_index &index, int section_num, int topk, int *ids, int final_topk, int *final_topk_ids0, int *final_topk_ids, int *final_topk_ids_sq, int *final_topk_ids_shuffle,
    uint8_t *quant_base, float scale, float zp, int b, int *final_topk_ids_sq_all_vec);
uint8_t clip(float x, float scale, float zp, int start, int end);
void quant_point(const float *point, uint8_t *quant_point, float scale, float zp, int dimension, int b);
void dequant_point(uint8_t *quant_point, float *point, float scale, float zp, int dimension);
float dequant_l2(float distance, float scale);

#endif