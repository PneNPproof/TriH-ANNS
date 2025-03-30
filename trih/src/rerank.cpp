#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>

#include "distance.h"
#include "rerank.h"
#include "pca.h"
#include "sq.h"
#include "distance.h"

// Scalar version of inner_product function without SIMD instructions
float inner_product_scalar(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    int d
) 
{
    float sum = 0.0f;
    
    // Simple loop to calculate dot product
    for (int i = 0; i < d; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

float inner_product(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    int d
) 
{
    constexpr int vec_size = 16;
    const float* a_ptr = a;
    const float* b_ptr = b;
    
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    // 主循环处理4*16=64个元素/迭代
    int main_loop_count = d / (vec_size * 4);
    while (main_loop_count--) {
        __m512 a0 = _mm512_load_ps(a_ptr);
        __m512 a1 = _mm512_load_ps(a_ptr + vec_size);
        __m512 a2 = _mm512_load_ps(a_ptr + vec_size*2);
        __m512 a3 = _mm512_load_ps(a_ptr + vec_size*3);
        
        __m512 b0 = _mm512_load_ps(b_ptr);
        __m512 b1 = _mm512_load_ps(b_ptr + vec_size);
        __m512 b2 = _mm512_load_ps(b_ptr + vec_size*2);
        __m512 b3 = _mm512_load_ps(b_ptr + vec_size*3);
        
        sum0 = _mm512_fmadd_ps(a0, b0, sum0);
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);
        sum2 = _mm512_fmadd_ps(a2, b2, sum2);
        sum3 = _mm512_fmadd_ps(a3, b3, sum3);
        
        a_ptr += vec_size*4;
        b_ptr += vec_size*4;
    }

    // 合并累加器
    sum0 = _mm512_add_ps(sum0, _mm512_add_ps(sum1, sum2));
    sum0 = _mm512_add_ps(sum0, sum3);

    // 处理剩余16的倍数部分
    int remaining = d % (vec_size * 4);
    while (remaining >= vec_size) {
        __m512 a_vec = _mm512_load_ps(a_ptr);
        __m512 b_vec = _mm512_load_ps(b_ptr);
        sum0 = _mm512_fmadd_ps(a_vec, b_vec, sum0);
        a_ptr += vec_size;
        b_ptr += vec_size;
        remaining -= vec_size;
    }

    // 处理尾部元素
    if (remaining > 0) {
        __mmask16 mask = (1U << remaining) - 1;
        __m512 a_vec = _mm512_maskz_loadu_ps(mask, a_ptr);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, b_ptr);
        sum0 = _mm512_fmadd_ps(a_vec, b_vec, sum0);
    }

    return _mm512_reduce_add_ps(sum0);
}


// 最大堆调整（基于结构体内部缓存的距离值）
template<typename T>
static void max_heapify(HeapElement<T>* heap, int size, int pos) {
    int largest = pos;
    int left = 2 * pos + 1;
    int right = 2 * pos + 2;

    if (left < size && heap[left].distance > heap[largest].distance)
        largest = left;
    if (right < size && heap[right].distance > heap[largest].distance)
        largest = right;
    if (largest != pos) {
        HeapElement<T> temp = heap[pos];
        heap[pos] = heap[largest];
        heap[largest] = temp;
        max_heapify(heap, size, largest);
    }
}

// 构建最大堆
template<typename T>
static void build_max_heap(HeapElement<T>* heap, int size) {
    for (int i = size / 2 - 1; i >= 0; i--)
        max_heapify(heap, size, i);
}

// 插入排序（针对小数据量优化）
template<typename T>
static void insertion_sort(HeapElement<T>* arr, int size) {
    for (int i = 1; i < size; ++i) {
        HeapElement<T> key = arr[i];
        int j = i - 1;
        
        while (j >= 0 && arr[j].distance > key.distance) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/**
 * @brief Selects the IDs corresponding to the k smallest distances from distance array
 * 
 * This function finds the k smallest elements in the distance array and returns
 * their corresponding IDs from the id array. It uses a max-heap algorithm for 
 * efficient selection, with optimized sorting for different data sizes.
 * 
 * @tparam T           Data type for distance values (typically float or double)
 * @param distance     Input array of distance values
 * @param id           Input array of corresponding IDs
 * @param n            Size of the input arrays
 * @param k            Number of smallest elements to find (k <= n)
 * @param result       Output array to store the k IDs corresponding to smallest distances
 * @param result_distance Optional output array to store the k smallest distances (can be NULL)
 * 
 * @note The function returns nothing if k <= 0, n <= 0, k > n, or if distance/id are NULL
 * @note Memory for result and result_distance must be pre-allocated by the caller
 * @note The output will be sorted in ascending order by distance
 */
template<typename T>
void get_min_k_ids(const T* distance, const int* id, int n, int k, int *result, T* result_distance) {
    /* 参数校验和边界处理 */
    if (k <= 0 || n <= 0 || k > n || !distance || !id)
        return;
    
    /* 初始化堆结构 */
    HeapElement<T>* heap = (HeapElement<T>*)malloc(k * sizeof(HeapElement<T>));
    if (!heap) return;

    // 初始化堆元素并缓存距离值
    for (int i = 0; i < k; i++) {
        heap[i].index = i;
        heap[i].distance = distance[i];
    }
    build_max_heap(heap, k);

    /* 处理后续元素 */
    for (int i = k; i < n; i++) {
        if (distance[i] < heap[0].distance) {
            // 替换堆顶并立即缓存新距离值
            heap[0].index = i;
            heap[0].distance = distance[i];
            max_heapify(heap, k, 0);
        }
    }

    /* 排序结果（根据数据量选择排序算法） */
    if (k <= 64) { // 经验阈值，可根据实际测试调整
        insertion_sort(heap, k);
    } else {
        qsort(heap, k, sizeof(HeapElement<T>), 
            [](const void* a, const void* b) {
                const HeapElement<T>* ha = (const HeapElement<T>*)a;
                const HeapElement<T>* hb = (const HeapElement<T>*)b;
                return (ha->distance > hb->distance) - (ha->distance < hb->distance);
            });
    }

    /* 转换最终结果 */
    for (int i = 0; i < k; i++) {
        result[i] = id[heap[i].index];
        if(result_distance != NULL)
            result_distance[i] = heap[i].distance;
    }
    
    free(heap);
}

/**
 * @brief Performs precise reranking on a filtered candidate set of vectors
 *
 * This function takes the results from a first-phase approximate search (phase1_topk_ids)
 * and computes precise L2 distances between the query and these candidates. It then 
 * selects the phase2_topk closest points for the final results.
 *
 * The distance calculation uses the formula: ||x-y||² = ||x||² + ||y||² - 2<x,y>
 * where ||y||² values are precomputed in dataset_squared_norms and <x,y> is calculated
 * using SIMD-optimized inner product.
 *
 * @param distances_buffer Buffer to store calculated distances (must be at least phase1_topk in size)
 * @param dataset Full dataset containing all vectors (dim-dimensional vectors)
 * @param dataset_squared_norms Precomputed squared L2 norms for each vector in dataset
 * @param query Query vector for which to find nearest neighbors
 * @param phase1_topk_ids Array of IDs selected from first-phase search (input)
 * @param phase1_topk Number of candidates from first-phase search
 * @param data_num Total number of vectors in the dataset
 * @param dim Dimensionality of the vectors
 * @param phase2_topk Number of closest vectors to select (final result size)
 * @param phase2_topk_ids Output array to store the selected vector IDs (must be at least phase2_topk in size)
 *
 * @return 0 on success
 * 
 * @note The phase2_topk must be less than or equal to phase1_topk
 * @note The function assumes all input arrays are properly allocated
 * @note Results are sorted in ascending order by distance
 */
int re_rank2(
  float* dataset, 
  float* dataset_squared_norms, 
  float *query, 
  int *phase1_topk_ids,
  int phase1_topk, 
  int data_num, 
  int dim, 
  int phase2_topk, 
  int *phase2_topk_ids
) 
{
  float distances_buffer[1024];
  
  for (int i=0; i<phase1_topk; i++) {
    int id = phase1_topk_ids[i];
    // distances_buffer[i] = dataset_squared_norms[id] - 2 * inner_product(query, dataset + id * dim, dim);
    distances_buffer[i] = euclideanDistance_avx512(query, dataset + id * dim, dim);
    // distances_buffer[i] = dataset_squared_norms[id] - 2 * inner_product_scalar(query, dataset + id * dim, dim);
  }

  //重新排序，获取最终结果
  get_min_k_ids(distances_buffer, phase1_topk_ids, phase1_topk, phase2_topk, phase2_topk_ids, (float*)NULL);

  return 0;
}


float distance_compensation_sq_precomputing_avx(uint8_t *qq, float min_q, uint32_t sum_qq, float scale_q, int dim, sq_info &sq) {
    float dist = sq.xx;

    // uint32_t tmp = dot_product(qq, sq.quant_x_uint8, dim);
    // uint32_t tmp = dot_product_uint8(qq, sq.quant_x_uint8, dim);//王哲
    uint32_t tmp = dot_product_uint8_avx512(qq, sq.quant_x_uint8, dim);
    
    dist -= 2*(sq.scale * scale_q * tmp + scale_q * sq.min * sum_qq + sq.scale * min_q * sq.sum_quant_x + min_q * sq.min * dim);
    
    return dist;
}

int re_rank(
    float *query,
    pca_index &index,
    sq_info *p_sq_info,
    float *distances,
    int *ids,
    int size,
    int final_topk,
    int *final_topk_ids
){

    float *q_proj_remain = static_cast<float*>(aligned_alloc(64, sizeof(float)*(index.dim - index.column_num)));
    queryProjectRest2(query, index, q_proj_remain);

    //针对query remain 进行量化
    float min_q, max_q, scale_q;
    uint8_t *qq = static_cast<uint8_t*>(aligned_alloc(64, index.dim-index.column_num));
    // scalar_quantize(q_proj_remain, qq, index.dim-index.column_num, 8, &min_q, &max_q, &scale_q);
    uint32_t sum_qq = 0;
    // for(int i=0; i<index.dim-index.column_num; i++) sum_qq += qq[i];

    //距离补偿
    // #pragma omp parallel for
    for(int i=0; i<size; i++) {
        distances[i] += distance_compensation_sq_precomputing_avx(qq, min_q, sum_qq, scale_q, index.dim-index.column_num, p_sq_info[ids[i]]);
    }

    // free(qq); 

    //重新排序，获取最终结果
    get_min_k_ids(distances, ids, size, final_topk, final_topk_ids, (float*)NULL);

    // free(q_proj_remain);
    // delete [] distances;
    // delete [] ids;

    return 0;
}