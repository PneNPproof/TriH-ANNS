#ifndef __DISTANCE__
#define __DISTANCE__

#include <stdint.h>
#include <math.h>
#include <immintrin.h>


//计算2点之间的l2距离
static inline float euclideanDistance(const float a[], const float b[], int dimension) {
    float sum = 0.0;
    float tmp;
    for (int i = 0; i < dimension; ++i) {
        tmp = a[i] - b[i];
        sum += tmp*tmp;
    }

    return sum;
}


//采用avx求L2距离
static inline float euclideanDistance_avx(
    const float* __restrict vec1,  // 向量1 [ALIGN64]
    const float* __restrict vec2,  // 向量2 [ALIGN64]
    int dim                   // 向量维度
) {
    size_t i = 0;
    const size_t simd_width = 8; // AVX处理8个float
    const size_t unroll_factor = 4; // 展开因子

    // 使用4个独立累加器打破依赖链
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    // 主循环：每次处理32个元素（4组x8元素）
    size_t block_size = unroll_factor * simd_width;
    size_t num_blocks = dim / block_size;
    
    for (size_t block = 0; block < num_blocks; ++block) {
        // 预取下个block数据（提前预取256字节）
        _mm_prefetch((const char*)(vec1 + i + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(vec2 + i + 64), _MM_HINT_T0);

        // 加载4组数据
        __m256 v1_0 = _mm256_load_ps(vec1 + i);
        __m256 v2_0 = _mm256_load_ps(vec2 + i);
        __m256 diff0 = _mm256_sub_ps(v1_0, v2_0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 v1_1 = _mm256_load_ps(vec1 + i + 8);
        __m256 v2_1 = _mm256_load_ps(vec2 + i + 8);
        __m256 diff1 = _mm256_sub_ps(v1_1, v2_1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        __m256 v1_2 = _mm256_load_ps(vec1 + i + 16);
        __m256 v2_2 = _mm256_load_ps(vec2 + i + 16);
        __m256 diff2 = _mm256_sub_ps(v1_2, v2_2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

        __m256 v1_3 = _mm256_load_ps(vec1 + i + 24);
        __m256 v2_3 = _mm256_load_ps(vec2 + i + 24);
        __m256 diff3 = _mm256_sub_ps(v1_3, v2_3);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

        i += block_size;
    }

    // 合并累加器
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    __m256 sum = _mm256_add_ps(acc0, acc2);

    // 处理剩余元素（SIMD部分）
    for (; i + simd_width <= dim; i += simd_width) {
        __m256 v1 = _mm256_load_ps(vec1 + i);
        __m256 v2 = _mm256_load_ps(vec2 + i);
        __m256 diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // 水平求和
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 res = _mm_add_ps(low, high);
    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);
    float total = _mm_cvtss_f32(res);

    // 处理尾部元素（标量部分）
    for (; i < dim; ++i) {
        float diff = vec1[i] - vec2[i];
        total += diff * diff;
    }

    return total;
}


//采用avx求L2距离
static inline float euclideanDistance_avx2(
    const float* __restrict a,  // 向量1 [ALIGN64]
    const float* __restrict b,  // 向量2 [ALIGN64]
    const int dim                   // 向量维度
) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    // 主循环处理完整AVX块（8元素/块）
    const int avx_blocks = dim / 8;
    for (; i < avx_blocks * 8; i += 8) {
        const __m256 a_vec = _mm256_load_ps(a + i);  // 对齐加载
        const __m256 b_vec = _mm256_load_ps(b + i);
        const __m256 diff = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(diff, diff, sum);     // 融合乘加
    }

    // 处理剩余元素（非AVX对齐部分）
    float tail_sum = 0;
    for (; i < dim; ++i) {
        const float delta = a[i] - b[i];
        tail_sum += delta * delta;
    }

    // 水平归约求和
    __m128 low = _mm256_extractf128_ps(sum, 0);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    low = _mm_add_ps(low, high);            // 合并高低128位
    __m128 shuf = _mm_movehdup_ps(low);     // 高效水平求和
    __m128 sums = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    
    return _mm_cvtss_f32(sums) + tail_sum; 
}




// 编译选项: -mavx512f -mavx512dq -march=native -O3
#define PREFETCH_DISTANCE 512  // 根据L1d Cache调整
#define FLOATS_PER_VEC 16      // 每个AVX512向量包含16个float

    
float euclideanDistance_avx512(const float* a, const float* b, size_t n) {

    const size_t total_vectors = n / FLOATS_PER_VEC;
    const size_t remainder = n % FLOATS_PER_VEC;
    const float* a_end = a + total_vectors * FLOATS_PER_VEC;
    const float* b_end = b + total_vectors * FLOATS_PER_VEC;

    // 使用8个累加器提升并行度
    __m512 sum[8] = {_mm512_setzero_ps()}; // 数组化处理更易维护

    // 主循环展开8次 + 软件流水优化
    size_t vec_count = 0;
    const size_t prefetch_step = PREFETCH_DISTANCE / (FLOATS_PER_VEC*sizeof(float));
    
    for (; vec_count + 8 <= total_vectors; vec_count += 8) {
        // 软件流水：提前预取未来数据
        _mm_prefetch((const char*)(a + vec_count*FLOATS_PER_VEC + prefetch_step), _MM_HINT_T0);
        _mm_prefetch((const char*)(b + vec_count*FLOATS_PER_VEC + prefetch_step), _MM_HINT_T0);

        // 批量加载数据（优化缓存访问）
        __m512 av[8], bv[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            av[i] = _mm512_loadu_ps(a + (vec_count+i)*FLOATS_PER_VEC);
            bv[i] = _mm512_loadu_ps(b + (vec_count+i)*FLOATS_PER_VEC);
        }

        // 交错计算减少延迟（4组独立计算）
        sum[0] = _mm512_fmadd_ps(_mm512_sub_ps(av[0], bv[0]), 
                            _mm512_sub_ps(av[0], bv[0]), sum[0]);
        sum[1] = _mm512_fmadd_ps(_mm512_sub_ps(av[1], bv[1]), 
                            _mm512_sub_ps(av[1], bv[1]), sum[1]);
        sum[2] = _mm512_fmadd_ps(_mm512_sub_ps(av[2], bv[2]), 
                            _mm512_sub_ps(av[2], bv[2]), sum[2]);
        sum[3] = _mm512_fmadd_ps(_mm512_sub_ps(av[3], bv[3]), 
                            _mm512_sub_ps(av[3], bv[3]), sum[3]);
        // 第二组计算
        sum[4] = _mm512_fmadd_ps(_mm512_sub_ps(av[4], bv[4]), 
                            _mm512_sub_ps(av[4], bv[4]), sum[4]);
        sum[5] = _mm512_fmadd_ps(_mm512_sub_ps(av[5], bv[5]), 
                            _mm512_sub_ps(av[5], bv[5]), sum[5]);
        sum[6] = _mm512_fmadd_ps(_mm512_sub_ps(av[6], bv[6]), 
                            _mm512_sub_ps(av[6], bv[6]), sum[6]);
        sum[7] = _mm512_fmadd_ps(_mm512_sub_ps(av[7], bv[7]), 
                            _mm512_sub_ps(av[7], bv[7]), sum[7]);
    }

    // 处理剩余完整块（0-7个向量）
    for (; vec_count < total_vectors; ++vec_count) {
        __m512 av = _mm512_loadu_ps(a + vec_count*FLOATS_PER_VEC);
        __m512 bv = _mm512_loadu_ps(b + vec_count*FLOATS_PER_VEC);
        __m512 diff = _mm512_sub_ps(av, bv);
        sum[0] = _mm512_fmadd_ps(diff, diff, sum[0]);
    }

    // 合并累加器（并行归约）
    for (int i = 4; i < 8; ++i) sum[i%4] = _mm512_add_ps(sum[i%4], sum[i]);
    for (int i = 2; i < 4; ++i) sum[i%2] = _mm512_add_ps(sum[i%2], sum[i]);
    sum[0] = _mm512_add_ps(sum[0], sum[1]);

    // 处理尾部数据（修正掩码错误）
    if (remainder > 0) {
        const __mmask16 mask = _cvtu32_mask16((1U << remainder) - 1);
        const __m512 a_tail = _mm512_maskz_loadu_ps(mask, a_end);
        const __m512 b_tail = _mm512_maskz_loadu_ps(mask, b_end);
        const __m512 diff = _mm512_sub_ps(a_tail, b_tail);
        sum[0] = _mm512_fmadd_ps(diff, diff, sum[0]); // 移除了掩码FMA
    }

    // 水平求和优化
    alignas(64) float buffer[16];
    _mm512_store_ps(buffer, sum[0]);
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) total += buffer[i];

    // 使用快速平方根指令
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(total)));
}



/* 利用 AVX 进行 uint8_t 的点积运算 */

// 水平相加AVX寄存器中的8个uint32_t
static inline uint32_t horizontal_sum(__m256i x) {
    __m128i low = _mm256_castsi256_si128(x);
    __m128i high = _mm256_extracti128_si256(x, 1);
    low = _mm_add_epi32(low, high);
    low = _mm_hadd_epi32(low, low);
    low = _mm_hadd_epi32(low, low);
    return _mm_extract_epi32(low, 0);
}

static inline uint32_t dot_product(const uint8_t *a, const uint8_t *b, int size) {
    uint32_t sum = 0;
    size_t i = 0;

    // 每次处理32个元素，分4个块，每个块8元素
    const size_t block_size = 32;
    const size_t vec_size = 8;
    size_t blocks = size / block_size;
    size_t remainder = size % block_size;

    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    for (; i < blocks * block_size; i += block_size) {
        // 处理4个8元素块
        __m128i a_vec0 = _mm_loadl_epi64((const __m128i*)(a + i));
        __m128i b_vec0 = _mm_loadl_epi64((const __m128i*)(b + i));
        __m256i a_ext0 = _mm256_cvtepu8_epi32(a_vec0);
        __m256i b_ext0 = _mm256_cvtepu8_epi32(b_vec0);
        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(a_ext0, b_ext0));

        __m128i a_vec1 = _mm_loadl_epi64((const __m128i*)(a + i + 8));
        __m128i b_vec1 = _mm_loadl_epi64((const __m128i*)(b + i + 8));
        __m256i a_ext1 = _mm256_cvtepu8_epi32(a_vec1);
        __m256i b_ext1 = _mm256_cvtepu8_epi32(b_vec1);
        acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(a_ext1, b_ext1));

        __m128i a_vec2 = _mm_loadl_epi64((const __m128i*)(a + i + 16));
        __m128i b_vec2 = _mm_loadl_epi64((const __m128i*)(b + i + 16));
        __m256i a_ext2 = _mm256_cvtepu8_epi32(a_vec2);
        __m256i b_ext2 = _mm256_cvtepu8_epi32(b_vec2);
        acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(a_ext2, b_ext2));

        __m128i a_vec3 = _mm_loadl_epi64((const __m128i*)(a + i + 24));
        __m128i b_vec3 = _mm_loadl_epi64((const __m128i*)(b + i + 24));
        __m256i a_ext3 = _mm256_cvtepu8_epi32(a_vec3);
        __m256i b_ext3 = _mm256_cvtepu8_epi32(b_vec3);
        acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(a_ext3, b_ext3));
    }

    // 合并累加器
    acc0 = _mm256_add_epi32(acc0, acc1);
    acc0 = _mm256_add_epi32(acc0, acc2);
    acc0 = _mm256_add_epi32(acc0, acc3);

    // 处理剩余元素（每次8个）
    for (; i + vec_size <= size; i += vec_size) {
        __m128i a_vec = _mm_loadl_epi64((const __m128i*)(a + i));
        __m128i b_vec = _mm_loadl_epi64((const __m128i*)(b + i));
        __m256i a_ext = _mm256_cvtepu8_epi32(a_vec);
        __m256i b_ext = _mm256_cvtepu8_epi32(b_vec);
        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(a_ext, b_ext));
    }

    // 累加SIMD结果
    sum += horizontal_sum(acc0);

    // 处理最后的剩余元素（不足8个）
    for (; i < size; ++i) {
        sum += (uint32_t)a[i] * (uint32_t)b[i];
    }

    return sum;
}


//王哲的实现


uint32_t dot_product_uint8(const uint8_t* a, const uint8_t* b, size_t n) {
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    
    for (size_t i = 0; i < n; i += 64) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b0 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(a + i + 32));
        __m256i b1 = _mm256_loadu_si256((const __m256i*)(b + i + 32));

        // 处理低位
        __m256i al = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a0));
        __m256i bl = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b0));
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(al, bl));
        
        // 处理高位
        __m256i ah = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a0, 1));
        __m256i bh = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b0, 1));
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(ah, bh));
        
        // 处理第二个寄存器
        al = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a1));
        bl = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b1));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(al, bl));
        
        ah = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a1, 1));
        bh = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b1, 1));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(ah, bh));
    }

    sum0 = _mm256_add_epi32(sum0, sum1);
    
    __m128i res = _mm_add_epi32(_mm256_extracti128_si256(sum0, 1),
                               _mm256_castsi256_si128(sum0));
    res = _mm_add_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(2,3,0,1)));
    res = _mm_add_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(1,0,3,2)));
    return _mm_extract_epi32(res, 0);
}


uint32_t dot_product_uint8_optimized(const uint8_t* a, const uint8_t* b, size_t n) {
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    size_t i = 0;
    const size_t block_size = 64; // Process 64 elements per iteration
    for (; i + block_size <= n; i += block_size) {
        // Load 64 elements from a and b
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b0 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(a + i + 32));
        __m256i b1 = _mm256_loadu_si256((const __m256i*)(b + i + 32));

        // Process lower halves of a0 and b0 (sum0)
        __m256i al = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a0));
        __m256i bl = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b0));
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(al, bl));

        // Process upper halves of a0 and b0 (sum1)
        __m256i ah = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a0, 1));
        __m256i bh = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b0, 1));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(ah, bh));

        // Process lower halves of a1 and b1 (sum2)
        al = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a1));
        bl = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b1));
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(al, bl));

        // Process upper halves of a1 and b1 (sum3)
        ah = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a1, 1));
        bh = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b1, 1));
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(ah, bh));
    }

    // Accumulate all partial sums
    sum0 = _mm256_add_epi32(sum0, sum1);
    sum2 = _mm256_add_epi32(sum2, sum3);
    sum0 = _mm256_add_epi32(sum0, sum2);

    // Handle remaining elements (1-63 bytes)
    uint32_t tail_sum = 0;
    for (; i < n; ++i) {
        tail_sum += (uint32_t)a[i] * (uint32_t)b[i];
    }

    // Horizontal sum of 256-bit accumulator
    __m128i low = _mm256_castsi256_si128(sum0);
    __m128i high = _mm256_extracti128_si256(sum0, 1);
    low = _mm_add_epi32(low, high);
    low = _mm_add_epi32(low, _mm_shuffle_epi32(low, _MM_SHUFFLE(2, 3, 0, 1)));
    low = _mm_add_epi32(low, _mm_shuffle_epi32(low, _MM_SHUFFLE(1, 0, 3, 2)));
    
    return _mm_extract_epi32(low, 0) + tail_sum;
}


// AVX 512 方案
//dot_product_uint8_avx512 性能表现最好，re-rank 300 能到 42 us
uint32_t dot_product_uint8_avx512(const uint8_t* a, const uint8_t* b, size_t n) {
    __m512i sum = _mm512_setzero_si512();

    for (size_t i = 0; i < n; i += 64) {
        // 一次性加载 64 个 uint8 元素
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        // 将 512 位向量拆分为低/高 256 位并扩展为 int16
        __m256i a_low = _mm512_castsi512_si256(a_vec);
        __m256i a_high = _mm512_extracti64x4_epi64(a_vec, 1);
        __m256i b_low = _mm512_castsi512_si256(b_vec);
        __m256i b_high = _mm512_extracti64x4_epi64(b_vec, 1);

        // 扩展 uint8 -> int16 并计算乘积和
        __m512i prod_low = _mm512_madd_epi16(
            _mm512_cvtepu8_epi16(a_low),
            _mm512_cvtepu8_epi16(b_low)
        );
        __m512i prod_high = _mm512_madd_epi16(
            _mm512_cvtepu8_epi16(a_high),
            _mm512_cvtepu8_epi16(b_high)
        );

        // 累加结果
        sum = _mm512_add_epi32(sum, prod_low);
        sum = _mm512_add_epi32(sum, prod_high);
    }

    // 水平求和 512 位寄存器中的 16 个 int32
    return _mm512_reduce_add_epi32(sum);
}






// 辅助函数：水平求和 __m512i 中的 16 个 int32_t
static inline int32_t _mm512_hadd_epi32(__m512i v) {
    __m256i v256_0 = _mm512_extracti64x4_epi64(v, 0);
    __m256i v256_1 = _mm512_extracti64x4_epi64(v, 1);
    __m256i sum256 = _mm256_add_epi32(v256_0, v256_1);

    __m128i v128_0 = _mm256_extracti128_si256(sum256, 0);
    __m128i v128_1 = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(v128_0, v128_1);

    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_extract_epi32(sum128, 0);
}

int32_t dot_product_uint8_avx512_2(const uint8_t* a, const uint8_t* b, size_t n) {
    const size_t BLOCK_SIZE = 128; // 每次处理 128 个元素（4x32）
    size_t i = 0;
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();

    // 主循环：每次处理 128 个元素，使用四个累加器隐藏延迟
    for (; i + BLOCK_SIZE <= n; i += BLOCK_SIZE) {
        for (int j = 0; j < 4; ++j) {
            const size_t offset = i + j * 32;
            __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + offset));
            __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + offset));

            // 扩展 uint8_t 到 int16_t
            __m512i a_ext = _mm512_cvtepu8_epi16(a_vec);
            __m512i b_ext = _mm512_cvtepu8_epi16(b_vec);

            // 计算乘积并累加
            __m512i prod = _mm512_madd_epi16(a_ext, b_ext);

            // 累加到对应的累加器
            switch (j % 4) {
                case 0: acc0 = _mm512_add_epi32(acc0, prod); break;
                case 1: acc1 = _mm512_add_epi32(acc1, prod); break;
                case 2: acc2 = _mm512_add_epi32(acc2, prod); break;
                case 3: acc3 = _mm512_add_epi32(acc3, prod); break;
            }
        }
    }

    // 合并累加器
    acc0 = _mm512_add_epi32(acc0, acc1);
    acc2 = _mm512_add_epi32(acc2, acc3);
    __m512i acc = _mm512_add_epi32(acc0, acc2);

    // 处理剩余不足 128 的部分（每次处理 32 元素）
    for (; i + 31 < n; i += 32) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));

        __m512i a_ext = _mm512_cvtepu8_epi16(a_vec);
        __m512i b_ext = _mm512_cvtepu8_epi16(b_vec);

        __m512i prod = _mm512_madd_epi16(a_ext, b_ext);
        acc = _mm512_add_epi32(acc, prod);
    }

    // 水平求和
    int32_t sum = _mm512_hadd_epi32(acc);

    // 处理尾部剩余元素（不足 32 个）
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}


#endif