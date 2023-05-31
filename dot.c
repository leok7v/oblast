#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include "dot.h"

// prefetch2_L1L2L3 - reportedly on 11th gen Intel processors it is 64 bytes (512 bits)
#define prefetch2_L1L2L3(v0, v1) do {             \
    _mm_prefetch((const char*)(v0), _MM_HINT_T0); \
    _mm_prefetch((const char*)(v1), _MM_HINT_T0); \
} while (0)

// AVX2 / AVX512 optimized dot product functions:

typedef struct avx2_if {
    void   (*init)(void);
    fp64_t (*dot32_c)(const fp32_t* restrict v0, const fp32_t* restrict v1, int64_t n);
    fp64_t (*dot64_c)(const fp64_t* restrict v0, const fp64_t* restrict v1, int64_t n);
} avx2_if;

typedef struct avx512_if {
    void   (*init)(void);
    fp64_t (*dot32_c)(const fp32_t* restrict v0, const fp32_t* restrict v1, int64_t n);
    fp64_t (*dot64_c)(const fp64_t* restrict v0, const fp64_t* restrict v1, int64_t n);
} avx512_if;

// _MM_HINT_T0 (temporal data) — prefetch data into all levels of the caches.
// _MM_HINT_T1 (temporal data with respect to first level cache misses) —
//              into L2 and higher.
// _MM_HINT_T2 (temporal data with respect to second level cache misses) —
//              into L3 and higher, or an implementation-specific choice.
// _MM_HINT_NTA (non-temporal data with respect to all cache levels) -
//              into non-temporal cache structure and into a location close
//              to the processor, minimizing cache pollution.

// Performance measuremnts confirm that prefetching into _MM_HINT_T0 is fastest

static void avx2_init(void);
static void avx512_init(void);

static avx2_if   avx2   = { .init = avx2_init };
static avx512_if avx512 = { .init = avx512_init };

static inline fp64_t cpu_dot16_c(const fp16_t* restrict v0,
        const fp16_t* restrict v1, int64_t n) {
    fp64_t sum = 0; // "_c" compact vector
    const fp16_t* e = v0 + n;
    while (v0 < e) { sum += fp16to32(fp16_mul(*v0++, *v1++)); }
    return sum;
}

static inline fp64_t cpu_dot16_s(const fp16_t* restrict v0, int64_t s0,
        const fp16_t* restrict v1, int64_t s1, int64_t n) {
    fp64_t sum = 0; // "_s" strided vector
    while (n > 0) { sum += fp16to32(fp16_mul(*v0, *v1)); v0 += s0; v1 += s1; n--; }
    return sum;
}

static inline fp64_t cpu_dot32_c(const fp32_t* restrict v0, const fp32_t* restrict v1,
        int64_t n) {
    fp64_t sum = 0;
    const fp32_t* e = v0 + n;
    while (v0 < e) { sum += *v0++ * *v1++; }
    return sum;
}

static inline fp64_t cpu_dot32_s(const fp32_t* restrict v0, int64_t s0,
        const fp32_t* restrict v1, int64_t s1, int64_t n) {
    fp64_t sum = 0;
    while (n > 0) { sum += *v0 * *v1; v0 += s0; v1 += s1; n--; }
    return sum;
}

static inline fp64_t cpu_dot64_c(const fp64_t* restrict v0,
        const fp64_t* restrict v1, int64_t n) {
    fp64_t sum = 0;
    const fp64_t* e = v0 + n;
    while (v0 < e) { sum += *v0++ * *v1++; }
    return sum;
}

static inline fp64_t cpu_dot64_s(const fp64_t* restrict v0, int64_t s0,
        const fp64_t* restrict v1, int64_t s1, int64_t n) {
    fp64_t sum = 0;
    while (n > 0) { sum += *v0 * *v1; v0 += s0; v1 += s1; n--; }
    return sum;
}

static fp64_t dot16_c(const fp16_t *v0, const fp16_t* v1, int64_t n) {
    prefetch2_L1L2L3(v0, v1);
    return cpu_dot16_c(v0, v1, n);
}

static fp64_t dot32_c(const fp32_t *v0, const fp32_t* v1, int64_t n) {
    prefetch2_L1L2L3(v0, v1);
    static bool init;
    if (!init) { avx2.init(); avx512.init(); init = true;}
    if (n >= 16 && avx512.dot32_c != null) {
        return avx512.dot32_c(v0, v1, n);
    } else if (n >= 8 && avx2.dot32_c != null) {
        return avx2.dot32_c(v0, v1, n);
    } else {
        return cpu_dot32_c(v0, v1, n);
    }
}

static fp64_t dot64_c(const fp64_t *v0, const fp64_t* v1, int64_t n) {
    prefetch2_L1L2L3(v0, v1);
    static bool init;
    if (!init) { avx2.init(); avx512.init(); init = true;}
    if (n >= 8 && avx512.dot64_c != null) {
        return avx512.dot64_c(v0, v1, n);
    } else if (n >= 4 && avx2.dot64_c != null) {
        return avx2.dot64_c(v0, v1, n);
    } else {
        return cpu_dot64_c(v0, v1, n);
    }
}

fp64_t dot16(const fp16_t* v0, int64_t s0, const fp16_t* v1, int64_t s1, int64_t n) {
    if (s0 == 1 && s1 == 1) {
        return dot16_c(v0, v1, n);
    } else {
        return cpu_dot16_s(v0, s0, v1, s1, n);
    }
}

fp64_t dot32(const fp32_t* v0, int64_t s0, const fp32_t* v1, int64_t s1, int64_t n) {
    if (s0 == 1 && s1 == 1) {
        return dot32_c(v0, v1, n);
    } else {
        return cpu_dot32_s(v0, s0, v1, s1, n);
    }
}

fp64_t dot64(const fp64_t* v0, int64_t s0, const fp64_t* v1, int64_t s1, int64_t n) {
    if (s0 == 1 && s1 == 1) {
        return dot64_c(v0, v1, n);
    } else {
        return cpu_dot64_s(v0, s0, v1, s1, n);
    }
}

// f64_t fp64_t
#define f64x2_t __m128d
#define f64x4_t __m256d
#define f64x8_t __m512d

// f32_t fp32_t
#define f32x4_t __m128
#define f32x8_t __m256
#define f32x16_t __m512

// f16_t
#define f16x16_t __m256bh
#define f16x32_t __m512bh

static fp64_t avx2_dot_f32(const fp32_t* restrict v0, const fp32_t* restrict v1,
        int64_t n) {
    fp64_t sum = 0;
    if (n >= 8) {
        f32x8_t mul_add_f32x8 = _mm256_setzero_ps();
        while (n >= 8) {
            f32x8_t a = _mm256_loadu_ps(v0);
            f32x8_t b = _mm256_loadu_ps(v1);
            n -= 8; v0 += 8; v1 += 8;
            if (n > 0) { prefetch2_L1L2L3(v0, v1); }
            mul_add_f32x8 = _mm256_fmadd_ps(a, b, mul_add_f32x8);
        }
        f32x4_t f32x4 = _mm_add_ps(
            _mm256_extractf32x4_ps(mul_add_f32x8, 0),  // 0,1,2,3
            _mm256_extractf32x4_ps(mul_add_f32x8, 1)); // 4,5,6,7
        sum = f32x4.m128_f32[0] + f32x4.m128_f32[1] + f32x4.m128_f32[2] + f32x4.m128_f32[3];
    }
    if (n > 0) { sum += cpu_dot32_c(v0, v1, n); }
    return sum;
}

static fp64_t avx2_dot_f64(const fp64_t* restrict v0,
        const fp64_t* restrict v1, int64_t n) {
    fp64_t sum = 0;
    if (n >= 4) {
        f64x4_t mul_add_f64x4 = _mm256_setzero_pd();
        while (n >= 4) {
            f64x4_t a = _mm256_loadu_pd(v0);
            f64x4_t b = _mm256_loadu_pd(v1);
            n -= 4; v0 +=4; v1 += 4;
            if (n > 0) { prefetch2_L1L2L3(v0, v1); }
            mul_add_f64x4 = _mm256_fmadd_pd(a, b, mul_add_f64x4);
        }
        f64x2_t f64x2 = _mm_add_pd(
            _mm256_castpd256_pd128(mul_add_f64x4),     // 0, 1
            _mm256_extractf64x2_pd(mul_add_f64x4, 1)); // 2, 3
        sum = f64x2.m128d_f64[0] + f64x2.m128d_f64[1];
    }
    if (n > 0) { sum += cpu_dot64_c(v0, v1, n); }
    return sum;
}

// avx512:

static fp64_t avx512_dot_f32(const fp32_t* restrict v0, const fp32_t* restrict v1, int64_t n) {
    fp64_t sum = 0;
    if (n >= 16) {
        f32x16_t mul_add_f32x16 = _mm512_setzero_ps(); // multiply and add
        while (n >= 16) {
            f32x16_t a = _mm512_loadu_ps(v0); // f32x8
            f32x16_t b = _mm512_loadu_ps(v1); // f32x8
            n -= 16; v0 +=16; v1 += 16;
            if (n > 0) { prefetch2_L1L2L3(v0, v1); }
            mul_add_f32x16 = _mm512_fmadd_ps(a, b, mul_add_f32x16);
        }
        // Reduce the 512-bit sum to a single 128-bit sum using AVX
        f32x8_t f32x8 = _mm256_add_ps(
            _mm512_castps512_ps256(mul_add_f32x16),     // 0,1,2,3,4,5,6,7
            _mm512_extractf32x8_ps(mul_add_f32x16, 1)); // 8,9,10,11,12,13,14,15
        f32x4_t f32x4 = _mm_add_ps(
            _mm256_castps256_ps128(f32x8),     // 0,1,2,3
            _mm256_extractf32x4_ps(f32x8, 1)); // 4,5,6,7
        sum = f32x4.m128_f32[0] + f32x4.m128_f32[1] + f32x4.m128_f32[2] + f32x4.m128_f32[3];
    }
    if (n > 0) { sum += cpu_dot32_c(v0, v1, n); }
    return sum;
}

static fp64_t avx512_dot_f64(const fp64_t* restrict v0, const fp64_t* restrict v1, int64_t n) {
    fp64_t sum = 0;
    if (n >= 8) {
        f64x8_t mul_add_f64x8 = _mm512_setzero_pd(); // multiply and add
        while (n >= 8) {
            f64x8_t a = _mm512_loadu_pd(v0); // f64x8
            f64x8_t b = _mm512_loadu_pd(v1); // f64x8
            n -= 8; v0 +=8; v1 += 8;
            if (n > 0) { prefetch2_L1L2L3(v0, v1); }
            mul_add_f64x8 = _mm512_fmadd_pd(a, b, mul_add_f64x8);
        }
        // Reduce the 512-bit sum to a single 128-bit sum using AVX
        f64x4_t f64x4 = _mm256_add_pd(
                _mm512_castpd512_pd256(mul_add_f64x8),     // 0,1,2,3
                _mm512_extractf64x4_pd(mul_add_f64x8, 1)); // 4,5,6,7
        f64x2_t f64x2 = _mm_add_pd(
            _mm256_castpd256_pd128(f64x4),     // 0,1
            _mm256_extractf64x2_pd(f64x4, 1)); // 2,3
        sum = f64x2.m128d_f64[0] + f64x2.m128d_f64[1];
    }
    if (n > 0) { sum += cpu_dot64_c(v0, v1, n); }
    return sum;
}

// 1. AXV512 on Gen-11 Intel CPU's measures slower then AVX2
// 2. AVX512-FP16
// https://cdrdv2-public.intel.com/678970/intel-avx512-fp16.pdf
// https://web.archive.org/web/20230426181123/https://cdrdv2-public.intel.com/678970/intel-avx512-fp16.pdf
//    is cool, but MAY NOT be supported on Gen-12 Intel CPU's:
// https://gist.github.com/FCLC/56e4b3f4a4d98cfd274d1430fabb9458
// 3. GPUs support FP16 for a long time and Microsoft Windows have weak
// provision for that:
// https://learn.microsoft.com/en-us/windows/win32/dxmath/half-data-type


static void avx2_init(void) {
    __try {
        fp32_t d0[16] = { 0 };
        fp32_t d1[16] = { 0 };
        fp64_t r = avx2_dot_f32(d0, d1, countof(d0));
        avx2.dot32_c = avx2_dot_f32;
        fatal_if(r != 0); // prevents optimizing out
    }
    __except (1) {
    }
    __try {
        fp64_t d0[16] = { 0 };
        fp64_t d1[16] = { 0 };
        fp64_t r = avx2_dot_f64(d0, d1, countof(d0));
        avx2.dot64_c = avx2_dot_f64;
        fatal_if(r != 0);
    } __except(1) {
    }
}

static void avx512_init(void) {
    __try {
        fp32_t d0[16] = { 0 };
        fp32_t d1[16] = { 0 };
        fp64_t r = avx512_dot_f32(d0, d1, countof(d0));
        avx512.dot32_c = avx512_dot_f32;
        fatal_if(r != 0);
    }
    __except (1) {
    }
    __try {
        fp64_t d0[16] = { 0 };
        fp64_t d1[16] = { 0 };
        fp64_t r = avx512_dot_f64(d0, d1, countof(d0));
        avx512.dot64_c = avx512_dot_f64;
        fatal_if(r != 0);
    }
    __except (1) {
    }
}

#undef DOT_TEST

#ifndef DOT_TEST

static void test_dot32_c() {
    fp32_t a[21];
    fp32_t b[21];
    for (int i = 0; i < countof(a); i++) {
        a[i] = (fp32_t)(i + 1);
        b[i] = (fp32_t)(countof(a) - i);
    }
    for (int i = 1; i < countof(a); i++) {
        fp64_t sum = 0;
        for (int j = 0; j < i; j++) { sum += a[j] * b[j]; }
        fp64_t sum0 = cpu_dot32_c(a, b, i);
        if (avx2.dot32_c != null) {
            fp64_t sum1 = avx2.dot32_c(a, b, i);
            fatal_if(fabs(sum - sum0) > FLT_EPSILON,
                "cpu: %.16f expected: %.16f delta: %.16e FLT_EPSILON: %.16e",
                sum0, sum, sum0 - sum, FLT_EPSILON);
            fatal_if(fabs(sum1 - sum0) > FLT_EPSILON,
                "cpu: %.16f avx: %.16f delta: %.16e FLT_EPSILON: %.16e",
                sum0, sum1, sum0 - sum1, FLT_EPSILON);
        }
        if (avx512.dot32_c != null) {
            fp64_t sum2 = avx512.dot32_c(a, b, i);
            fatal_if(fabs(sum2 - sum0) > FLT_EPSILON,
                "cpu: %.16f avx: %.16f delta: %.16e FLT_EPSILON: %.16e",
                sum0, sum2, sum0 - sum2, FLT_EPSILON);
        }
    }
}

static void test_dot64_c() {
    fp64_t a[21];
    fp64_t b[21];
    for (int i = 0; i < countof(a); i++) {
        a[i] = (fp64_t)(i + 1);
        b[i] = (fp64_t)(countof(a) - i);
    }
    for (int i = 1; i < countof(a); i++) {
        fp64_t sum = 0;
        for (int j = 0; j < i; j++) { sum += a[j] * b[j]; }
        fp64_t sum0 = cpu_dot64_c(a, b, i);
        if (avx2.dot64_c != null) {
            fp64_t sum1 = avx2.dot64_c(a, b, i);
            fatal_if(fabs(sum - sum0) > DBL_EPSILON,
                "cpu: %.16f expected: %.16f delta: %.16e DBL_EPSILON: %.16e",
                sum0, sum, sum0 - sum, DBL_EPSILON);
            fatal_if(fabs(sum1 - sum0) > DBL_EPSILON,
                "cpu: %.16f avx: %.16f delta: %.16e DBL_EPSILON: %.16e",
                sum0, sum1, sum0 - sum1, DBL_EPSILON);
        }
        if (avx512.dot64_c != null) {
            fp64_t sum2 = avx512.dot64_c(a, b, i);
            fatal_if(fabs(sum2 - sum0) > DBL_EPSILON,
                "cpu: %.16f avx: %.16f delta: %.16e DBL_EPSILON: %.16e",
                sum0, sum2, sum0 - sum2, DBL_EPSILON);
        }
    }
}

static uint64_t flushL1L2L3() {
    enum { count = 16 * 1024 * 1024 }; // 128MB
    uint64_t* L1L2L3 = (uint64_t*)malloc(count * sizeof(uint64_t));
    uint64_t sum = 0;
    if (L1L2L3 != null) {
        memset(L1L2L3, 0xFF, count * sizeof(uint64_t));
        for (int i = 0; i < countof(L1L2L3); i++) { sum |= L1L2L3[i]; }
        free(L1L2L3);
    }
    return sum;
}

typedef struct dot_performance_s { // per element
    fp64_t ns_c;
    fp64_t ns_avx2;
    fp64_t ns_avx512;
} dot_performance_t;

static void measure_dot32(int n, dot_performance_t* p) {
    enum { m = 128 * 1024 };
    typedef fp32_t vector_t[m];
    vector_t* a = (vector_t*)malloc(n * sizeof(vector_t));
    vector_t* b = (vector_t*)malloc(n * sizeof(vector_t));
    if (a != null && b != null) {
        fp64_t t = 0;
        uint32_t seed = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] = random32(&seed) / (fp32_t)UINT32_MAX - 0.5f;
                b[i][j] = random32(&seed) / (fp32_t)UINT32_MAX - 0.5f;
            }
        }
        // flush caches for n > 1:
        if (n > 1) { fatal_if(flushL1L2L3() == 0); }
        // C
        fp64_t ns_c = seconds() * NSEC_IN_SEC;
        for (int i = 0; i < n; i++) { t += cpu_dot32_c(a[i], b[i], m); }
        ns_c = seconds() * NSEC_IN_SEC - ns_c;
        p->ns_c = ns_c / (n * m);
        // AVX-2
        if (avx2.dot32_c != null) {
            if (n > 1) { fatal_if(flushL1L2L3() == 0); }
            fp64_t ns_avx2 = seconds() * NSEC_IN_SEC;
            for (int i = 0; i < n; i++) { t += avx2_dot_f32(a[i], b[i], m); }
            ns_avx2 = seconds() * NSEC_IN_SEC - ns_avx2;
            p->ns_avx2 = ns_avx2 / (n * m);
        }
        // AVX-512
        if (avx512.dot32_c != null) {
            if (n > 1) { fatal_if(flushL1L2L3() == 0); }
            fp64_t ns_avx512 = seconds() * NSEC_IN_SEC;
            for (int i = 0; i < n; i++) { t += avx512_dot_f32(a[i], b[i], m); }
            ns_avx512 = seconds() * NSEC_IN_SEC - ns_avx512;
            p->ns_avx512 = ns_avx512 / (n * m);
        }
        // t referenced to prevent compiler from optimizing out
        fatal_if(t == 0); // what are the odds of that?!
    }
    free(b); // free(null) is OK
    free(a);
}

static void measure_dot64(int n, dot_performance_t* p) {
    enum { m = 64 * 1024 };
    typedef fp64_t vector_t[m];
    vector_t* a = (vector_t*)malloc(n * sizeof(vector_t));
    vector_t* b = (vector_t*)malloc(n * sizeof(vector_t));
    if (a != null && b != null) {
        fp64_t t = 0;
        uint32_t seed = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] = random32(&seed) / (fp64_t)UINT32_MAX - 0.5f;
                b[i][j] = random32(&seed) / (fp64_t)UINT32_MAX - 0.5f;
            }
        }
        // C
        if (n > 1) { fatal_if(flushL1L2L3() == 0); }
        fp64_t ns_c = seconds() * NSEC_IN_SEC;
        for (int i = 0; i < n; i++) { t += cpu_dot64_c(a[i], b[i], m); }
        ns_c = seconds() * NSEC_IN_SEC - ns_c;
        p->ns_c = ns_c / (n * m);
        // AVX-2
        if (avx2.dot64_c != null) {
            if (n > 1) { fatal_if(flushL1L2L3() == 0); }
            fp64_t ns_avx2 = seconds() * NSEC_IN_SEC;
            for (int i = 0; i < n; i++) { t += avx2.dot64_c(a[i], b[i], m); }
            ns_avx2 = seconds() * NSEC_IN_SEC - ns_avx2;
            p->ns_avx2 = ns_avx2 / (n * m);
        }
        // AVX-512
        if (avx512.dot64_c != null) {
            if (n > 1) { fatal_if(flushL1L2L3() == 0); }
            fp64_t ns_avx512 = seconds() * NSEC_IN_SEC;
            for (int i = 0; i < n; i++) { t += avx512.dot64_c(a[i], b[i], m); }
            ns_avx512 = seconds() * NSEC_IN_SEC - ns_avx512;
            p->ns_avx512 = ns_avx512 / (n * m);
        }
        // t referenced to prevent compiler from optimizing out
        fatal_if(t == 0); // what are the odds of that?!
    }
    free(b); // free(null) is OK
    free(a);
}

static void performance(int n, int bestof, dot_performance_t* m,
    void (*measure)(int n, dot_performance_t* p)) {
    measure(n, m); // best of 100 runs
    for (int i = 0; i < bestof; i++) {
        dot_performance_t p = {0};
        measure(n, &p);
        m->ns_c      = min(m->ns_c, p.ns_c);
        m->ns_avx2   = min(m->ns_avx2, p.ns_avx2);
        m->ns_avx512 = min(m->ns_avx512, p.ns_avx512);
    }
}

static void report_preformance(dot_performance_t* p, const char* label) {
    println("%s", label);
//  println("C     : %7.3f nanoseconds", p->ns_c);
//  if (p->ns_avx2   != 0) { println("avx2  : %7.3f nanoseconds", p->ns_avx2); }
//  if (p->ns_avx512 != 0) { println("avx512: %7.3f nanoseconds", p->ns_avx512); }
    // GFlops (2 flops per element)
    fp64_t gfps_c      = 2.0 / p->ns_c;
    fp64_t gfps_avx2   = p->ns_avx2   != 0 ? 2.0 / p->ns_avx2   : 0;
    fp64_t gfps_avx512 = p->ns_avx512 != 0 ? 2.0 / p->ns_avx512 : 0;
    println("C     : %7.3f Gflops", gfps_c);
    if (p->ns_avx2   != 0) { println("avx2  : %7.3f Gflops", gfps_avx2); }
    if (p->ns_avx512 != 0) { println("avx512: %7.3f Gflops", gfps_avx512); }
}

static void dot_test_performance() {
    dot_performance_t p = {0};
    performance(1,   100, &p, measure_dot32); report_preformance(&p, "fp32 L1");
    performance(128,  25, &p, measure_dot32); report_preformance(&p, "fp32 RAM");
    performance(1,   100, &p, measure_dot64); report_preformance(&p, "fp64 L1");
    performance(128,  25, &p, measure_dot64); report_preformance(&p, "fp64 RAM");
}

void dot_init() {
    static bool init;
    if (!init) { avx2.init(); avx512.init(); init = true; }
}

void dot_test() {
    dot_init();
    test_dot32_c();
    test_dot64_c();
    dot_test_performance();
}

/*
    11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz / 4.00 GHz

    fp16 L1
    C     :   0.315 Gflops
    fp16 RAM
    C     :   0.306 Gflops

    fp32 L1
    C     :   2.254 Gflops
    avx2  :  17.955 Gflops
    avx512:  22.029 Gflops

    fp32 RAM
    C     :   2.232 Gflops
    avx2  :   6.974 Gflops
    avx512:   5.936 Gflops

    fp64 L1
    C     :   2.248 Gflops
    avx2  :   9.039 Gflops
    avx512:  11.015 Gflops

    fp64 RAM
    C     :   2.177 Gflops
    avx2  :   3.483 Gflops
    avx512:   2.980 Gflops (note: something wrong with fp64 prefetch)
*/

#endif // TEST_DOT_PRODUCT_PERFORMANCE
