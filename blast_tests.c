#include "rt.h"
#include "blast.h"
#include "dot.h"

// TODO: test 1..16 all types, test permutations of offset and shift, test limited max_items = 4, max_groups = 2, test huge, test performance

static uint32_t seed;

static size_t sizes[] = { sizeof(fp16_t), sizeof(fp32_t), sizeof(fp64_t) };

typedef struct test_dot_s {
    int64_t bytes0;
    int64_t bytes1;
    blast_memory_t v0;
    blast_memory_t v1;
    void* a0;
    void* a1;
    double expected;
    double dot;
    double rse; // root square error
} test_dot_t;

static test_dot_t test_dot_alloc(blast_t* b, int fpp, int64_t n0, int64_t n1) {
    test_dot_t td = {0};
    td.bytes0 = n0 * sizes[fpp];
    td.bytes1 = n1 * sizes[fpp];
    enum { write_only = CL_MEM_WRITE_ONLY|CL_MEM_HOST_WRITE_ONLY };
    td.v0 = blast.allocate(b, write_only, td.bytes0);
    td.v1 = blast.allocate(b, write_only, td.bytes1);
    return td;
}

static void test_dot_map(test_dot_t* td) {
    td->a0 = blast.map(&td->v0, CL_MAP_WRITE_INVALIDATE_REGION, 0, td->bytes0);
    td->a1 = blast.map(&td->v1, CL_MAP_WRITE_INVALIDATE_REGION, 0, td->bytes1);
}

static void test_dot_unmap(test_dot_t* td) {
    blast.unmap(&td->v0);
    blast.unmap(&td->v1);
//  ocl.migrate(td->v0.b->c, td->v0.h);
//  ocl.migrate(td->v1.b->c, td->v1.h);
}

static void test_dot_free(test_dot_t* td) {
    blast.deallocate(&td->v0);
    blast.deallocate(&td->v1);
}

static void test_first_n(blast_t* b, int64_t n, int fpp,
        int64_t o0, int64_t s0, int64_t o1, int64_t s1, bool verbose) {
    assert(1 <= n && n <= 16);
    assert(o0 >= 0 && s0 >= 1 && o1 >= 0 && s1 >= 1);
    #pragma push_macro("at0")
    #pragma push_macro("at1")
    #define at0(type, i) ((type*)td.a0 + o0 + i * s0)
    #define at1(type, i) ((type*)td.a1 + o1 + i * s1)
    test_dot_t td = test_dot_alloc(b, fpp, o0 + n * s0, o1 + n * s1);
    test_dot_map(&td);
    // init memory by garbage
    for (int i = 0; i < td.bytes0; i++) {
        *((byte_t*)td.a0 + i) = (byte_t)random32(&seed);
    }
    for (int i = 0; i < td.bytes1; i++) { // init memory by garbage
        *((byte_t*)td.a1 + i) = (byte_t)random32(&seed);
    }
    td.expected = 0;
    for (int i = 0; i < n; i++) {
        if (fpp == ocl_fpp16) {
            *at0(fp16_t, i) = fp32to16((fp32_t)(i + 1));
            *at1(fp16_t, i) = fp32to16((fp32_t)(n - i));
        } else if (fpp == ocl_fpp32) {
            *at0(fp32_t, i) = (fp32_t)(i + 1);
            *at1(fp32_t, i) = (fp32_t)(n - i);
        } else if (fpp == ocl_fpp64) {
            *at0(fp64_t, i) = (fp64_t)(i + 1);
            *at1(fp64_t, i) = (fp64_t)(n - i);
        } else {
            fatal_if("fpp", "%d", fpp);
        }
        td.expected += (fp64_t)(i + 1) * (fp64_t)(n - i);
    }
    #pragma pop_macro("at1")
    #pragma pop_macro("at0")
    test_dot_unmap(&td);
    td.dot = 0;
    td.dot = b->dot[fpp](&td.v0, o0, s0, &td.v1, o1, s1, n);
    test_dot_free(&td);
    td.rse = td.expected - td.dot;
    td.rse = sqrt(td.rse * td.rse);
    if (verbose || td.rse > FLT_EPSILON) {
        println("%s[%2d] [o:%2d s:%2d] [o:%2d s:%2d] "
                "%25.17f expected: %25.17f rse: %.17f",
                ocl_fpp_names[fpp], n, o0, s0, o1, s1,
                td.dot, td.expected, td.rse);
    }
    fatal_if(td.rse > FLT_EPSILON);
}

static void test_permutations(blast_t* b) {
    for (int n = 1; n < 7; n++) {
        for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
            if (b->dot[fpp] != null) {
                for (int o0 = 0; o0 < 4; o0++) {
                    for (int o1 = 0; o1 < 4; o1++) {
                        for (int s0 = 1; s0 < 3; s0++) {
                            for (int s1 = 1; s1 < 3; s1++) {
                                test_first_n(b, n, fpp, o0, s0, o1, s1, false);
                            }
                        }
                    }
                }
            }
        }
    }
    for (int n = 1; n < 11; n++) {
        for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
            if (b->dot[fpp] != null) {
                for (int o0 = 0; o0 < 4; o0++) {
                    for (int o1 = 0; o1 < 4; o1++) {
                        for (int s0 = 1; s0 < 3; s0++) {
                            for (int s1 = 1; s1 < 3; s1++) {
                                test_first_n(b, n, fpp, o0, s0, o1, s1, false);
                            }
                        }
                    }
                }
            }
        }
    }
}

static void test_performance(blast_t* b, const int32_t n) {
    const int64_t bytes = n * sizeof(fp32_t);
    enum { write_only = CL_MEM_WRITE_ONLY|CL_MEM_HOST_WRITE_ONLY };
    blast_memory_t m0 = blast.allocate(b, write_only, bytes);
    blast_memory_t m1 = blast.allocate(b, write_only, bytes);
    fp32_t* x = (fp32_t*)blast.map(&m0, CL_MAP_WRITE_INVALIDATE_REGION, 0, bytes);
    fp32_t* y = (fp32_t*)blast.map(&m1, CL_MAP_WRITE_INVALIDATE_REGION, 0, bytes);
    fp32_t delta = (fp32_t)(1.0 / (double)(1ULL << 63));
    fp32_t sum = 0;
    for (int64_t i = 0; i < n; i++) {
        fp32_t sign = (i % 2 == 0 ? -1.0f : +1.f);
        x[i] = 1.0f + sign * ((i + 1) * delta);
        y[i] = 1.0f - sign * ((i + 1) * delta);
        assert(x[i] * y[i] == 1.0f);
        sum += x[i] * y[i];
    }
    blast.unmap(&m1);
    blast.unmap(&m0);
    fp64_t res = b->dot[ocl_fpp32](&m0, 0, 1, &m1, 0, 1, n);
    blast.deallocate(&m0);
    blast.deallocate(&m1);
    double rse = sqrt(pow(res - sum, 2));
    if (rse > FLT_EPSILON) {
        println("n: %d res: %.7E sum: %.7E sum - res: %.7E rse: %.7E\n",
                    n, res, sum, sum - res, rse);
    }
    assert(fabs(res - sum) <= FLT_EPSILON, "res: %.7e != %.7e\n", res, sum);
}

static void test_dot_compare_gpu_avx(blast_t* b,
        const ocl_profiling_t* p) {
    enum { n = 16 * 1024 * 1024 };
    test_dot_t td = test_dot_alloc(b, ocl_fpp32, n, n);
    test_dot_map(&td);
    fp32_t* x = (fp32_t*)td.a0;
    fp32_t* y = (fp32_t*)td.a1;
    fp32_t delta = (fp32_t)(1.0 / (double)(1ULL << 63));
    for (int64_t i = 0; i < n; i++) {
        fp32_t sign = (i % 2 == 0 ? -1.0f : +1.f);
        x[i] = 1.0f + sign * ((i + 1) * delta);
        y[i] = 1.0f - sign * ((i + 1) * delta);
        assert(x[i] * y[i] == 1.0f);
    }
    println("Nx1000,     AVX,       GPU, milliseconds");
    for (int i = 4096; i < n / 1024; i += 512) {
        double avx = seconds();
        fp64_t sum0 = dot.fp32(x, 1, y, 1, i * 1024);
        avx = seconds() - avx;
        test_dot_unmap(&td);
        double gpu = seconds();
        fp64_t sum1 = b->dot[ocl_fpp32](&td.v0, 0, 1, &td.v1, 0, 1, i * 1024);
        gpu = seconds() - gpu;
        gpu = p->time;
        test_dot_map(&td);
        x = (fp32_t*)td.a0;
        y = (fp32_t*)td.a1;
        println("%6d, %8.3f, %8.3f", i, avx * MSEC_IN_SEC, gpu * MSEC_IN_SEC);
        fatal_if(sum0 != sum1);
    }
    test_dot_unmap(&td);
    test_dot_free(&td);
}

static void tests() {
    if (dot.test != null) { dot.test(); }
    for (int dix = 0; dix < ocl.count; dix++) {
//      ocl.dump(dix);
        ocl_context_t c = ocl.open(dix, null);
        blast_t b = { 0 };
        blast.init(&b, &c);
        test_permutations(&b);
        blast.fini(&b);
        ocl.close(&c);
    }
    for (int d = 0; d < ocl.count; d++) {
        static ocl_profiling_t p[16 * 1024];
        static ocl_override_t ov = {
            .profiling = p,
            .max_profiling_count = countof(p),
            .profiling_count = 0,
        };
        ocl_context_t c = ocl.open(d, &ov);
        println("%s", ocl.devices[d].name);
        blast_t b = { 0 };
        blast.init(&b, &c);
        // because fp32 have 24 binary digits significand and 2^24 is 16M:
        // 16M is the largest number w/o losing precision
        enum { n = 16 * 1024 * 1024 };
        test_performance(&b, n);
        println("dot_fp32 x %d: %7.3f user: %7.3f (ms) GFlops: %7.3f", n,
            p[0].time * MSEC_IN_SEC, p[0].user * MSEC_IN_SEC, p[0].gflops);
        blast.fini(&b);
        ocl.close(&c);
    }
    for (int d = 0; d < ocl.count; d++) {
        static ocl_profiling_t p[16 * 1024];
        static ocl_override_t ov = {
            .profiling = p,
            .max_profiling_count = countof(p),
            .profiling_count = 0,
        };
        ocl_context_t c = ocl.open(d, &ov);
        println("%s", ocl.devices[d].name);
        blast_t b = { 0 };
        blast.init(&b, &c);
        test_dot_compare_gpu_avx(&b, p);
        blast.fini(&b);
        ocl.close(&c);
    }
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    ocl.init();
    tests();
    return 0;
}

/*

##11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz / 4.00 GHz

    TODO: document Intel
          11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz / 4.00 GHz
    and
          11th Gen Intel(R) Core(TM) i5-????
    TODO: move to performance.md

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

##Compare AVX vs GPU

###NVIDIA GeForce RTX 3080 Laptop GPU

dot_fp32 x 16,777,216:   5.413 user:  12.988 (ms) GFlops: 111.254

Nx1000,   AVX,     GPU, millisecond
  4096, 1.605,  27.338
  4608, 1.909,  28.990
  5120, 1.990,  29.465
  5632, 2.203,  29.602
  6144, 2.328,  29.690
  6656, 2.528,  32.793
  7168, 3.289,  33.768
  7680, 3.172,  35.278
  8192, 4.212,  33.517
  8704, 3.458,  33.645
  9216, 3.952,  35.780
  9728, 3.784,  39.720
 10240, 5.181,  38.146
 10752, 5.377,  38.698
 11264, 4.361,  38.514
 11776, 4.825,  40.229
 12288, 4.929,  39.591
 12800, 4.961,  41.711
 13312, 6.921,  43.100
 13824, 5.600,  46.037
 14336, 5.796,  46.795
 14848, 5.828,  46.704
 15360, 5.994,  47.089
 15872, 8.071,  46.622

###Intel(R) UHD Graphics i7-11800H

dot_fp32 x 16,777,216:  12.171 user: 779.006 (ms) GFlops:  44.923

Nx1000,   AVX,     GPU, millisecond
  4096, 1.612, 226.997
  4608, 1.837, 246.279
  5120, 2.099, 274.806
  5632, 2.543, 301.524
  6144, 2.333, 323.371
  6656, 2.472, 348.716
  7168, 2.931, 383.605
  7680, 3.066, 407.997
  8192, 3.182, 433.629
  8704, 3.302, 454.198
  9216, 3.390, 495.033
  9728, 3.723, 505.975
 10240, 3.911, 548.299
 10752, 4.043, 569.160
 11264, 4.336, 577.827
 11776, 4.409, 630.333
 12288, 5.801, 655.570
 12800, 4.811, 676.025
 13312, 5.014, 705.027
 13824, 5.334, 731.948
 14336, 5.941, 761.361
 14848, 5.954, 808.245
 15360, 5.909, 836.726
 15872, 6.261, 842.422


 ## AMD 7th Gen A9-9420 APU runs with 2 "GPU" devices:
  
  GPU: AMD A9-9420 RADEON R5, 5 COMPUTE CORES 2C+3G
    compute_units: 2 @ 2994MHz 7647MB
    OpenCL 1.2 C 1.2
    cl_khr_fp64
  GPU: Stoney OpenCL 2.0 C 2.0
    OpenCL 2.0 C 2.0
    compute_units: 3 @ 847MHz 3187MB
    *cl_khr_fp16* cl_khr_fp64 

    bf16 L1
    C     :   1.030 GFlops
    avx2  :   5.181 GFlops
    bf16 RAM
    C     :   1.048 GFlops
    avx2  :   3.799 GFlops
    fp16 L1
    C     :   0.129 GFlops
    avx2  :   8.091 GFlops
    fp16 RAM
    C     :   0.124 GFlops
    avx2  :   3.947 GFlops
    fp32 L1
    C     :   1.301 GFlops
    avx2  :   5.958 GFlops
    fp32 RAM
    C     :   1.210 GFlops
    avx2  :   2.130 GFlops
    fp64 L1
    C     :   1.242 GFlops
    fp64 RAM
    C     :   0.950 GFlops

###Stoney

dot_fp32 x 16777216:  39.468 user:  11.228 (ms) GFlops: 144.659

Nx1000,   AVX,     GPU, milliseconds
  4096,  668.777,   8.806
  4608,  785.833,   9.886
  5120,  846.521,  13.222
  5632,  931.697,  12.063
  6144,  991.968,  13.242
  6656, 1092.674,  15.451
  7168, 1168.366,  16.688
  7680, 1256.263,  17.909
  8192, 1347.751,  17.592
  8704, 1408.098,  24.292
  9216, 1493.905,  19.714
  9728, 1587.863,  20.792
 10240, 1633.865,  24.485
 10752, 1750.020,  25.677
 11264, 1811.006,  25.432
 11776, 1902.879,  25.381
 12288, 1995.633,  26.448
 12800, 2069.339,  27.478
 13312, 2142.564,  34.890
 13824, 2242.065,  29.734
 14336, 2312.985,  31.631
 14848, 2403.492,  33.604
 15360, 2471.045,  38.786
 15872, 2558.356,  34.218

###AMD A9-9420 RADEON R5, 5 COMPUTE CORES 2C+3G   

dot_fp32 x 16777216: 215.335 user:   0.778 (ms) GFlops: 104.773

Nx1000,   AVX,     GPU, milliseconds
  4096,  6.868,  44.344
  4608,  7.065,  46.644 
  5120,  8.326,  70.038
  5632, 12.074,  79.984
  6144,  9.719,  93.215
  6656, 11.991,  79.264
  7168, 12.847,  86.595
  7680, 14.022, 107.594
  8192, 12.718, 108.530
  8704, 23.717, 136.188
  9216, 11.775,  93.820
  9728, 16.925, 141.138
 10240, 19.159, 139.296
 10752, 17.939, 124.410
 11264, 20.808, 136.159
 11776, 23.602, 191.307
 12288, 24.607, 142.637
 12800, 25.410, 168.782
 13312, 22.229, 166.912
 13824, 22.995, 204.608
 14336, 28.556, 145.811
 14848, 27.416, 218.053
 15360, 30.391, 174.085
 15872, 25.585, 200.692
 */