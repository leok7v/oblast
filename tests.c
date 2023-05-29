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
    td.v0 = blast.allocate(b, blast_access_write, td.bytes0);
    td.v1 = blast.allocate(b, blast_access_write, td.bytes1);
    return td;
}

static void test_dot_map(test_dot_t* td) {
    td->a0 = blast.map(&td->v0, blast_access_write, 0, td->bytes0);
    td->a1 = blast.map(&td->v1, blast_access_write, 0, td->bytes1);
}

static void test_dot_unmap(test_dot_t* td) {
    blast.unmap(&td->v0);
    blast.unmap(&td->v1);
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
        traceln("%s[%2d] [o:%2d s:%2d] [o:%2d s:%2d] "
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
    blast_memory_t m0 = blast.allocate(b, blast_access_write, bytes);
    blast_memory_t m1 = blast.allocate(b, blast_access_write, bytes);
    fp32_t* x = (fp32_t*)blast.map(&m0, blast_access_write, 0, bytes);
    fp32_t* y = (fp32_t*)blast.map(&m1, blast_access_write, 0, bytes);
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
    fp64_t dot = b->dot[ocl_fpp32](&m0, 0, 1, &m1, 0, 1, n);
    blast.deallocate(&m0);
    blast.deallocate(&m1);
    double rse = sqrt(pow(dot - sum, 2));
    if (rse > FLT_EPSILON) {
        traceln("n: %d dot: %.7E sum: %.7E sum - dot: %.7E rse: %.7E\n",
                    n, dot, sum, sum - dot, rse);
    }
    assert(fabs(dot - sum) <= FLT_EPSILON, "dot: %.7e != %.7e\n", dot, sum);
}

static void test_dot_compare_gpu_avx(blast_t* b) {
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
    traceln("Nx1000,   AVX,     GPU, milliseconds");
    for (int i = 4096; i < n / 1024; i += 512) {
        double avx = seconds();
        fp64_t sum0 = dot32(x, 1, y, 1, i * 1024);
        avx = seconds() - avx;
        test_dot_unmap(&td);
        double gpu = seconds();
        fp64_t sum1 = b->dot[ocl_fpp32](&td.v0, 0, 1, &td.v1, 0, 1, i * 1024);
        gpu = seconds() - gpu;
        test_dot_map(&td);
        x = (fp32_t*)td.a0;
        y = (fp32_t*)td.a1;
        traceln("%6d, %5.3f, %7.3f", i, avx * MSEC_IN_SEC, gpu * MSEC_IN_SEC);
        fatal_if(sum0 != sum1);
    }
    test_dot_unmap(&td);
    test_dot_free(&td);
}

static void dot_tests() {
    dot_test();
    for (int d = 0; d < ocl.count; d++) {
//      ocl.dump(i);
        ocl_context_t c = ocl.open(d, null);
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
        traceln("%s", ocl.devices[d].name);
        blast_t b = { 0 };
        blast.init(&b, &c);
        // because fp32 have 24 binary digits significand and 2^24 is 16M:
        // 16M is the largest number w/o losing precision
        enum { n = 16 * 1024 * 1024 };
        test_performance(&b, n);
        traceln("dot_fp32 x %d: %7.3f user: %7.3f (ms) GFlops: %7.3f", n,
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
        traceln("%s", ocl.devices[d].name);
        blast_t b = { 0 };
        blast.init(&b, &c);
        test_dot_compare_gpu_avx(&b);
        blast.fini(&b);
        ocl.close(&c);
    }
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    ocl.init();
    dot_tests();
    return 0;
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


NVIDIA GeForce RTX 3080 Laptop GPU
dot_fp32 x 16,777,216:   5.413 user:  12.988 (ms) GFlops: 111.254

Intel(R) UHD Graphics
dot_fp32 x 16,777,216:  12.171 user: 779.006 (ms) GFlops:  44.923

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

Intel(R) UHD Graphics i7-11800H
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

 */