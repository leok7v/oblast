#include "rt.h"
#include "ocl.h"
#include "dot.h"
#include "gemv.h"
#include <conio.h>

static int  best_of = 3;
static bool verbose = true;
static bool unchecked;

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

static fp64_t cpu_time;
static fp64_t avx_time;
static fp64_t ocl_time; // total time for gemv() call
static fp64_t gpu_time; // gpu time
static fp64_t gpu_gfps; // gpu GFlops

static void v32println(const fp32_t* vc, const int32_t n) {
    for (int32_t i = 0; i < n; i++) { printf("%5g ", vc[i]); }
    printf("\n");
}

static void v64println(const fp64_t* vc, const int32_t n) {
    for (int32_t i = 0; i < n; i++) { printf("%5g ", vc[i]); }
    printf("\n");
}

static void m16println(const fp16_t* mx, const int32_t n, const int32_t m) {
    for (int32_t j = 0; j < m; j++) {
        printf("m[%3d] ", j);
        for (int32_t i = 0; i < n; i++) { printf("%5g ", fp16to32(mx[j * n + i])); }
        printf("\n");
    }
}

static void m32println(const fp32_t* mx, const int32_t n, const int32_t m) {
    for (int32_t j = 0; j < m; j++) {
        printf("m[%3d] ", j);
        for (int32_t i = 0; i < n; i++) { printf("%5g ", mx[j * n + i]); }
        printf("\n");
    }
}

static void m64println(const fp64_t* mx, const int32_t n, const int32_t m) {
    for (int32_t j = 0; j < m; j++) {
        printf("m[%3d] ", j);
        for (int32_t i = 0; i < n; i++) { printf("%5g ", mx[j * n + i]); }
        printf("\n");
    }
}

static void print(int fpp, int32_t n, int32_t m) { // performance measurements
    if (n > 64 && m > 64) {
        if (avx_time < DBL_MAX) {
            if (gpu_time < DBL_MAX) {
                println("fp%d_t %5d x %-5d gpu: %7.3f (call: %7.3f) avx: %8.3f "
                    "ms %5.1fGFlops",
                    ocl_fpp_bytes[fpp] * 8, n, m,
                    gpu_time * MSEC_IN_SEC, ocl_time * MSEC_IN_SEC,
                    avx_time * MSEC_IN_SEC, gpu_gfps);
            } else {
                gpu_gfps = 3.0 * m * n / (ocl_time * NSEC_IN_SEC);
                println("fp%d_t %5d x %-5d gpu: %7.3f avx: %8.3f ms %5.1fGFlops",
                    ocl_fpp_bytes[fpp] * 8, n, m,
                    ocl_time * MSEC_IN_SEC,
                    avx_time * MSEC_IN_SEC, gpu_gfps);
            }
        } else {
            if (gpu_time < DBL_MAX) {
                println("fp%d_t %5d x %-5d gpu: %7.3f (call: %7.3f) cpu: %8.3f ms "
                    "%5.1fGFlops",
                    ocl_fpp_bytes[fpp] * 8, n, m,
                    gpu_time * MSEC_IN_SEC, ocl_time * MSEC_IN_SEC,
                    cpu_time * MSEC_IN_SEC, gpu_gfps);
            } else {
                gpu_gfps = 3.0 * m * n / (ocl_time * NSEC_IN_SEC);
                println("fp%d_t %5d x %-5d gpu: %7.3f cpu: %8.3f ms %5.1fGFlops",
                    ocl_fpp_bytes[fpp] * 8, n, m,
                    ocl_time * MSEC_IN_SEC,
                    cpu_time * MSEC_IN_SEC, gpu_gfps);
            }
        }
    }
}

static void test(gemv_t* g, int fpp,
                 int32_t n, int32_t m,
                 fp64_t (*init_vc)(int32_t i),
                 fp64_t (*init_mx)(int32_t j, int32_t i, int32_t n)) {
    ocl_context_t* c = g->c;
//  println("%d x %d fp%d_t", n, m, ocl_fpp_bytes[fpp] * 8);
    ocl_time = DBL_MAX;
    gpu_time = DBL_MAX;
    avx_time = DBL_MAX;
    cpu_time = DBL_MAX;
    gpu_gfps = 0;
    const size_t meb = ocl_fpp_bytes[fpp]; // matrix element bytes
    // vector element bytes
    const size_t veb = fpp == ocl_fpp16 ? 4 : 8;
    ocl_memory_t matrix = ocl.allocate(c, ocl_allocate_write, (size_t)m * n * meb);
    ocl_memory_t vector = ocl.allocate(c, ocl_allocate_write, (size_t)n * veb);
    ocl_memory_t result = ocl.allocate(c, ocl_allocate_read,  (size_t)m * veb);
    void* mx = ocl.map(c, ocl_map_write, matrix, 0, (size_t)m * n * meb);
    void* vc = (fp32_t*)ocl.map(c, ocl_map_write, vector, 0, n * veb);
    for (int32_t i = 0; i < n; i++) {
        switch (fpp) {
            case ocl_fpp16:
            case ocl_fpp32: ((fp32_t*)vc)[i] = (fp32_t)init_vc(i); break;
            case ocl_fpp64: ((fp64_t*)vc)[i] = init_vc(i); break;
            default: fatal_if("fpp?", "fpp: %d", fpp);
        }
    }
    byte_t* p = (byte_t*)mx;
    for (int32_t j = 0; j < m; j++) {
        for (int32_t i = 0; i < n; i++) {
            switch (fpp) {
                case ocl_fpp16: *((fp16_t*)p) = fp32to16((fp32_t)init_mx(j, i, n)); break;
                case ocl_fpp32: *((fp32_t*)p) = (fp32_t)init_mx(j, i, n); break;
                case ocl_fpp64: *((fp64_t*)p) = init_mx(j, i, n); break;
                default: fatal_if("fpp?", "fpp: %d", fpp);
            }
            p += meb;
        }
    }
    void* avx = (fp32_t*)alloca(m * veb);
    fatal_if(avx == null);
    if (fpp != ocl_fpp16) {
        fp64_t user = seconds();
        switch (fpp) {
            case ocl_fpp32:
                for (int32_t j = 0; j < m; j++) {
                    fp32_t* row = (fp32_t*)mx + j * n;
                    ((fp32_t*)avx)[j] = (fp32_t)dot32((fp32_t*)vc, 1, row, 1, n);
                }
                break;
            case ocl_fpp64:
                for (int64_t j = 0; j < m; j++) {
                    fp64_t* row = (fp64_t*)mx + j * n;
                    ((fp64_t*)avx)[j] = dot64((fp64_t*)vc, 1, row, 1, n);
                }
                break;
            default:
                fatal_if("fpp?", "fpp: %d", fpp);
        }
        user = seconds() - user;
        avx_time = min(avx_time, user);
    }
    void* cpu = (fp32_t*)alloca(m * veb);
    fatal_if(cpu == null);
    fp64_t user = seconds();
    switch (fpp) {
        case ocl_fpp16:
            for (int32_t j = 0; j < m; j++) {
                fp16_t* row = (fp16_t*)mx + j * n;
                fp32_t s = 0;
                for (int32_t i = 0; i < n; i++) {
                    s += ((fp32_t*)vc)[i] * fp16to32(row[i]);
                }
                ((fp32_t*)cpu)[j] = s;
            }
            break;
        case ocl_fpp32:
            for (int32_t j = 0; j < m; j++) {
                fp32_t* row = (fp32_t*)mx + j * n;
                fp32_t s = 0;
                for (int32_t i = 0; i < n; i++) {
                    s += ((fp32_t*)vc)[i] * row[i];
                }
                ((fp32_t*)cpu)[j] = s;
            }
            break;
        case ocl_fpp64:
            for (int64_t j = 0; j < m; j++) {
                fp64_t* row = (fp64_t*)mx + j * n;
                fp64_t s = 0;
                for (int64_t i = 0; i < n; i++) {
                    s += ((fp64_t*)vc)[i] * row[i];
                }
                ((fp64_t*)cpu)[j] = s;
            }
            break;
        default:
            fatal_if("fpp?", "fpp: %d", fpp);
    }
    user = seconds() - user;
    cpu_time = min(cpu_time, user);
    if (verbose && n <= 64 && m <= 64) {
        switch (fpp) {
            case ocl_fpp16:
                printf("mx:\n"); m16println((fp16_t*)mx, n, m);
                printf("vc : "); v32println((fp32_t*)vc, n);
                printf("cpu: "); v32println((fp32_t*)cpu, m);
                break;
            case ocl_fpp32:
                printf("mx:\n"); m32println((fp32_t*)mx, n, m);
                printf("vc : "); v32println((fp32_t*)vc, n);
                printf("cpu: "); v32println((fp32_t*)cpu, m);
                printf("avx: "); v32println((fp32_t*)avx, m);
                break;
            case ocl_fpp64:
                printf("mx:\n"); m64println((fp64_t*)mx, n, m);
                printf("vc : "); v64println((fp64_t*)vc, n);
                printf("cpu: "); v64println((fp64_t*)cpu, m);
                printf("avx: "); v64println((fp64_t*)avx, m);
                break;
            default: fatal_if("fpp?", "fpp: %d", fpp);
        }
    }
    ocl.unmap(c, matrix, mx);
    ocl.unmap(c, vector, vc);
    user = seconds();
    ocl.migrate(c, matrix); // these are "hints"
    ocl.migrate(c, vector);
    ocl.migrate_undefined(c, result);
    user = seconds() - user;
//  println("migrate: %.6f ms", user * 1000.0);
    assert(best_of >= 1);
    for (int repeat = 0; repeat < best_of; repeat++) {
        user = seconds();
        gemv.gemv(g, fpp, 0, matrix, vector, result, n, m);
        user = seconds() - user;
        ocl_time = min(ocl_time, user);
        if (ocl.is_profiling(g->c)) {
            ocl_profiling_t* pf = &g->c->ov->profiling[0];
            gpu_time = min(gpu_time, pf->time);
            gpu_gfps = max(gpu_gfps, pf->gflops);
        }
    }
    void* rs = ocl.map(c, ocl_map_read, result, 0, m * veb);
    if (verbose && m <= 64) {
        printf("gpu: ");
        switch (fpp) {
            case ocl_fpp16:
            case ocl_fpp32: v32println((fp32_t*)rs, m); break;
            case ocl_fpp64: v64println((fp64_t*)rs, m); break;
            default: fatal_if("fpp?", "fpp: %d", fpp);
        }
    }
    // verification
    const fp64_t epsilon = CL_FLT_EPSILON * n * m;
    for (int32_t j = 0; j < m; j++) {
        fp64_t delta_cpu_gpu = 0;
        fp64_t delta_avx_gpu = 0;
        fp64_t delta_avx_cpu = 0;
        switch (fpp) {
            case ocl_fpp16:
            case ocl_fpp32:
                delta_cpu_gpu = fabs(((fp32_t*)cpu)[j] - ((fp32_t*)rs)[j]);
                if (fpp != ocl_fpp16) {
                    delta_avx_cpu = fabs(((fp32_t*)cpu)[j] - ((fp32_t*)avx)[j]);
                    delta_avx_gpu = fabs(((fp32_t*)rs)[j] - ((fp32_t*)avx)[j]);
                }
                break;
            case ocl_fpp64:
                delta_cpu_gpu = fabs(((fp64_t*)cpu)[j] - ((fp64_t*)rs)[j]);
                delta_avx_cpu = fabs(((fp64_t*)cpu)[j] - ((fp64_t*)avx)[j]);
                delta_avx_gpu = fabs(((fp64_t*)rs)[j]     - ((fp64_t*)avx)[j]);
                break;
            default:
                fatal_if("fpp?", "fpp: %d", fpp);
        }
        if (!unchecked) {
            switch (fpp) {
                case ocl_fpp16:
                case ocl_fpp32:
                    fatal_if(delta_cpu_gpu > epsilon, "delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                        delta_cpu_gpu, epsilon, j, ((fp32_t*)cpu)[j], j, ((fp32_t*)rs)[j]);
                    fatal_if(delta_avx_cpu > epsilon, "delta: %g epsilon: %g cpu[%d]: %g avx[%d]: %g",
                        delta_avx_cpu, epsilon, j, ((fp32_t*)cpu)[j], j, ((fp32_t*)avx)[j]);
                    fatal_if(delta_avx_gpu > epsilon, "delta: %g epsilon: %g avx[%d]: %g gpu[%d]: %g",
                        delta_avx_gpu, epsilon, j, ((fp32_t*)avx)[j], j, ((fp32_t*)rs)[j]);
                    break;
                case ocl_fpp64:
                    fatal_if(delta_cpu_gpu > epsilon, "delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                        delta_cpu_gpu, epsilon, j, ((fp64_t*)cpu)[j], j, ((fp64_t*)rs)[j]);
                    fatal_if(delta_avx_cpu > epsilon, "delta: %g epsilon: %g cpu[%d]: %g avx[%d]: %g",
                        delta_avx_cpu, epsilon, j, ((fp64_t*)cpu)[j], j, ((fp64_t*)avx)[j]);
                    fatal_if(delta_avx_gpu > epsilon, "delta: %g epsilon: %g avx[%d]: %g gpu[%d]: %g",
                        delta_avx_gpu, epsilon, j, ((fp64_t*)avx)[j], j, ((fp64_t*)rs)[j]);
                    break;
                default:
                    fatal_if("fpp?", "fpp: %d", fpp);
            }
        }
    }
    // cleanup
    ocl.unmap(c, result, rs);
    ocl.deallocate(result);
    ocl.deallocate(vector);
    ocl.deallocate(matrix);
    print(fpp, n, m); // performance measurements
}

static fp64_t init_vc0(int32_t i) {
    return (fp64_t)(i + 1);
}

static fp64_t init_mx0(int32_t r, int32_t c, int32_t n) {
    int32_t ix = r * n + c;
    return (fp64_t)(ix + 1);
}

static fp64_t init_vc1(int32_t i) {
    return (fp64_t)(1.0 / pow(2, (i % 9)));
}

static fp64_t init_mx1(int32_t r, int32_t c, int32_t n) {
    int32_t ix = r * n + c;
    return (fp32_t)(1.0 / pow(2.0, (ix % 9)));
}

static void permutations(gemv_t* g, const ocl_device_t* d) {
#ifndef PERMUTATIONS_DEBUG_SPECIFIC
    // all 1..64 x 1..64 permutations
    verbose = false; // set to true if crashes
    for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
        for (int n = 1; n <= 64; n++) {
            for (int m = 1; m <= 64; m++) {
                if (fpp != ocl_fpp64 || d->double_fp_config != 0) {
//                  if (_isatty(_fileno(stdout))) {
//                      printf("%2dx%-2d: %d\r", n, m, ocl_fpp_bytes[fpp] * 8);
//                  }
                    test(g, fpp, n, m, init_vc0, init_mx0);
                }
            }
        }
    }
#else
    // debug and trace a single failing case from permutaions above
    verbose = true;
    test(g, ocl_fpp32, 33, 1, init_vc0, init_mx0);
#endif
}

static void tests(bool profile) {
    for (int i = 0; i < ocl.count; i++) {
//      ocl.dump(i);
        const ocl_device_t* d = &ocl.devices[i];
        ocl_profiling_t profiling[1]; // [1] because single kernel
        ocl_override_t ov = {
            .profiling = profiling,
            .max_profiling_count = countof(profiling),
            .profiling_count = 0
        };
        ocl_context_t c = ocl.open(i, profile ? &ov : null);
        println("*** %s : %.1fGB ***", d->name, d->global_memory / (double)GB);
        gemv_t g = {0};
        gemv.init(&g, &c);
        permutations(&g, d);
        // Intel(R) UHD Graphics is not capable of 4GB+ allocation
        // despite of 12.7GB shared memory size:
        int64_t global_memory = strstr(d->name, "Intel(R) UHD Graphics") ?
            4LL * GB : d->global_memory;
        // large matrix/vectors performance tests
        for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
            if (fpp != ocl_fpp64 || d->double_fp_config != 0) {
                struct { int32_t n; int32_t m; } tests[] = {
                    {     1024,      1024},
                    { 4 * 1024,  4 * 1024},
                    { 4 * 1024, 16 * 1024}, // GPT-J 6b innermost gemv()
                    {16 * 1024, 48 * 1024}, // GPT-J 6b innermost gemv() x 8 layers
                    {30 * 1024, 60 * 1024},
                    {64 * 1024,  8 * 1024},
                };
                for (int k = 0; k < countof(tests); k++) {
                    int64_t bytes = (int64_t)tests[k].n * tests[k].m *
                        ocl_fpp_bytes[fpp];
//                  println("%.3fGB", bytes / (double)GB);
                    if (bytes < global_memory - 128 * MB) {
                        test(&g, fpp, tests[k].n, tests[k].m, init_vc1, init_mx1);
                    }
                }
            }
        }
        gemv.fini(&g);
        ocl.close(&c);
    }
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    ocl.init();
    if (argc > 1 && strcmp(argv[1], "compile") == 0) {
        if (argc >= 3) {
            ocl.compiler(argc, argv);
        } else {
            println("compile <source> <options>\nNot enough arguments.");
        }
    } else {
        tests(true);  // with profiling
        tests(false); // w/o  profiling
    }
}

#if 0

OpenCL.gemv() (matrix[m,n] x vector[n])

NVIDIA GeForce RTX 3080 Laptop GPU

n:       m:                   gemv16x16() kernel
30,720 x 61,440 gpu:  65.250ms 87 GFlops
n:       m:                   gemv16x32() kernel
30,720 x 61,440 gpu: 218.131ms 26 GFlops
(at gemv16x32() compiler runs out of registers
 performance decreases)

Intel(R) UHD Graphics GPU:
n:       m:
32,768 x 2,048 gpu: 14.533ms 15 GFlops

May 30, 2023 results:

 *** NVIDIA GeForce RTX 3080 Laptop GPU ***
fp16_t  1024 x 1024  gpu:   4.262 (call:   4.285) cpu:    1.626 ms   0.7GFlops
fp16_t  1024 x 1024  gpu:   4.263 (call:   4.285) cpu:    1.663 ms   0.7GFlops
fp16_t  4096 x 4096  gpu:  42.332 (call:  42.401) cpu:   26.204 ms   1.2GFlops
fp16_t  4096 x 16384 gpu:  48.358 (call:  48.377) cpu:  108.772 ms   4.2GFlops
fp16_t 16384 x 65536 gpu:  83.343 (call:  83.373) cpu: 1769.062 ms  38.7GFlops
 --------------------------------------------------
fp16_t 30720 x 61440 gpu:  63.847 (call:  64.035) cpu: 3150.996 ms  88.7GFlops
 ==================================================
fp32_t  1024 x 1024  gpu:   0.988 (call:   1.006) avx:    0.192 ms   3.2GFlops
fp32_t  1024 x 1024  gpu:   1.012 (call:   1.026) avx:    0.159 ms   3.1GFlops
fp32_t  4096 x 4096  gpu:   6.823 (call:   6.843) avx:    3.083 ms   7.4GFlops
fp32_t  4096 x 16384 gpu:  37.561 (call:  37.582) avx:   11.591 ms   5.4GFlops
fp32_t 16384 x 65536 gpu:  66.903 (call:  66.925) avx:  195.364 ms  48.1GFlops
 --------------------------------------------------
fp32_t 30720 x 61440 gpu:  49.191 (call:  49.217) avx:  393.702 ms 115.1GFlops
 ==================================================
 *** Intel(R) UHD Graphics ***
fp16_t  1024 x 1024  gpu:   2.782 (call:   3.014) cpu:    1.679 ms   1.1GFlops
fp16_t  1024 x 1024  gpu:   2.788 (call:   2.985) cpu:    1.640 ms   1.1GFlops
fp16_t  4096 x 4096  gpu:  19.435 (call:  19.650) cpu:   27.112 ms   2.6GFlops
fp16_t  4096 x 16384 gpu:  78.662 (call:  80.059) cpu:  110.753 ms   2.6GFlops
fp16_t 32768 x 2048  gpu:  13.305 (call:  13.554) cpu:  105.328 ms  15.1GFlops
fp32_t  1024 x 1024  gpu:   1.367 (call:   1.536) avx:    0.192 ms   2.3GFlops
fp32_t  1024 x 1024  gpu:   1.376 (call:   1.578) avx:    0.207 ms   2.3GFlops
fp32_t  4096 x 4096  gpu:  14.073 (call:  15.146) avx:    2.969 ms   3.6GFlops
fp32_t  4096 x 16384 gpu:  55.773 (call:  56.055) avx:   10.915 ms   3.6GFlops
fp32_t 32768 x 2048  gpu:  13.616 (call:  14.351) avx:   11.575 ms  14.8GFlops
 *** NVIDIA GeForce RTX 3080 Laptop GPU ***
fp16_t  1024 x 1024  gpu:   7.199 cpu:    1.833 ms   0.4GFlops
fp16_t  1024 x 1024  gpu:   6.534 cpu:    1.636 ms   0.5GFlops
fp16_t  4096 x 4096  gpu:  45.165 cpu:   26.211 ms   1.1GFlops
fp16_t  4096 x 16384 gpu:  48.383 cpu:  110.253 ms   4.2GFlops
fp16_t 16384 x 65536 gpu:  83.338 cpu: 1725.135 ms  38.7GFlops
 --------------------------------------------------
fp16_t 30720 x 61440 gpu:  64.037 cpu: 3144.595 ms  88.4GFlops
 ==================================================
fp32_t  1024 x 1024  gpu:   0.996 avx:    0.171 ms   3.2GFlops
fp32_t  1024 x 1024  gpu:   0.995 avx:    0.207 ms   3.2GFlops
fp32_t  4096 x 4096  gpu:   6.842 avx:    3.171 ms   7.4GFlops
fp32_t  4096 x 16384 gpu:  37.605 avx:   11.858 ms   5.4GFlops
fp32_t 16384 x 65536 gpu:  66.900 avx:  200.672 ms  48.2GFlops
 --------------------------------------------------
fp32_t 30720 x 61440 gpu:  49.335 avx:  331.514 ms 114.8GFlops
 ==================================================
 *** Intel(R) UHD Graphics ***
fp16_t  1024 x 1024  gpu:   2.863 cpu:    1.624 ms   1.1GFlops
fp16_t  1024 x 1024  gpu:   2.869 cpu:    1.650 ms   1.1GFlops
fp16_t  4096 x 4096  gpu:  21.223 cpu:   27.250 ms   2.4GFlops
fp16_t  4096 x 16384 gpu:  78.921 cpu:  106.590 ms   2.6GFlops
fp16_t 32768 x 2048  gpu:  13.427 cpu:  108.990 ms  15.0GFlops
fp32_t  1024 x 1024  gpu:   1.571 avx:    0.145 ms   2.0GFlops
fp32_t  1024 x 1024  gpu:   1.657 avx:    0.162 ms   1.9GFlops
fp32_t  4096 x 4096  gpu:  15.538 avx:    2.958 ms   3.2GFlops
fp32_t  4096 x 16384 gpu:  57.393 avx:   11.624 ms   3.5GFlops
fp32_t 32768 x 2048  gpu:  13.769 avx:   13.695 ms  14.6GFlops

#endif