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
                println("%s %5d x %-5d gpu: %9.3f (call: %9.3f) avx: %9.3f "
                    "ms %5.1fGFlops",
                    ocl_fpp_names[fpp], n, m,
                    gpu_time * MSEC_IN_SEC, ocl_time * MSEC_IN_SEC,
                    avx_time * MSEC_IN_SEC, gpu_gfps);
            } else {
                gpu_gfps = 3.0 * m * n / (ocl_time * NSEC_IN_SEC);
                println("%s %5d x %-5d gpu: %9.3f avx: %9.3f ms %5.1fGFlops",
                    ocl_fpp_names[fpp], n, m,
                    ocl_time * MSEC_IN_SEC,
                    avx_time * MSEC_IN_SEC, gpu_gfps);
            }
        } else {
            if (gpu_time < DBL_MAX) {
                println("%s %5d x %-5d gpu: %9.3f (call: %9.3f) cpu: %9.3f ms "
                    "%5.1fGFlops",
                    ocl_fpp_names[fpp], n, m,
                    gpu_time * MSEC_IN_SEC, ocl_time * MSEC_IN_SEC,
                    cpu_time * MSEC_IN_SEC, gpu_gfps);
            } else {
                gpu_gfps = 3.0 * m * n / (ocl_time * NSEC_IN_SEC);
                println("%s %5d x %-5d gpu: %9.3f cpu: %9.3f ms %5.1fGFlops",
                    ocl_fpp_names[fpp], n, m,
                    ocl_time * MSEC_IN_SEC,
                    cpu_time * MSEC_IN_SEC, gpu_gfps);
            }
        }
    }
}

static void init_mx_vc(int fpp,
                 void* mx, void* vc,
                 int32_t n, int32_t m,
                 fp64_t (*init_mx)(int32_t j, int32_t i, int32_t n),
                 fp64_t (*init_vc)(int32_t i)) {
    const size_t meb = ocl_fpp_bytes[fpp]; // matrix element bytes
    for (int32_t i = 0; i < n; i++) {
        switch (fpp) {
            case ocl_bfp16:
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
                case ocl_bfp16: *((bf16_t*)p) = bf32to16((fp32_t)init_mx(j, i, n)); break;
                case ocl_fpp16: *((fp16_t*)p) = fp32to16((fp32_t)init_mx(j, i, n)); break;
                case ocl_fpp32: *((fp32_t*)p) = (fp32_t)init_mx(j, i, n); break;
                case ocl_fpp64: *((fp64_t*)p) = init_mx(j, i, n); break;
                default: fatal_if("fpp?", "fpp: %d", fpp);
            }
            p += meb;
        }
    }
}

static void test_avx(int fpp, void* mx, void* vc, void* avx,
        int32_t n, int32_t m) {
    fp64_t user = seconds();
    switch (fpp) {
        case ocl_fpp16:
            for (int32_t j = 0; j < m; j++) {
                fp16_t* row = (fp16_t*)mx + j * n;
                ((fp32_t*)avx)[j] = (fp32_t)dot.fp32x16((fp32_t*)vc, 1, row, 1, n);
            }
            break;
        case ocl_bfp16:
            for (int32_t j = 0; j < m; j++) {
                bf16_t* row = (bf16_t*)mx + j * n;
                ((fp32_t*)avx)[j] = (fp32_t)dot.bf32x16((fp32_t*)vc, 1, row, 1, n);
            }
            break;
        case ocl_fpp32:
            for (int32_t j = 0; j < m; j++) {
                fp32_t* row = (fp32_t*)mx + j * n;
                ((fp32_t*)avx)[j] = (fp32_t)dot.fp32((fp32_t*)vc, 1, row, 1, n);
            }
            break;
        case ocl_fpp64:
            for (int64_t j = 0; j < m; j++) {
                fp64_t* row = (fp64_t*)mx + j * n;
                ((fp64_t*)avx)[j] = dot.fp64((fp64_t*)vc, 1, row, 1, n);
            }
            break;
        default:
            fatal_if("fpp?", "fpp: %d", fpp);
    }
    user = seconds() - user;
    avx_time = min(avx_time, user);
}

static void test_cpu(int fpp, void* mx, void* vc, void* cpu,
        int32_t n, int32_t m) {
    fp64_t user = seconds();
    switch (fpp) {
        case ocl_bfp16:
            for (int32_t j = 0; j < m; j++) {
                bf16_t* row = (bf16_t*)mx + j * n;
                fp32_t s = 0;
                for (int32_t i = 0; i < n; i++) {
                    s += ((fp32_t*)vc)[i] * bf16to32(row[i]);
                }
                ((fp32_t*)cpu)[j] = s;
            }
            break;
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
}

static void dump_mx_vc(int fpp, void* mx, void* vc, void* avx, void* cpu,
        int32_t n, int32_t m) {
    switch (fpp) {
        case ocl_fpp16:
            printf("mx:\n"); m16println((fp16_t*)mx, n, m);
            printf("vc : "); v32println((fp32_t*)vc, n);
            printf("cpu: "); v32println((fp32_t*)cpu, m);
            printf("avx: "); v32println((fp32_t*)avx, m);
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

static void dump_result(int fpp, void* rs, int32_t m) {
    printf("gpu: ");
    switch (fpp) {
        case ocl_fpp16:
        case ocl_fpp32: v32println((fp32_t*)rs, m); break;
        case ocl_fpp64: v64println((fp64_t*)rs, m); break;
        default: fatal_if("fpp?", "fpp: %d", fpp);
    }
}

static void verify(int fpp, void* avx, void* cpu, void* rs, int32_t n, int32_t m) {
    const fp64_t epsilon = CL_FLT_EPSILON * n * m;
    for (int32_t j = 0; j < m; j++) {
        fp64_t delta_cpu_gpu = 0;
        fp64_t delta_avx_gpu = 0;
        fp64_t delta_avx_cpu = 0;
        switch (fpp) {
            case ocl_bfp16:
            case ocl_fpp16:
            case ocl_fpp32:
                delta_cpu_gpu = fabs(((fp32_t*)cpu)[j] - ((fp32_t*)rs)[j]);
                delta_avx_cpu = fabs(((fp32_t*)cpu)[j] - ((fp32_t*)avx)[j]);
                delta_avx_gpu = fabs(((fp32_t*)rs)[j] - ((fp32_t*)avx)[j]);
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
                case ocl_bfp16:
                case ocl_fpp16:
                case ocl_fpp32:
                    fatal_if(delta_cpu_gpu > epsilon, "%d x %d delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                        n, m, delta_cpu_gpu, epsilon, j, ((fp32_t*)cpu)[j], j, ((fp32_t*)rs)[j]);
                    fatal_if(delta_avx_cpu > epsilon, "%d x %d delta: %g epsilon: %g cpu[%d]: %g avx[%d]: %g",
                        n, m, delta_avx_cpu, epsilon, j, ((fp32_t*)cpu)[j], j, ((fp32_t*)avx)[j]);
                    fatal_if(delta_avx_gpu > epsilon, "%d x %d delta: %g epsilon: %g avx[%d]: %g gpu[%d]: %g",
                        n, m, delta_avx_gpu, epsilon, j, ((fp32_t*)avx)[j], j, ((fp32_t*)rs)[j]);
                    break;
                case ocl_fpp64:
                    fatal_if(delta_cpu_gpu > epsilon, "%d x %d delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                        n, m, delta_cpu_gpu, epsilon, j, ((fp64_t*)cpu)[j], j, ((fp64_t*)rs)[j]);
                    fatal_if(delta_avx_cpu > epsilon, "%d x %d delta: %g epsilon: %g cpu[%d]: %g avx[%d]: %g",
                        n, m, delta_avx_cpu, epsilon, j, ((fp64_t*)cpu)[j], j, ((fp64_t*)avx)[j]);
                    fatal_if(delta_avx_gpu > epsilon, "%d x %d delta: %g epsilon: %g avx[%d]: %g gpu[%d]: %g",
                        n, m, delta_avx_gpu, epsilon, j, ((fp64_t*)avx)[j], j, ((fp64_t*)rs)[j]);
                    break;
                default:
                    fatal_if("fpp?", "fpp: %d", fpp);
            }
        }
    }
}

static void run(gemv_t* g, int fpp,
        ocl_memory_t matrix,
        ocl_memory_t vector,
        ocl_memory_t result,
        int32_t n, int32_t m) {
    assert(best_of >= 1);
    for (int repeat = 0; repeat < best_of; repeat++) {
        fp64_t user = seconds();
        gemv.gemv(g, fpp, 0, matrix, vector, result, n, m);
        user = seconds() - user;
        ocl_time = min(ocl_time, user);
        if (ocl.is_profiling(g->c)) {
            ocl_profiling_t* p = &g->c->ov->profiling[0];
            assert(p->e == null, "did anyone call ocl.profile?");
            gpu_time = min(gpu_time, p->time);
            gpu_gfps = max(gpu_gfps, p->gflops);
        }
    }
}

ocl_memory_t alloc(ocl_context_t* c, int access, size_t bytes) {
    // ocl.alloc() can return not null but map() will actually fail
    ocl_memory_t m = ocl.alloc(c, access, bytes);
    if (m != null) {
        void* p = ocl.map(c, ocl.access_to_map(access), m, 0, bytes);
        if (p == null) {
            ocl.deallocate(m);
            m = null;
        } else {
            ocl.unmap(c, m, p);
        }
    }
    return m;
}

static void test(gemv_t* g, int fpp,
                 int32_t n, int32_t m,
                 fp64_t (*init_mx)(int32_t j, int32_t i, int32_t n),
                 fp64_t (*init_vc)(int32_t i)) {
    ocl_context_t* c = g->c;
//  println("%d x %d fp%d_t", n, m, ocl_fpp_bytes[fpp] * 8);
    ocl_time = DBL_MAX;
    gpu_time = DBL_MAX;
    avx_time = DBL_MAX;
    cpu_time = DBL_MAX;
    gpu_gfps = 0;
    const size_t meb = ocl_fpp_bytes[fpp]; // matrix element bytes
    const size_t veb = fpp == ocl_fpp16 ? 4 : 8; // vector bytes
    enum { write_only = CL_MEM_WRITE_ONLY|CL_MEM_HOST_WRITE_ONLY };
    enum { read_only  = CL_MEM_READ_ONLY|CL_MEM_HOST_READ_ONLY };
    ocl_memory_t matrix = alloc(c, write_only, (size_t)m * n * meb);
    ocl_memory_t vector = alloc(c, write_only, (size_t)n * veb);
    ocl_memory_t result = alloc(c, read_only,  (size_t)m * veb);
    if (matrix != null && vector != null && result != null) {
        void* mx = ocl.map(c, CL_MAP_WRITE, matrix, 0, (size_t)m * n * meb);
        void* vc = ocl.map(c, CL_MAP_WRITE, vector, 0, n * veb);
        init_mx_vc(fpp, mx, vc, n, m, init_mx, init_vc);
        void* avx = alloca(m * veb);
        fatal_if(avx == null);
        test_avx(fpp, mx, vc, avx, n, m);
        void* cpu = alloca(m * veb);
        fatal_if(cpu == null);
        test_cpu(fpp, mx, vc, cpu, n, m);
        if (verbose && n <= 64 && m <= 64) {
            dump_mx_vc(fpp, mx, vc, avx, cpu, n, m);
        }
        ocl.unmap(c, vector, vc);
        ocl.unmap(c, matrix, mx);
        fp64_t user = seconds();
//      ocl.migrate(c, matrix); // migrate is a "hint" to move memory
//      ocl.migrate(c, vector); // to the GPU dedicated fater access region
//      ocl.migrate_undefined(c, result);
        user = seconds() - user;
    //  println("migrate: %.6f ms", user * 1000.0);
        run(g, fpp, matrix, vector, result, n, m);
        void* rs = ocl.map(c, CL_MAP_READ, result, 0, m * veb);
        if (verbose && m <= 64) { dump_result(fpp, rs, m); }
        verify(fpp, avx, cpu, rs, n, m);
        // cleanup
        ocl.unmap(c, result, rs);
    }
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

static void permutations(gemv_t* g) {
#ifndef PERMUTATIONS_DEBUG_SINGLE_CASE
    // all 1..33 x 1..33 permutations of all precisions
    println("33x33...");
    verbose = false; // set to true if crashes
    for (int fpp = ocl_fpp_first; fpp <= ocl_fpp_last; fpp++) {
        if (ocl.has_fpp(g->c, fpp)) {
            for (int n = 1; n <= 33; n++) {
                for (int m = 1; m <= 33; m++) {
//                  if (_isatty(_fileno(stdout))) {
//                      printf("%2dx%-2d: %s\r", n, m, ocl_fpp_names[fpp]);
//                  }
                    test(g, fpp, n, m, init_mx0, init_vc0);
                }
            }
        }
    }
#else
    // debug and trace a single failing case from permutaions above
    verbose = true;
    test(g, ocl_fpp16, 4, 1, init_mx0, init_vc0);
#endif
}

static void tests(bool profile) {
    for (int i = 0; i < ocl.count; i++) {
//      ocl.dump(i);
        const ocl_device_t* d = &ocl.devices[i];
        ocl_profiling_t profiling[1] = {0}; // [1] single kernel
        ocl_override_t ov = {
            .profiling = profiling,
            .max_profiling_count = countof(profiling),
            .profiling_count = 0
        };
        ocl_context_t c = ocl.open(i, profile ? &ov : null);
        println("*** %s : %.1fGB *** %s", d->name,
            d->global_memory / (double)GB, profile ? "PROFILING" : "");
        gemv_t g = {0};
        gemv.init(&g, &c);
        permutations(&g);
        // large matrix/vectors performance tests
        for (int fpp = ocl_fpp_first; fpp <= ocl_fpp_last; fpp++) {
            if (ocl.has_fpp(&c, fpp)) {
                struct { int32_t n; int32_t m; } tests[] = {
                    {     1024,      1024},
                    { 4 * 1024,  4 * 1024},
                    { 4 * 1024, 16 * 1024}, // GPT-J 6b innermost gemv()
                    {16 * 1024, 48 * 1024}, // GPT-J 6b innermost x 8 layers
                    {16 * 1024, 64 * 1024},
                    {64 * 1024, 16 * 1024},
                };
                for (int k = 0; k < countof(tests); k++) {
                    const int64_t n = tests[k].n;
                    const int64_t m = tests[k].m;
                    const int64_t bytes = n * m * ocl_fpp_bytes[fpp];
                    // 128MB is reserved inside most of modern GPUs
                    if (bytes < d->global_memory) {
#if 0
                        double gb = bytes / (double)GB;
                        double dgb = d->global_memory / (double)GB;
                        println("%d x %d %.3fGB of %.3fGB", n, m, gb, dgb);
                        println("press any key to continue");
                        while (_kbhit() == 0) { sleep(1.0 / 64); }
                        getch();
#endif
                        test(&g, fpp, tests[k].n, tests[k].m, init_mx1, init_vc1);
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
    if (dot.test != null) { dot.test(); }
//  if (argc > 0) exit(0);
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