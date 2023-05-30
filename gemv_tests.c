#include "rt.h"
#include "ocl.h"
#include "dot.h"
#include "gemv.h"

static int  best_of = 3;
static bool verbose = true;
static bool unchecked;

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

static fp64_t cpu_time;
static fp64_t avx_time;
static fp64_t ocl_time; // total time for gemv() call
static fp64_t gpu_time; // gpu time
static fp64_t gpu_gfps; // gpu GFlops

#define strprintf(s, ...) \
    (snprintf(s, countof(s) - 1, "" __VA_ARGS__))
#define catprintf(s, ...) \
    (strlen(s) < counts(s) - 5 ? \
        snprintf(s + strlen(s), countof(s) - 1 - strlen(s), "" __VA_ARGS__) : \
        "..." \
    )

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

static void test(gemv_t* g, int fpp,
                 int32_t n, int32_t m,
                 fp64_t (*init_vc)(int32_t i),
                 fp64_t (*init_mx)(int32_t j, int32_t i, int32_t n)) {
    ocl_context_t* c = g->c;
//  traceln("%d x %d fp%d_t", n, m, ocl_fpp_bytes[fpp] * 8);
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
//  traceln("migrate: %.6f ms", user * 1000.0);
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
 // if (ocl.is_profiling(g->c)) {
 //     ocl_profiling_t* pf = &g->c->ov->profiling[0];
 //     if (n > 64 && m > 64) {
 //         traceln("%dx%d gpu: %6.3fms %4.1fGFlops",
 //                 n, m, pf->time * MSEC_IN_SEC, pf->gflops);
 //     }
 // }
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
    if (n > 64 && m > 64) {
        if (avx_time < DBL_MAX) {
            traceln("fp%d_t %5d x %-5d gpu: %6.3f (call: %7.3f) avx: %8.3f ms %5.1fGFlops",
                ocl_fpp_bytes[fpp] * 8, n, m,
                gpu_time * MSEC_IN_SEC, ocl_time * MSEC_IN_SEC,
                avx_time * MSEC_IN_SEC, gpu_gfps);
        } else {
            traceln("fp%d_t %5d x %-5d gpu: %6.3f (call: %7.3f) cpu: %8.3f ms %5.1fGFlops",
                ocl_fpp_bytes[fpp] * 8, n, m,
                gpu_time * MSEC_IN_SEC, ocl_time * MSEC_IN_SEC,
                cpu_time * MSEC_IN_SEC, gpu_gfps);
        }
    }
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

static void tests() {
    static ocl_profiling_t p[1];
    // profiling measurement:
    for (int i = 0; i < ocl.count; i++) {
//      ocl.dump(i);
        const ocl_device_t* d = &ocl.devices[i];
        ocl_override_t ov = {
            .profiling = p,
            .max_profiling_count = countof(p),
            .profiling_count = 0
        };
        ocl_context_t c = ocl.open(i, &ov);
        traceln("*** %s ***", d->name);
        gemv_t g = {0};
        gemv.init(&g, &c);
//      test(&g, ocl_fpp32, 4 * 1024, 16 * 1024, init_vc1, init_mx1); // GPT-J 6b innermost gemv()
#if 0
        // used to explorer and trace failing case from permutaions below
        verbose = true;
        test(&g, ocl_fpp32, 33, 1, init_vc0, init_mx0);
#else
        // all 1..64 x 1..64 permutations
        verbose = false; // set to true if crashes
        for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
            for (int n = 1; n <= 64; n++) {
                for (int m = 1; m <= 64; m++) {
                    if (fpp != ocl_fpp64 || d->double_fp_config != 0) {
                        test(&g, fpp, n, m, init_vc0, init_mx0);
                    }
                }
            }
        }
#endif

#if 1   // large matrix/vectors performance tests
        for (int fpp = ocl_fpp16; fpp <= ocl_fpp32; fpp++) {
            test(&g, fpp, 1024, 1024, init_vc1, init_mx1);
            test(&g, fpp, 1024, 1024, init_vc1, init_mx1);
            test(&g, fpp, 4 * 1024,  4 * 1024, init_vc1, init_mx1);
            test(&g, fpp, 4 * 1024, 16 * 1024, init_vc1, init_mx1); // GPT-J 6b innermost gemv()
            // only run on NVIDIA GPU. Intel UHD Graphics GPU reports 16GB as global memory
            // but cannot allocate any of this huge memory chunks
            if (strstr(d->vendor, "NVIDIA") != null && fpp < ocl_fpp64) {
                test(&g, fpp, 16 * 1024, 64 * 1024, init_vc1, init_mx1);
                traceln("--------------------------------------------------");
                // GPT-J 6b [16K, 4K, 7] * [4K, 7] 7 is probably dimension of word embedding?
                // 32*64 = 2048M x sizeof(fp32_t) = 8GB
                // use 30x60 instead to fit into 8GB of GPU memory
                // the accumulated error is too big to check:
                unchecked++;
                test(&g, fpp, 30 * 1024, 60 * 1024, init_vc1, init_mx1);
                unchecked--;
                traceln("==================================================");
            } else {
                test(&g, fpp, 32 * 1024, 2 * 1024, init_vc1, init_mx1); // GPT-J 6b innermost gemv()
            }
        }
#endif
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
            traceln("compile source binary\nNot enough arguments.");
        }
    } else {
        tests();
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


#endif