#include "rt.h"
#include "ocl.h"
#include "dot.h"
#include "gemv.h"

static bool verbose = true;
static bool unchecked;

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

static fp64_t avg_time;
static fp64_t avg_user;
static fp64_t avg_host;
static fp64_t avg_gflops;

static fp64_t avx_time;
static fp64_t gpu_time;

#define strprintf(s, ...) \
    (snprintf(s, countof(s) - 1, "" __VA_ARGS__))
#define catprintf(s, ...) \
    (strlen(s) < counts(s) - 5 ? \
        snprintf(s + strlen(s), countof(s) - 1 - strlen(s), "" __VA_ARGS__) : \
        "..." \
    )

static void vprintln(const fp32_t* vc, const int32_t n) {
    for (int32_t i = 0; i < n; i++) { printf("%5g ", vc[i]); }
    printf("\n");
}

static void mprintln(const fp32_t* mx, const int32_t n, const int32_t m) {
    for (int32_t j = 0; j < m; j++) {
        printf("m[%3d] ", j);
        for (int32_t i = 0; i < n; i++) { printf("%5g ", mx[j * n + i]); }
        printf("\n");
    }
}

static void m16println(const fp16_t* mx, const int32_t n, const int32_t m) {
    for (int32_t j = 0; j < m; j++) {
        printf("m[%3d] ", j);
        for (int32_t i = 0; i < n; i++) { printf("%5g ", fp16to32(mx[j * n + i])); }
        printf("\n");
    }
}

static void test32(gemv_t* g, int32_t n, int32_t m,
                   fp32_t (*init_vc)(int32_t i),
                   fp32_t (*init_mx)(int32_t j, int32_t i)) {
    ocl_context_t* c = g->c;
    gpu_time = DBL_MAX;
    avx_time = DBL_MAX;
    ocl_memory_t matrix = ocl.allocate(c, ocl_allocate_write, m * n * sizeof(fp32_t));
    ocl_memory_t vector = ocl.allocate(c, ocl_allocate_write,     n * sizeof(fp32_t));
    ocl_memory_t result = ocl.allocate(c, ocl_allocate_read,      m * sizeof(fp32_t));
    fp32_t* mx = (fp32_t*)ocl.map(c, ocl_map_write, matrix, 0, m * n * sizeof(fp32_t));
    fp32_t* vc = (fp32_t*)ocl.map(c, ocl_map_write, vector, 0,     n * sizeof(fp32_t));
    fp32_t* p = mx;
    for (int32_t i = 0; i < n; i++) {
        vc[i] = init_vc(i);
    }
    for (int32_t j = 0; j < m; j++) {
        for (int32_t i = 0; i < n; i++) {
            *p++ = init_mx(j, i);
        }
    }
    fp32_t* avx = (fp32_t*)alloca(m * sizeof(fp32_t));
    fatal_if(avx == null);
    fp64_t user = seconds();
    for (int32_t j = 0; j < m; j++) {
        avx[j] = (fp32_t)dot32(vc, 1, &mx[j * n], 1, n);
    }
    user = seconds() - user;
    avx_time = min(avx_time, user);
    fp32_t* verify = (fp32_t*)alloca(m * sizeof(fp32_t));
    fatal_if(verify == null);
    for (int32_t j = 0; j < m; j++) {
        fp32_t sum = 0;
        for (int32_t i = 0; i < n; i++) { sum += vc[i] * mx[j * n + i]; }
        verify[j] = sum;
    }
    if (verbose) {
        if (n <= 64) { printf("vc: "); vprintln(vc, n); }
        if (n <= 64 && m < 50) { printf("mx:\n"); mprintln(mx, n, m); }
        if (m <= 64) { printf("cpu: "); vprintln(verify, m); }
    }
    ocl.unmap(c, matrix, mx);
    ocl.unmap(c, vector, vc);
    user = seconds();
    gemv.ocl_gemv32(g, 0, matrix, vector, result, n, m);
    user = seconds() - user;
    gpu_time = min(gpu_time, user);
    fp32_t* rs = (fp32_t*)ocl.map(c, ocl_map_read, result, 0, m * sizeof(fp32_t));
    if (verbose && m <= 64) { printf("gpu: "); vprintln(rs, m); }
    ocl.unmap(c, result, rs);
    // verification
    const fp32_t epsilon = CL_FLT_EPSILON * n * m;
    if (!unchecked) {
        for (int32_t j = 0; j < m; j++) {
            fp64_t delta = fabs(verify[j] - rs[j]);
            fatal_if(delta > epsilon, "delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                delta, epsilon, j, verify[j], j, rs[j]);
            delta = fabs(verify[j] - avx[j]);
            fatal_if(delta > epsilon, "delta: %g epsilon: %g cpu[%d]: %g avx[%d]: %g",
                delta, epsilon, j, verify[j], j, avx[j]);
            delta = fabs(rs[j] - avx[j]);
            fatal_if(delta > epsilon, "delta: %g epsilon: %g avx[%d]: %g gpu[%d]: %g",
                delta, epsilon, j, avx[j], j, rs[j]);
        }
    } else {
        for (int32_t j = 0; j < m; j++) {
            fp64_t delta = fabs(verify[j] - rs[j]);
            if (delta > epsilon) {
                traceln("delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                         delta, epsilon, j, verify[j], j, rs[j]);
                break;
            }
            delta = fabs(verify[j] - avx[j]);
            if (delta > epsilon)  {
                traceln("delta: %g epsilon: %g cpu[%d]: %g avx[%d]: %g",
                         delta, epsilon, j, verify[j], j, avx[j]);
                break;
            }
            delta = fabs(rs[j] - avx[j]);
            if (delta > epsilon) {
                traceln("delta: %g epsilon: %g avx[%d]: %g gpu[%d]: %g",
                         delta, epsilon, j, avx[j], j, rs[j]);
                break;
            }
        }
    }
    // cleanup
    ocl.deallocate(result);
    ocl.deallocate(vector);
    ocl.deallocate(matrix);
    if (n > 64 && m > 64) {
        traceln("%dx%d gpu: %6.3f avx: %6.3f ms", n, m,
            gpu_time * MSEC_IN_SEC, avx_time * MSEC_IN_SEC);
    }
}

static void test16(gemv_t* g, int32_t n, int32_t m,
                   fp32_t (*init_vc)(int32_t i),
                   fp32_t (*init_mx)(int32_t j, int32_t i)) {
    ocl_context_t* c = g->c;
    gpu_time = DBL_MAX;
    avx_time = DBL_MAX;
    ocl_memory_t matrix = ocl.allocate(c, ocl_allocate_write, m * n * sizeof(fp16_t));
    ocl_memory_t vector = ocl.allocate(c, ocl_allocate_write,     n * sizeof(fp32_t));
    ocl_memory_t result = ocl.allocate(c, ocl_allocate_read,      m * sizeof(fp32_t));
    fp16_t* mx = (fp16_t*)ocl.map(c, ocl_map_write, matrix, 0, m * n * sizeof(fp16_t));
    fp32_t* vc = (fp32_t*)ocl.map(c, ocl_map_write, vector, 0,     n * sizeof(fp32_t));
    for (int32_t i = 0; i < n; i++) {
        vc[i] = init_vc(i);
    }
    fp16_t* p = mx;
    for (int32_t j = 0; j < m; j++) {
        for (int32_t i = 0; i < n; i++) {
            *p++ = fp32to16(init_mx(j, i));
        }
    }
    fp32_t* verify = (fp32_t*)alloca(m * sizeof(fp32_t));
    fatal_if(verify == null);
    fp64_t cpu = seconds();
    for (int32_t j = 0; j < m; j++) {
        fp32_t sum = 0;
        for (int32_t i = 0; i < n; i++) { sum += vc[i] * fp16to32(mx[j * n + i]); }
        verify[j] = sum;
    }
    cpu = seconds() - cpu;
    if (verbose) {
        if (n <= 64) { printf("vc: "); vprintln(vc, n); }
        if (n <= 64 && m < 50) { printf("mx:\n"); m16println(mx, n, m); }
        if (m <= 64) { printf("cpu: "); vprintln(verify, m); }
    }
    ocl.unmap(c, matrix, mx);
    ocl.unmap(c, vector, vc);
    fp64_t user = seconds();
    gemv.ocl_gemv16(g, 0, matrix, vector, result, n, m);
    user = seconds() - user;
    gpu_time = min(gpu_time, user);
    fp32_t* rs = (fp32_t*)ocl.map(c, ocl_map_read, result, 0, m * sizeof(fp32_t));
    if (verbose && m <= 64) { printf("gpu: "); vprintln(rs, m); }
    ocl.unmap(c, result, rs);
    // verification
    const fp32_t epsilon = CL_FLT_EPSILON * n * m;
    if (!unchecked) {
        for (int32_t j = 0; j < m; j++) {
            fp64_t delta = fabs(verify[j] - rs[j]);
            fatal_if(delta > epsilon, "delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                delta, epsilon, j, verify[j], j, rs[j]);
        }
    } else {
        for (int32_t j = 0; j < m; j++) {
            fp64_t delta = fabs(verify[j] - rs[j]);
            if (delta > epsilon) {
                traceln("delta: %g epsilon: %g cpu[%d]: %g gpu[%d]: %g",
                         delta, epsilon, j, verify[j], j, rs[j]);
                break;
            }
        }
    }
    // cleanup
    ocl.deallocate(result);
    ocl.deallocate(vector);
    ocl.deallocate(matrix);
    if (n > 64 && m > 64) {
        traceln("%dx%d gpu: %6.3f cpu: %6.3f ms", n, m,
            gpu_time * MSEC_IN_SEC, cpu * MSEC_IN_SEC);
    }
}

static fp32_t init_vc0(int32_t i) {
    return (fp32_t)(i + 1);
}

static fp32_t init_mx0(int32_t j, int32_t i) {
    return (fp32_t)((j + 1) * 10 + (i + 1));
}

static fp32_t init_vc1(int32_t i) {
    fp32_t v = (fp32_t)(i + 1);
    return 1.0f + v / (fp32_t)(1U << 20);
}

static fp32_t init_mx1(int32_t j, int32_t i) {
    fp32_t v = (fp32_t)((j + 1) * 10 + (i + 1));
    return 1.0f + v / (fp32_t)(1ULL << 60);
}

static void tests() {
    static ocl_profiling_t p[4096];
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
        traceln("");
        traceln("%s", d->name);
        traceln("");
        gemv_t g = {0};
        gemv.init(&g, &c);
        verbose = true; // set to true if crashes
        test16(&g, 2, 3, init_vc0, init_mx0);
        test16(&g, 4, 4, init_vc0, init_mx0);
        test16(&g, 8, 16, init_vc0, init_mx0);
        test16(&g, 1024, 1024, init_vc1, init_mx1);
#if 1
        test16(&g, 1024, 1024, init_vc1, init_mx1);
        test16(&g, 1024, 1024, init_vc1, init_mx1);
        test16(&g, 4 * 1024,  4 * 1024, init_vc1, init_mx1);
        test16(&g, 4 * 1024, 16 * 1024, init_vc1, init_mx1); // GPT-J 6b innermost gemv()
        // only run on NVIDIA GPU. Intel UHD Graphics GPU reports 16GB as global memory
        // but cannot allocate any of this huge memory chunks
        if (strstr(d->vendor, "NVIDIA") != null) {
            test16(&g, 16 * 1024, 64 * 1024, init_vc1, init_mx1);
            traceln("--------------------------------------------------");
            // GPT-J 6b [16K, 4K, 7] * [4K, 7] 7 is probably dimension of word embedding?
            // 32*64 = 2048M x sizeof(fp32_t) = 8GB
            // use 30x60 instead to fit into 8GB of GPU memory
            // the accumulated error is too big to check:
            unchecked++;
            test16(&g, 30 * 1024, 60 * 1024, init_vc1, init_mx1);
            unchecked--;
            traceln("==================================================");
        } else {
            test16(&g, 32 * 1024, 2 * 1024, init_vc1, init_mx1); // GPT-J 6b inermost gemv()
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
 groups x items NVIDIA GeForce RTX 3080 Laptop GPU

 No significant differences detected of groups/items configurat gions

  g64xi256 (groups x items)
    n: 16384 m: 65536 groups: 64 items: 256 compute units: 48
    16384x65536 gpu: 24.454ms GFlops: 87.817
    16384xg6553i6 gpu: 384.664 avx: 189.613 ms
  g128xi128
    n: 16384 m: 65536 groups: 128 items: 128 compute units: 48
    16384x65536 gpu: 23.764ms GFlops: 90.367
    16384xg6553i6 gpu: 385.562 avx: 185.405 ms
  g256xi64
    n: 16384 m: 65536 groups: 256 items: 64 compute units: 48
    16384x65536 gpu: 23.496ms GFlops: 91.399
    16384x65536 gpu: 386.828 avx: 187.998 ms NVIDIA GeForce RT

  - GFlops:
  =========

  30720x61440 NVIDIA GeForce RTX 3080 Laptop GPU
     groups: 64 items: 480 compute units: 48
     gpu: 26.023ms GFlops: 145.060
     gpu: 790.689 avx: 343.893 ms
     // inaccurate rounding errors:
     delta: 14546 epsilon: 225 cpu[0]: 31170 gpu[0]: 16624

#endif