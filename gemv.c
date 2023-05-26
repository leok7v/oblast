#include "rt.h"
#include "ocl.h"
#include "dot.h"

static ocl_kernel_t gemv_kernel[3]; // TODO: move to gemv_context_t
static bool verbose = true;

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

static void gemv(ocl_context_t* c, int fpp,
        ocl_memory_t mx, ocl_memory_t vc,
        uint32_t n, uint32_t m, ocl_memory_t rs) {
    fatal_if(n == 0 || (n & (n - 1)) != 0 ||
             m == 0 || (m & (m - 1)) != 0,
             "n: %d n: %d both must be power of 2", n, m);
    if (ocl.is_profiling(c)) { c->ov->profiling_count = 0; }
    const int64_t compute_units   = ocl.devices[c->ix].compute_units;
    const int64_t local_bytes     = ocl.devices[c->ix].local_memory;
    const int64_t max_groups      = ocl.devices[c->ix].max_groups;
    const int64_t max_local_items = local_bytes / ocl_fpp_bytes[fpp];
    const int64_t max_items       =
        min(ocl.devices[c->ix].max_items[0], max_local_items);
    // ancient NVIDIA sample link below (*) assumed 2 work-groups
    // per compute unit
    int64_t cu_x_2 = compute_units * 2;
    int64_t items = n;
    int64_t groups = 1;
    while (items > 64 && groups <= cu_x_2 / 2 && (groups < items / 2 || items > max_items)) {
        items /= 2; groups *= 2;
    }
    if (items > max_items) { // same but more groups per compute unit
        while (items > 64 && groups <= max_groups / 2 && (groups < items / 2 || items > max_items)) {
            items /= 2; groups *= 2;
        }
    }
    if (n == 16 * 1024 && m == 64 * 1024) {
        // forced experiments:
        groups = 256; items = 64;
    }
    traceln("n: %d m: %d groups: %lld items: %lld compute units: %lld",
        n, m, groups, items, compute_units);
    assert(groups <= max_groups && items <= max_items && groups * items == n);
    ocl_kernel_t k = gemv_kernel[fpp];
    ocl_arg_t a[] =
        {{&mx,  sizeof(ocl_memory_t)},
         {&vc,  sizeof(ocl_memory_t)},
         {null, items * sizeof(fp32_t)}, // wc
         {&n,   sizeof(int32_t)},
         {&m,   sizeof(int32_t)},
         {&rs,  sizeof(ocl_memory_t)}
    };
    fp64_t user = seconds();
    ocl_event_t done = ocl.enqueue_range_kernel(c, k, groups, items, countof(a), a);
    if (ocl.is_profiling(c)) { ocl.profile_add(c, done); }
    ocl.finish(c);
    user = seconds() - user;
    gpu_time = min(gpu_time, user);
    if (ocl.is_profiling(c)) {
        ocl_profiling_t* p = &c->ov->profiling[0];
        p[0].count  = m;      // kernel invocations
        p[0].fops   = n * 2U; // + and * fp32_t operation each
        p[0].i32ops = n * 2U; // + and * int32_t operation each
        ocl.release_event(p[0].e);
        ocl.profile(&p[0]);
        for (int i = 1; i < c->ov->profiling_count; i++) {
            p[i].count  = m;     // kernel invocations
            p[i].fops   = n * 2; // + and * fp32_t operation each
            p[i].i32ops = n * 2; // + and * int32_t operation each
            ocl.release_event(p[i].e);
            ocl.profile(&p[i]);
            p[0].time   += p[i].time;
            p[0].user   += p[i].user;
            p[0].gflops += p[i].gflops;
            p[0].i32ops += p[i].i64ops;
            p[0].i64ops += p[i].i64ops;
        }
        p->gflops /= c->ov->profiling_count;
        p->i32ops /= c->ov->profiling_count;
        p->i64ops /= c->ov->profiling_count;
        if (n > 64 && m > 64) {
            traceln("%dx%d gpu: %6.3fms GFlops: %6.3f",
                    n, m, p->time * MSEC_IN_SEC, p->gflops);
        }
        avg_time += p->time;
        avg_gflops += p->gflops;
    }
}

// (*) note: https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclMatVecMul/oclMatVecMul.cpp#L223

static void test(ocl_context_t* c, int32_t n, int32_t m,
                 fp32_t (*init_vc)(int32_t i),
                 fp32_t (*init_mx)(int32_t j, int32_t i),
                 const char* name) {
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
        if (n < 50) { vprintln(vc, n); }
        if (n < 50 && m < 50) { mprintln(mx, n, m); }
        if (m < 50) { vprintln(verify, m); }
    }
    ocl.unmap(c, matrix, mx);
    ocl.unmap(c, vector, vc);
    gemv(c, ocl_fpp32, matrix, vector, n, m, result);
    fp32_t* rs = (fp32_t*)ocl.map(c, ocl_map_read, result, 0, m * sizeof(fp32_t));
    if (m < 50) { vprintln(rs, m); }
    ocl.unmap(c, result, rs);
    // verification
    const fp32_t epsilon = CL_FLT_EPSILON * n * m;
    for (int32_t j = 0; j < m; j++) {
        fp64_t delta = fabs(verify[j] - rs[j]);
        fatal_if(delta > epsilon, "delta: %g epsilon: %g cpu[%d]: %g result[%d]: %g",
            delta, epsilon, j, verify[j], j, rs[j]);
    }
    for (int32_t j = 0; j < m; j++) {
        fp64_t delta = fabs(verify[j] - avx[j]);
        fatal_if(delta > epsilon, "delta: %g epsilon: %g avx[%d]: %g result[%d]: %g",
            delta, epsilon, j, avx[j], j, rs[j]);
    }
    // cleanup
    ocl.deallocate(result);
    ocl.deallocate(vector);
    ocl.deallocate(matrix);
    if (n > 64 && m > 64) {
        traceln("%dx%d gpu: %6.3f avx: %6.3f ms %s", n, m,
            gpu_time * MSEC_IN_SEC, avx_time * MSEC_IN_SEC,
            name);
    }
}

static const char* gemv_program_options(ocl_context_t* c, int fpp) {
    const ocl_device_t* d = &ocl.devices[c->ix];
    static char options[4096];
    char* p = options;
    #pragma push_macro("append")
    #define append(...) do {                                             \
        intptr_t k = options + countof(options) - p - 1;                 \
        fatal_if(k <= 0, "options[%d] overflow", (int)countof(options)); \
        p += snprintf(p, k, "" __VA_ARGS__);                             \
    } while (0)
    append("-D int16_t=short -D uint16_t=ushort ");
    append("-D int32_t=int   -D uint32_t=uint ");
    append("-D int64_t=long  -D uint64_t=ilong ");
    append("-D fp16_t=half -D fp32_t=float -D fp64_t=double ");
    append("-cl-std=CL%d.%d ", d->c_version_major, d->c_version_minor);
    static const char* type_t[] = {"half", "float", "double"};
    append("-D fp_t=%s -D vec4=%s4 -D fpp=%d ", type_t[fpp], type_t[fpp],
        ocl_fpp_bytes[fpp] * 8);
    #pragma pop_macro("append")
    *p = 0;
//  traceln("%s", options);
    return options;
}

static ocl_program_t gemv_compile(ocl_context_t* c, int fpp,
        const void* code, int bytes) {
    const char* opts = gemv_program_options(c, fpp);
    return ocl.compile_program(c, code, bytes, opts);
}

static void gemv_init(ocl_context_t* c) {
    ocl_device_t* d = &ocl.devices[c->ix];
    void* code = null;
    int64_t bytes64 = 0;
    int r = memmap_resource("gemv_cl", &code, &bytes64);
    fatal_if(r != 0 || code == null || bytes64 == 0, "is gemv.cl in gemv.rc?");
    fatal_if(bytes64 > INT_MAX, "blast.cl %lld bytes", bytes64);
    int bytes = (int)bytes64;
    const bool has_fp16 = (d->fp_config & ocl_fp16) != 0 && false; // xxx
    const bool has_fp64 =  d->double_fp_config != 0 && false; // xxx
    ocl_program_t p[3] = {
        has_fp16 ? gemv_compile(c, ocl_fpp16, code, bytes) : null,
                   gemv_compile(c, ocl_fpp32, code, bytes),
        has_fp64 ? gemv_compile(c, ocl_fpp64, code, bytes) : null
    };
    static const char* gemv_kernel_name[] = {"gemv16", "gemv32", "gemv64"};
    for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
        if (p[fpp] != null) {
            gemv_kernel[fpp] = ocl.create_kernel(p[fpp], gemv_kernel_name[fpp]);
            ocl.release_program(p[fpp]);
        }
    }
}

static void gemv_fini() {
    for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
        if (gemv_kernel[fpp] != null) {
            ocl.release_kernel(gemv_kernel[fpp]);
            gemv_kernel[fpp] = null;
        }
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
    ocl.init();
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
        gemv_init(&c);
        verbose = false; // set to true if crashes
        test(&c, 8, 16, init_vc0, init_mx0, d->name);
        test(&c, 256, 256, init_vc1, init_mx1, d->name);
        test(&c, 1024, 1024, init_vc1, init_mx1, d->name);
        test(&c, 4 * 1024,  4 * 1024, init_vc1, init_mx1, d->name);
        test(&c, 4 * 1024, 16 * 1024, init_vc1, init_mx1, d->name); // GPT-J 6b inermost gemv()
        // GPT-J 6b [16K, 4K, 7] * [4K, 7] 7 is probably dimension of word embedding?
        // 32*64 = 2048M x sizeof(fp32_t) = 8GB
//      test(&c, 32 * 1024, 64 * 1024, init_vc1, init_mx1, d->name); // too much
        if (i < 1) {
            traceln("");
            traceln(">>>");
            test(&c, 16 * 1024, 64 * 1024, init_vc1, init_mx1, d->name);
            traceln("<<<");
            traceln("");
        }
        traceln("");
        traceln("done");
        gemv_fini(&c);
        ocl.close(&c);
    }
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    tests();
}

#if 0
 groups x items NVIDIA GeForce RTX 3080 Laptop GPU

 No significant differences detected of groups/items configurations

 64x256
    n: 16384 m: 65536 groups: 64 items: 256 compute units: 48
    16384x65536 gpu: 24.454ms GFlops: 87.817
    16384x65536 gpu: 384.664 avx: 189.613 ms
  128x128
    n: 16384 m: 65536 groups: 128 items: 128 compute units: 48
    16384x65536 gpu: 23.764ms GFlops: 90.367
    16384x65536 gpu: 385.562 avx: 185.405 ms
  256x64
    n: 16384 m: 65536 groups: 256 items: 64 compute units: 48
    16384x65536 gpu: 23.496ms GFlops: 91.399
    16384x65536 gpu: 386.828 avx: 187.998 ms NVIDIA GeForce RT

#endif