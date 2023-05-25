#include "rt.h"
#include "ocl.h"
#include "dot.h"

static ocl_kernel_t gemv_kernel[3]; // TODO: move to gemv_context_t
static int measures = 1;
static bool verbose = true;

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

static double avg_time;
static double avg_user;
static double avg_host;
static double avg_gflops;

static double avx_time;
static double gpu_time;

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
        int32_t n, int32_t m, ocl_memory_t rs) {
    if (ocl.is_profiling(c)) { c->ov->profiling_count = 0; }
    int64_t max_groups = ocl.devices[c->ix].max_items[0];
    int64_t max_items  = ocl.devices[c->ix].max_items[0];
    int64_t groups = (m + max_items - 1) / max_items;
    int64_t items  = m / groups;
    assert(groups <= max_groups && items <= max_items && groups * items == m);
    ocl_kernel_t k = gemv_kernel[fpp];
    ocl_arg_t a[] =
        {{&mx, sizeof(ocl_memory_t)},
         {&vc, sizeof(ocl_memory_t)},
         {&n,  sizeof(int32_t)},
         {&rs, sizeof(ocl_memory_t)}
    };
    double user = seconds();
    for (int i = 0; i < measures; i++) {
        ocl_event_t done = ocl.enqueue_range_kernel(c, k, groups, items, countof(a), a);
        if (ocl.is_profiling(c)) { ocl.profile_add(c, done); }
    }
    ocl.finish(c);
    user = seconds() - user;
    gpu_time = min(gpu_time, user);
    if (ocl.is_profiling(c)) {
        ocl_profiling_t* p = &c->ov->profiling[0];
        p[0].count  = m;     // kernel invocations
        p[0].fops   = n * 2; // + and * fp32_t operation each
        p[0].i32ops = n * 2; // + and * int32_t operation each
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
        if (verbose) {
            traceln("%dx%d gpu: %6.3f GFlops: %6.3f",
                    n, m, p->time * USEC_IN_SEC / measures, p->gflops);
        }
        avg_time += p->time;
        avg_gflops += p->gflops;
    }
}

static void test(ocl_context_t* c, int32_t n, int32_t m,
                 fp32_t (*init_vc)(int32_t i),
                 fp32_t (*init_mx)(int32_t j, int32_t i)) {
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
    if (n < 50) { vprintln(vc, n);    }
    if (n < 50) { mprintln(mx, n, m); }
    ocl.unmap(c, matrix, mx);
    ocl.unmap(c, vector, vc);
    gemv(c, ocl_fpp32, matrix, vector, n, m, result);
    fp32_t* rs = (fp32_t*)ocl.map(c, ocl_map_read, result, 0, m * sizeof(fp32_t));
    if (m < 50) { vprintln(rs, m); }
    ocl.unmap(c, result, rs);
    // verification
    mx = (fp32_t*)ocl.map(c, ocl_map_write, matrix, 0, m * n * sizeof(fp32_t));
    vc = (fp32_t*)ocl.map(c, ocl_map_write, vector, 0,     n * sizeof(fp32_t));
    for (int32_t j = 0; j < m; j++) {
        fp32_t sum = 0;
        for (int32_t i = 0; i < n; i++) { sum += vc[i] * mx[j * n + i]; }
        assert(sum == rs[j], "%g", fabs(sum - rs[j]));
    }
    double user = seconds();
    for (int32_t j = 0; j < m; j++) {
        fp32_t sum = (fp32_t)dot32(vc, 1, &mx[j * n], 1, n);
        fatal_if(isnan(sum));
//      assert(sum == rs[j], "%g", fabs(sum - rs[j]));
    }
    user = seconds() - user;
    avx_time = min(avx_time, user);
    ocl.unmap(c, matrix, mx);
    ocl.unmap(c, vector, vc);
    // cleanup
    ocl.deallocate(result);
    ocl.deallocate(vector);
    ocl.deallocate(matrix);
    if (measures == 1) {
        traceln("gpu: %.3f avx: %.3f ms", gpu_time * MSEC_IN_SEC, avx_time * MSEC_IN_SEC);
    }
}

static const char* gemv_program_options(ocl_context_t* c, int fpp) {
    static const char* type_t[] = {"half", "float", "double"};
    static const char* suffix[] = {"16", "32", "64"};
    const char* fp_t = type_t[fpp];
    const ocl_device_t* d = &ocl.devices[c->ix];
    static char options[4096];
    char* p = options;
    #pragma push_macro("append")
    #define append(...) do {                                             \
        intptr_t k = options + countof(options) - p - 1;                 \
        fatal_if(k <= 0, "options[%d] overflow", (int)countof(options)); \
        p += snprintf(p, k, "" __VA_ARGS__);                             \
    } while (0)
    append("-D fp16_t=half -D fp32_t=float -D fp64_t=double ");
    append("-D int32_t=int -D int64_t=long ");
    append("-cl-std=CL%d.%d ", d->c_version_major, d->c_version_minor);
    append("-D fp_t=%s -D vec4=%s4 -D vec8=%s8 -D vec16=%s16 -D fpp=%s %s ",
           fp_t, fp_t,fp_t, fp_t, suffix[fpp],
          (fpp == ocl_fpp16 ? "-D fp16_surrogate" : ""));
    #pragma pop_macro("append")
    *p = 0;
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
    const bool has_fp16 = (d->fp_config & ocl_fp16) != 0;
    const bool has_fp64 =  d->double_fp_config != 0;
    ocl_program_t p[3] = {
        has_fp16 ? gemv_compile(c, ocl_fpp16, code, bytes) : null,
                   gemv_compile(c, ocl_fpp32, code, bytes),
        has_fp64 ? gemv_compile(c, ocl_fpp64, code, bytes) : null
    };
    static const char* gemv_kernel_name[] = {"gemv16", "gemv32", "gemv64"};
    for (int fp = ocl_fpp16; fp <= ocl_fpp64; fp++) {
        if (p[fp] != null) {
            gemv_kernel[fp] = ocl.create_kernel(p[fp], gemv_kernel_name[fp]);
            ocl.release_program(p[fp]);
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
        const ocl_device_t* d = &ocl.devices[i];
        ocl_override_t ov = {
            .profiling = p,
            .max_profiling_count = countof(p),
            .profiling_count = 0
        };
        ocl_context_t c = ocl.open(i, &ov);
        traceln("%s", d->name);
        gemv_init(&c);
        measures = 1;
        test(&c,  4 * 1024, 16 * 1024, init_vc1, init_mx1); // 1 layer of ANN
        if (i < 1) {
            if (32 * 1024 * 16 * 1024 < d->global_memory) {
                test(&c, 32 * 1024, 16 * 1024, init_vc1, init_mx1); // 8 x 1 layer of ANN
            }
            if (32 * 1024 * 32 * 1024 < d->global_memory) {
                test(&c, 32 * 1024, 32 * 1024, init_vc1, init_mx1);
            }
        }
//      test(&c, 8, 16, init_vc0, init_mx0);
        gemv_fini(&c);
        ocl.close(&c);
    }
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    tests();
}