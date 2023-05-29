#include "gemv.h"

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

static void ocl_gemv(gemv_t* g, int fpp, int64_t offset,
        ocl_memory_t mx, ocl_memory_t vc,
        ocl_memory_t rs, int64_t n, int64_t m) {
    (void) offset; // TODO: use it
    if (ocl.is_profiling(g->c)) { g->c->ov->profiling_count = 0; }
    ocl_device_t* d = &ocl.devices[g->c->ix];
    // if n > max items per group GPU will run multiple groups:
    const int xn = n % 16 == 0 ? 16 : (n % 4 == 0) ? 4 : 1;
    // row width in fp_t or vec4 of fpXX_t elements
    const int64_t rw = n / xn; // row[] width
    ocl_kernel_t k = xn == 16 ? g->kernel16x[fpp] :
                    (xn ==  4 ? g->kernel4x[fpp] : g->kernel[fpp]);
    int64_t items_per_group = min(d->max_items[0], rw);
    int64_t local_bytes = sizeof(fp32_t) * items_per_group *
        (d->max_subgroups == 0 ? 1 : d->max_subgroups);
    fp64_t user = seconds();
    ocl_event_t done = ocl.enqueue(g->c, k, rw,
        &mx,  sizeof(ocl_memory_t),
        &vc,  sizeof(ocl_memory_t),
        &rs,  sizeof(ocl_memory_t),
        null, local_bytes, // shared memory for all work-items inside group
        &rw,  sizeof(int32_t),
        &m,   sizeof(int32_t),
        null, 0
    );
    if (ocl.is_profiling(g->c)) { ocl.profile_add(g->c, done); }
    ocl.finish(g->c);
    ocl.release_event(done);
    user = seconds() - user;
    gpu_time = min(gpu_time, user);
    if (ocl.is_profiling(g->c)) {
        ocl_profiling_t* p = &g->c->ov->profiling[0];
        // n * m * 2 does not account for sum parallel reduction
        int32_t log2_m = 0; int32_t cm = m;
        while (cm >>= 1) { log2_m++; }
        p[0].count  = n * m * log2_m;  // kernel `invocations`
        p[0].fops   = 2U;     // + and * fp32_t operation each
        p[0].i32ops = n * 2U; // + and * int32_t operation each
        ocl.profile(&p[0]);
        if (n > 64 && m > 64) {
            traceln("%dx%d gpu: %6.3fms GFlops: %6.3f",
                    n, m, p[0].time * MSEC_IN_SEC, p[0].gflops);
        }
        avg_time += p->time;
        avg_gflops += p->gflops;
    }
}

static const char* gemv_program_options(gemv_t* g, int fpp) {
    const ocl_device_t* d = &ocl.devices[g->c->ix];
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
    append("-D int64_t=long  -D uint64_t=ulong ");
    append("-D fp16_t=half -D fp32_t=float -D fp64_t=double ");
    append("-cl-std=CL%d.%d ", d->c_version_major, d->c_version_minor);
    static const char* type_t[] = {"half",  "float", "double"};
    static const char* fpmx_t[] = {"float", "float", "double"}; // fpmx_t sum += ...
    append("-D fp_t=%s -D fpmx_t=%s -D fpmx4_t=%s4 -D fpp=%d ",
        type_t[fpp], fpmx_t[fpp], fpmx_t[fpp], ocl_fpp_bytes[fpp] * 8);
    append("-D fp16x4_t=half4 -D fp32x4_t=float4 -D fp64x4_t=double4 ");
    append("-D vec4_t=%s4 ", type_t[fpp]);
    append("-D max_subgroups=%lld ", d->max_subgroups);
    #pragma pop_macro("append")
    *p = 0;
//  traceln("%s", options);
    return options;
}

static ocl_program_t gemv_compile(gemv_t* g, int fpp,
        const void* code, int64_t bytes) {
    const char* opts = gemv_program_options(g, fpp);
    return ocl.compile(g->c, code, bytes, opts, null, 0);
}

static void gemv_init(gemv_t* g, ocl_context_t* c) {
    memset(g, 0, sizeof(*g));
    g->c = c;
    ocl_device_t* d = &ocl.devices[c->ix];
    void* code = null;
    int64_t bytes64 = 0;
    int r = memmap_resource("gemv_cl", &code, &bytes64);
    fatal_if(r != 0 || code == null || bytes64 == 0, "is gemv.cl in gemv.rc?");
    fatal_if(bytes64 > INT_MAX, "blast.cl %lld bytes", bytes64);
    int bytes = (int)bytes64;
    const bool has_fp16 = true; // TODO: ? strstr(d->ext, "cl_khr_fp16") != null;
    const bool has_fp32 = d->float_fp_config  != 0;
    const bool has_fp64 = d->double_fp_config != 0;
    ocl_program_t p[3] = {
        has_fp16 ? gemv_compile(g, ocl_fpp16, code, bytes) : null,
        has_fp32 ? gemv_compile(g, ocl_fpp32, code, bytes) : null,
        has_fp64 ? gemv_compile(g, ocl_fpp64, code, bytes) : null
    };
    static const char* gemv_kernel_name[] = {"gemv16", "gemv32", "gemv64"};
    static const char* gemv_kernel4x_name[] = {"gemv16x4", "gemv32x4", "gemv64x4"};
    static const char* gemv_kernel16x_name[] = {"gemv16x16", "gemv32x16", "gemv64x16"};
    for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
        if (p[fpp] != null) {
            g->kernel[fpp] = ocl.create_kernel(p[fpp], gemv_kernel_name[fpp]);
            g->kernel4x[fpp] = ocl.create_kernel(p[fpp], gemv_kernel4x_name[fpp]);
            g->kernel16x[fpp] = ocl.create_kernel(p[fpp], gemv_kernel16x_name[fpp]);
            // TODO: x16
            ocl.release_program(p[fpp]);
        }
    }
}

static void gemv_fini(gemv_t* g) {
    for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
        if (g->kernel[fpp] != null) {
            ocl.release_kernel(g->kernel[fpp]);
            ocl.release_kernel(g->kernel4x[fpp]);
            // TODO: x16
            g->kernel[fpp] = null;
            g->kernel4x[fpp] = null;
            g->kernel16x[fpp] = null;
        }
    }
    g->c = null;
}

#pragma warning(disable: 4100) // TODO: remove me

void ocl_gemv16(gemv_t* g, int64_t offset, ocl_memory_t mx,
    ocl_memory_t vc, ocl_memory_t rs, int64_t n, int64_t m) {
    ocl_gemv(g, ocl_fpp16, offset, mx, vc, rs, n, m);
}

void ocl_gemv32(gemv_t* g, int64_t offset, ocl_memory_t mx,
    ocl_memory_t vc, ocl_memory_t rs, int64_t n, int64_t m) {
    ocl_gemv(g, ocl_fpp32, offset, mx, vc, rs, n, m);
}

void ocl_gemv64(gemv_t* g, int64_t offset, ocl_memory_t mx,
    ocl_memory_t vc, ocl_memory_t rs, int64_t n, int64_t m) {
    ocl_gemv(g, ocl_fpp64, offset, mx, vc, rs, n, m);
}

// testing convenience (slow performs copy to/from GPU memory):
void gemv16(gemv_t* g, fp16_t mx[/*m][n*/], fp32_t vc[/*n*/], fp32_t rs[/*n*/],
    int64_t n, int64_t m) {
}

void gemv32(gemv_t* g, fp32_t mx[/*m][n*/], fp32_t vc[/*n*/], fp32_t rs[/*n*/],
    int64_t n, int64_t m) {
}

void gemv64(gemv_t* g, fp64_t mx[/*m][n*/], fp64_t vc[/*n*/], fp64_t rs[/*n*/],
    int64_t n, int64_t m) {
}

gemv_if gemv = {
    .init = gemv_init,
    .ocl_gemv16 = ocl_gemv16,
    .ocl_gemv32 = ocl_gemv32,
    .ocl_gemv64 = ocl_gemv64,
    .gemv16 = gemv16,
    .gemv32 = gemv32,
    .gemv64 = gemv64,
    .fini = gemv_fini
};
