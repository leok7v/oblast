#include "gemv.h"

static void ocl_gemv(gemv_t* g, int fpp, int64_t offset,
        ocl_memory_t mx, ocl_memory_t vc,
        ocl_memory_t rs, int64_t n, int64_t m) {
    assert(offset == 0, "Not implemented yet");
    (void) offset; // TODO: use it
    if (ocl.is_profiling(g->c)) { g->c->ov->profiling_count = 0; }
    ocl_device_t* d = &ocl.devices[g->c->ix];
    // if n > max items per group GPU will run multiple groups:
//  const int xn = n % 32 == 0 ? 32 : (n % 16 == 0 ? 16 : (n % 4 == 0) ? 4 : 1);
    const int xn = n % 16 == 0 ? 16 : (n % 4 == 0) ? 4 : 1;
    // row width in fp_t or vec4 of fpXX_t elements
    const int64_t rw = n / xn; // row[] width
    ocl_kernel_t k = // xn == 32 ? g->kernel32x[fpp] :
                       (xn == 16 ? g->kernel16x[fpp] :
                       (xn ==  4 ? g->kernel4x[fpp] : g->kernel[fpp]));
    int64_t items_per_group = min(d->max_items[0], rw);
    int64_t local_bytes = ocl_fpp_bytes[fpp] * items_per_group *
        max(d->max_subgroups, 1);
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
    if (ocl.is_profiling(g->c)) {
        ocl_profiling_t* p = &g->c->ov->profiling[0];
        p[0].count  = rw; // kernel invocations
        p[0].fops   = m * xn * 3; // fp ops
        p[0].i32ops = m * xn * 3; // indexing ops
        ocl.profile(&p[0]);
//      if (n > 64 && m > 64) {
//          println("%dx%d gpu: %6.3fms %4.1fGFlops",
//                  n, m, p[0].time * MSEC_IN_SEC, p[0].gflops);
//      }
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
    static const int   fpmx_b[] = {32, 32, 64}; // bits in fpmx_t type
    append("-D fp_t=%s -D fpmx=%d -D fpmx_t=%s -D fpmx4_t=%s4 -D fpp=%d ",
        type_t[fpp], fpmx_b[fpp], fpmx_t[fpp], fpmx_t[fpp], ocl_fpp_bytes[fpp] * 8);
    append("-D fp16x4_t=half4 -D fp32x4_t=float4 -D fp64x4_t=double4 ");
    append("-D vec4_t=%s4 ", type_t[fpp]);
    append("-D max_subgroups=%lld ", d->max_subgroups);
    #pragma pop_macro("append")
    *p = 0;
//  println("%s", options);
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
    const bool has_fp16 = d->fp16_config != 0;
    const bool has_fp32 = d->fp32_config != 0;
    const bool has_fp64 = d->fp64_config != 0;
    ocl_program_t p[3] = {
        has_fp16 ? gemv_compile(g, ocl_fpp16, code, bytes) : null,
        has_fp32 ? gemv_compile(g, ocl_fpp32, code, bytes) : null,
        has_fp64 ? gemv_compile(g, ocl_fpp64, code, bytes) : null
    };
    static const char* kernel_name[4][3] = {
        {"gemv16",    "gemv32",    "gemv64"},
        {"gemv16x4",  "gemv32x4",  "gemv64x4"},
        {"gemv16x16", "gemv32x16", "gemv64x16"},
        {"gemv16x32", "gemv32x32", "gemv64x32"}
    };
    for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
        if (p[fpp] != null) {
            g->kernel[fpp]    = ocl.create_kernel(p[fpp], kernel_name[0][fpp]);
            g->kernel4x[fpp]  = ocl.create_kernel(p[fpp], kernel_name[1][fpp]);
            g->kernel16x[fpp] = ocl.create_kernel(p[fpp], kernel_name[2][fpp]);
            g->kernel32x[fpp] = ocl.create_kernel(p[fpp], kernel_name[3][fpp]);
            ocl.release_program(p[fpp]);
        }
    }
}

static void gemv_fini(gemv_t* g) {
    for (int fpp = ocl_fpp16; fpp <= ocl_fpp64; fpp++) {
        if (g->kernel[fpp] != null) {
            ocl.release_kernel(g->kernel[fpp]);
            ocl.release_kernel(g->kernel4x[fpp]);
            ocl.release_kernel(g->kernel16x[fpp]);
            ocl.release_kernel(g->kernel32x[fpp]);
            g->kernel[fpp]    = null;
            g->kernel4x[fpp]  = null;
            g->kernel16x[fpp] = null;
            g->kernel32x[fpp] = null;
        }
    }
    g->c = null;
}

#pragma warning(disable: 4100) // TODO: remove me

// TODO: implement or remove?

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
    .gemv = ocl_gemv,
    .gemv16 = gemv16,
    .gemv32 = gemv32,
    .gemv64 = gemv64,
    .fini = gemv_fini
};
