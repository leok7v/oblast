#include "gemv.h"

static void ocl_gemv(gemv_t* g, int fpp,
        intptr_t mx_offset, ocl_memory_t mx/*[m][n]*/,
        intptr_t vc_offset, ocl_memory_t vc/*[n]*/,
        intptr_t rs_offset, ocl_memory_t rs/*[m]*/,
        int64_t n, int64_t m) {
    if (ocl.is_profiling(g->c)) { g->c->ov->profiling_count = 0; }
    ocl_device_t* d = &ocl.devices[g->c->ix];
    // if n > max items per group GPU will run multiple groups:
    const int xn = n % 16 == 0 ? 16 : (n % 4 == 0) ? 4 : 1;
    // row width in fp_t or vec4 of fpXX_t elements
    const int64_t rw = n / xn; // row[] width
    ocl_kernel_t k = xn == 16 ? g->kernel16x[fpp] :
                    (xn ==  4 ? g->kernel4x[fpp] : g->kernel[fpp]);
    int64_t items_per_group = min(d->max_items[0], rw);
    int64_t local_bytes = ocl_fpp_bytes[fpp] * items_per_group *
        max(d->max_subgroups, 1);
    ocl_event_t done = ocl.enqueue(g->c, k, rw,
        &mx_offset, sizeof(intptr_t),
        &mx,        sizeof(ocl_memory_t),
        &vc_offset, sizeof(intptr_t),
        &vc,        sizeof(ocl_memory_t),
        &rs_offset, sizeof(intptr_t),
        &rs,        sizeof(ocl_memory_t),
        null,       local_bytes, // shared memory for all work-items inside group
        &rw,        sizeof(int32_t),
        &m,         sizeof(int32_t),
        null, 0
    );
    if (ocl.is_profiling(g->c)) { ocl.profile_add(g->c, done); }
    ocl.finish(g->c);
    ocl.release_event(done); // p->e is still holding it
    if (ocl.is_profiling(g->c)) {
        ocl_profiling_t* p = &g->c->ov->profiling[0];
        p[0].count  = rw; // kernel invocations
        p[0].fops   = m * xn * 3; // fp ops
        p[0].i32ops = m * xn * 3; // indexing ops
        ocl.profile(&p[0]); // p->e will be released
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
    static const char* type_t[] = {"fp16_t", "fp32_t", "fp64_t", "bf16_t"};
    static const char* vec4_t[] = {"fp16x4_t", "fp32x4_t", "fp64x4_t", "bf16x4_t"};
    // accu_t sum += ...
    static const char* accu_t[] = {"fp32_t", "fp32_t", "fp64_t", "fp32_t"};
    static const char* acc4_t[] = {"fp32x4_t", "fp32x4_t", "fp64x5_t", "fp32x4_t"};
    static const int   accu_b[] = {32, 32, 64, 32}; // bits in accu_t type
    append("-D fp_t=%s -D accu=%d -D accu_t=%s -D acc4_t=%s -D fpp=%d ",
        type_t[fpp], accu_b[fpp], accu_t[fpp], acc4_t[fpp], ocl_fpp_bytes[fpp] * 8);
    if (fpp == ocl_bfp16) {
        append("-D bfp16=1 ");
    } else { // (fpp != ocl_bfp16) bf16 does not have vec4
        append("-D fpv4_t=%s ", vec4_t[fpp]);
    }
    append("-D max_subgroups=%lld ", d->max_subgroups); // Intel extension
    append("-cl-std=CL%d.%d ", d->c_version_major, d->c_version_minor);
    // https://man.opencl.org/clBuildProgram.html
    append("-Werror "); // --warnings-as-errors / does not work :(
    append("-cl-std=CL%d.%d ", d->c_version_major, d->c_version_minor);
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
    ocl_program_t p[4] = {
        has_fp16 ? gemv_compile(g, ocl_fpp16, code, bytes) : null,
        has_fp32 ? gemv_compile(g, ocl_fpp32, code, bytes) : null,
        has_fp64 ? gemv_compile(g, ocl_fpp64, code, bytes) : null,
        has_fp32 ? gemv_compile(g, ocl_bfp16, code, bytes) : null,
    };
    static const char* kernel_name[4][4] = {
        {"gemv16",    "gemv32",    "gemv64",    "bfmv16"},
        {"gemv16x4",  "gemv32x4",  "gemv64x4",  "bfmv16x4"},
        {"gemv16x16", "gemv32x16", "gemv64x16", "bfmv16x16"}
    };
    for (int fpp = ocl_fpp_first; fpp <= ocl_fpp_last; fpp++) {
        if (p[fpp] != null) {
            g->kernel[fpp]    = ocl.create_kernel(p[fpp], kernel_name[0][fpp]);
            g->kernel4x[fpp]  = ocl.create_kernel(p[fpp], kernel_name[1][fpp]);
            g->kernel16x[fpp] = ocl.create_kernel(p[fpp], kernel_name[2][fpp]);
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
            g->kernel[fpp]    = null;
            g->kernel4x[fpp]  = null;
            g->kernel16x[fpp] = null;
        }
    }
    g->c = null;
}

gemv_if gemv = {
    .init = gemv_init,
    .gemv = ocl_gemv,
    .fini = gemv_fini
};
