#include "rt.h"
#include "ocl.h"
#include "dot.h"

static ocl_kernel_t gemv_kernel[3]; // TODO: move to gemv_context_t
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

static void gemv(ocl_context_t* c, int fpp,
        ocl_memory_t mx, ocl_memory_t vc,
        uint32_t n, uint32_t m, ocl_memory_t rs) {
//  fatal_if(n == 0 || (n & (n - 1)) != 0 ||
//           m == 0 || (m & (m - 1)) != 0,
//           "n: %d n: %d both must be power of 2", n, m);
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
         {&rs,  sizeof(ocl_memory_t)},
         {null, items * sizeof(fp32_t)}, // wc
         {&n,   sizeof(int32_t)},
         {&m,   sizeof(int32_t)}
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
        if (n <= 64) { printf("vc: "); vprintln(vc, n); }
        if (n <= 64 && m < 50) { printf("mx:\n"); mprintln(mx, n, m); }
        if (m <= 64) { printf("cpu: "); vprintln(verify, m); }
    }
    ocl.unmap(c, matrix, mx);
    ocl.unmap(c, vector, vc);
    gemv(c, ocl_fpp32, matrix, vector, n, m, result);
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
    append("-D max_subgroups=%lld ", d->max_subgroups);
    // for fp16_t dot(half4, half4) is not availabe.
    // TODO: This needs to be dynamic check in ocl.create() context.
    if (fpp != ocl_fpp16) { append("-D dot%dx4=dot ", fpp); }
    #pragma pop_macro("append")
    *p = 0;
//  traceln("%s", options);
    return options;
}

static ocl_program_t gemv_compile(ocl_context_t* c, int fpp,
        const void* code, int64_t bytes) {
    const char* opts = gemv_program_options(c, fpp);
    return ocl.compile(c, code, bytes, opts, null, 0);
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
#if XXX
            int64_t max_sub_group_size_for_ndrange = 0;
            r = clGetKernelSubGroupInfo((cl_kernel)gemv_kernel[fpp],
                (cl_device_id)ocl.devices[c->ix].id,
                CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                sizeof(int64_t), &max_sub_group_size_for_ndrange, 0, null, null);
#endif // TODO
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
        verbose = true; // set to true if crashes
//      test(&c, 2, 3, init_vc0, init_mx0, d->name);
        test(&c, 16, 32, init_vc0, init_mx0, d->name);
#if 0
        test(&c, 1024, 1024, init_vc1, init_mx1, d->name);
        test(&c, 4 * 1024,  4 * 1024, init_vc1, init_mx1, d->name);
        test(&c, 4 * 1024, 16 * 1024, init_vc1, init_mx1, d->name); // GPT-J 6b inermost gemv()
        // only run on NVIDIA GPU. Intel UHD Graphics GPU reports 16GB as global memory
        // but cannot allocate any of this huge memory chunks
        if (strstr(d->vendor, "NVIDIA") != null) {
            test(&c, 16 * 1024, 64 * 1024, init_vc1, init_mx1, d->name);
            traceln("--------------------------------------------------");
            // GPT-J 6b [16K, 4K, 7] * [4K, 7] 7 is probably dimension of word embedding?
            // 32*64 = 2048M x sizeof(fp32_t) = 8GB
            // use 30x60 instead to fit into 8GB of GPU memory
            // the accumulated error is too big to check:
            unchecked++;
            test(&c, 30 * 1024, 60 * 1024, init_vc1, init_mx1, d->name);
            unchecked--;
            traceln("==================================================");
        }
#endif
        gemv_fini(&c);
        ocl.close(&c);
    }
}

// run with args: compile ..\gemv.cl ..\gemv

static void compile(int32_t argc, const char* argv[]) {
    fatal_if(argc != 4);
    enum { source_max = 256 * 1024 };
    char* source = malloc(source_max);
    fatal_if(source == null);
    FILE* f = fopen(argv[2], "r");
    fatal_if(f == null, "failed to open file: %s", argv[2]);
    int64_t source_bytes = (int64_t)fread(source, 1, source_max, f);
    if (f != null) { fclose(f); }
    int64_t binary_sizes[16] = {0};
    for (int i = 0; i < ocl.count && source_bytes > 0; i++) {
//      ocl.dump(i);
        const ocl_device_t* d = &ocl.devices[i];
        ocl_context_t c = ocl.open(i, null);
        // fp32_t supported on most of GPU of interest
        // Intel UHD Graphics GPU does not support doubles at all and reports
        // double_fp_config == 0.
        // fp16_t (half) is much trickier... because
        // NVIDIA GeForce RTX 3080 Laptop GPU supports "half" w/o reporting cl_khr_fp16
        // Intel UHD Graphics GPU supports "half" and reports cl_khr_fp16
        // PS: Intel also supports and reports cl_khr_subgroup_extended_types,
        //     NVIDIA is silent about its support if any TODO: investigate
        int from = ocl_fpp16;
        int to = d->double_fp_config == 0 ? ocl_fpp32 : ocl_fpp64;
        for (int fpp = from; fpp <= to; fpp++) {
            traceln("compile: %s for %s @ %s", argv[2], ocl_fpp_names[fpp], d->name);
            traceln("");
            ocl_program_t p = gemv_compile(&c, fpp, source, source_bytes);
            if (p == null) {
                traceln("failed to compile for %s: %s", ocl_fpp_names[fpp], argv[2]);
            } else {
                int64_t n = 0; // number of devices
                fatal_if(clGetProgramInfo((cl_program)p, CL_PROGRAM_NUM_DEVICES,
                    sizeof(n), &n, null) != 0);
                fatal_if(n != 1, "should be compiled for single device");
                int r = clGetProgramInfo((cl_program)p, CL_PROGRAM_BINARY_SIZES,
                    sizeof(binary_sizes), binary_sizes, null);
                if (r == 0 && n == 1 && binary_sizes[0] > 0) {
                    byte_t* binary = (byte_t*)malloc(binary_sizes[0]);
                    fatal_if(binary == null);
                    fatal_if(clGetProgramInfo((cl_program)p, CL_PROGRAM_BINARIES,
                            binary_sizes[0], &binary, null) != 0);
                    fatal_if(binary_sizes[0] <= 0);
                    char dn[256]; // device name
                    strncpy(dn, d->name, countof(dn)); // first work only:
                    if (strchr(dn, 0x20) != 0) { *strchr(dn, 0x20) = 0; }
                    char* s = dn;
                    while (*s != 0) { *s = (char)tolower(*s); s++; }
                    char fn[256]; // file name
                    snprintf(fn, countof(fn), "%s%d.%s.bin", argv[3],
                        ocl_fpp_bytes[fpp] * 8, dn);
                    f = fopen(fn, "wb");
                    fatal_if(f == null, "failed to create file: %s", fn);
                    int64_t written = (int64_t)fwrite(binary, 1, binary_sizes[0], f);
                    fatal_if(written != binary_sizes[0]);
                    fclose(f);
                    // .bin suitable for clCreateProgramWithBinary() which is a bit
                    // useless because it is device specific.
                    // https://www.khronos.org/blog/offline-compilation-of-opencl-kernels-into-spir-v-using-open-source-tooling
                    // can be used to create portable .spv SPIR-V binaries and load them
                    // with clCreateProgramWithIL() call.
                }
            }
        }
        ocl.close(&c);
    }
    if (source_bytes <= 0) {
        traceln("failed to read: %s", argv[2]);
    }
    free(source);
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    ocl.init();
    if (argc > 1 && strcmp(argv[1], "compile") == 0) {
        if (argc == 4) {
            compile(argc, argv);
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

  - GFlops for now:

  30720x61440 NVIDIA GeForce RTX 3080 Laptop GPU
     groups: 64 items: 480 compute units: 48
     gpu: 26.023ms GFlops: 145.060
     gpu: 790.689 avx: 343.893 ms
     // inaccurate rounding errors:
     delta: 14546 epsilon: 225 cpu[0]: 31170 gpu[0]: 16624

#endif