#include "rt.h"
#include "ocl.h"

#ifdef OCL_USE_NVIDIA_12_LIB_BINDINGS
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64\OpenCL.lib
#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64/OpenCL.lib")
#else // dynamic bindings with OpenCL.dll on the path
// #include <CL/cl_bind.inc> // dynamically bind everything
#endif

static_assert(ocl_fpp16 == 0 && ocl_fpp32 == 1 &&
              ocl_fpp64 == 2 && ocl_bfp16 == 3, "order");

const char* ocl_fpp_names[4] = {"fp16", "fp32", "fp64", "bf16"};

const int ocl_fpp_bytes[4] = {
    (int)sizeof(fp16_t), (int)sizeof(fp32_t),
    (int)sizeof(fp64_t), (int)sizeof(bf16_t)
};

static ocl_device_t ocl_devices[32]; // up to 32 GPUs supported

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

#define call(f) do { /* fail fast OpenCL API call */           \
    cl_int __err__;                                            \
    fatal_if((__err__  = (f)) != 0, "%s", ocl.error(__err__)); \
} while (0)

#define not_null(p, r) do {                             \
    fatal_if(r != 0 || p == null, "%s", ocl.error(r));  \
} while (0)

static void ocl_error_notify(const char * errinfo,
    const void* private_info, size_t cb, void* user_data) {
    println("ERROR: %.256s", errinfo);
    (void)private_info;
    (void)cb;
    (void)user_data;
}

static void* ocl_create_queue(ocl_context_t* c, bool profiling);

static bool ocl_is_profiling(const ocl_context_t* c) {
    const bool profiling = c->ov != null && c->ov->max_profiling_count > 0;
    if (profiling) { fatal_if(c->ov->profiling == null, "need array"); }
    return profiling;
}

static ocl_context_t ocl_open(int32_t ix, ocl_override_t* ov) {
    ocl_context_t c = {0};
    call(!(0 <= ix && ix < ocl.count));
    ocl_device_t* d = &ocl.devices[ix];
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)d->platform, 0
    };
    cl_int r = 0;
    cl_device_id id = (cl_device_id)d->id;
    c.ov = ov;
    c.ix = ix;
    /* user_data: null will be passed to notify() */
    c.c = clCreateContext(properties, 1, &id, ocl_error_notify, null, &r);
    not_null(c.c, r);
    c.q = ocl_create_queue(&c, ocl.is_profiling(&c));
    return c;
}

static bool ocl_has_fpp(ocl_context_t* c, int fpp) {
    const ocl_device_t* d = &ocl.devices[c->ix];
    switch (fpp) {
        case ocl_fpp16: return d->fp16_config != 0;
        case ocl_fpp32: return d->fp32_config != 0;
        case ocl_fpp64: return d->fp64_config != 0;
        case ocl_bfp16: return d->fp32_config != 0; // bf16 simulated via fp32
        default: fatal_if("fpp", "%d", fpp);
    }
    return false;
}

static void* ocl_create_queue(ocl_context_t* c, bool profiling) {
    cl_context ctx = c->c;
    cl_device_id device_id = (cl_device_id)ocl.devices[c->ix].id;
    cl_int r = 0;
    static const cl_command_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device_id,
            profiling ? properties : null, &r);
    not_null(q, r);
    return q;
}

static void ocl_flush(ocl_context_t* c) {
    call(clFlush((cl_command_queue)c->q));
}

static void ocl_finish(ocl_context_t* c) {
    call(clFinish((cl_command_queue)c->q));
}

static void ocl_dispose_queue(ocl_context_t* c) {
    call(clReleaseCommandQueue((cl_command_queue)c->q));
}

// https://streamhpc.com/blog/2013-02-03/opencl-basics-flags-for-the-creating-memory-objects/
// https://man.opencl.org/clCreateBuffer.html

static ocl_memory_t ocl_alloc(ocl_context_t* c, int access, size_t bytes) {
    cl_int r = 0;
    cl_mem m = clCreateBuffer(c->c, access|CL_MEM_ALLOC_HOST_PTR, bytes, null, &r);
    return (ocl_memory_t)m;
}

static ocl_memory_t ocl_allocate(ocl_context_t* c, int access, size_t bytes) {
    cl_int r = 0;
    cl_mem m = clCreateBuffer(c->c, access|CL_MEM_ALLOC_HOST_PTR, bytes, null, &r);
    not_null(m, r);
    return (ocl_memory_t)m;
}

static void ocl_deallocate(ocl_memory_t m) {
    // Customary free(null) is OK because 1. it's mostly harmless
    // 2. simplifies error hangling in multiple alloc() situations
    // (which are almost always).
    if (m != null) { call(clReleaseMemObject((cl_mem)m)); }
}

static void* ocl_map(ocl_context_t* c, int mapping, ocl_memory_t m, size_t offset,
        size_t bytes) {
    cl_int r = 0;
    // blocking_map: true sync mapping
    void* a = clEnqueueMapBuffer((cl_command_queue)c->q, (cl_mem)m,
        /*blocking_map: */ true, mapping, offset, bytes, 0, null, null, &r);
//  if (a == null) { println("WARGNING: %s", ocl.error(r)); }
    return a;
}

static void  ocl_unmap(ocl_context_t* c, ocl_memory_t m, const void* a) {
    call(clEnqueueUnmapMemObject((cl_command_queue)c->q, (cl_mem)m, (void*)a,
        0, null, null));
}

static ocl_shared_t ocl_alloc_shared(ocl_context_t* c, int access, size_t bytes) {
    ocl_shared_t s = {
        .access = access,
        .bytes = bytes,
        .c = c
    };
    cl_int r = 0;
    s.a = clSVMAlloc(c->c, access, bytes, sizeof(uint64_t) * 16);
    if (s.a != null) {
        s.m = clCreateBuffer(c->c, access|CL_MEM_USE_HOST_PTR, bytes, s.a, &r);
        fatal_if(s.m == null || r != 0, "%s", ocl.error(r));
    }
    return s;
}

static int ocl_access_to_map(int access) {
    int map = 0;
    if (access & CL_MEM_READ_ONLY)  {
        map |= CL_MAP_READ;
    } else if (access & CL_MEM_READ_WRITE) {
        map |= CL_MAP_READ|CL_MAP_WRITE;
    } else if (access & CL_MEM_WRITE_ONLY) {
        map |= CL_MAP_WRITE_INVALIDATE_REGION;
    } else {
        fatal_if("invalid access", "%d", access);
    }
    return map;
}

static void* ocl_map_shared(ocl_shared_t* s) {
    int map = 0;
    if (s->access & CL_MEM_READ_ONLY)  { map |= CL_MAP_READ; }
    if (s->access & CL_MEM_READ_WRITE) { map |= CL_MAP_READ|CL_MAP_WRITE; }
    if (s->access & CL_MEM_WRITE_ONLY) { map |= CL_MAP_WRITE_INVALIDATE_REGION; }
    //                   blocking:                 wait_list:    done:
    call(clEnqueueSVMMap(s->c->q, true, map, s->a, s->bytes, 0, null, null));
    // TODO: do we need double mapping. I think we do NOT!!!
//  void* a = ocl.map(s->c, mapping, s->m, 0, s->bytes);
//  fatal_if(a != s->a, "expected to be the same s->a: %p a: %p", s->a, a);
    s->p = s->a;
    return s->p;
}

static void ocl_unmap_shared(ocl_shared_t* s) {
    // TODO: do we need double mapping. I think we do NOT!!!
//  ocl.unmap(s->c, s->m, s->a);
    call(clEnqueueSVMUnmap(s->c->q, s->a, 0, null, null));
    s->p = null;
}

static void ocl_free_shared(ocl_shared_t* s) {
    if (s->m != null) { call(clReleaseMemObject(s->m)); }
    if (s->a != null) { clSVMFree(s->c->c, s->a); }
    memset(s, 0, sizeof(*s));
}

static ocl_program_t ocl_compile(ocl_context_t* c,
        const char* code, size_t bytes, const char* options,
        char log[], int64_t log_capacity) {
    cl_int r = 0;
    cl_program p = clCreateProgramWithSource(c->c, 1, &code, &bytes, &r);
    not_null(p, r);
    // Build the program
    cl_device_id device_id = (cl_device_id)ocl.devices[c->ix].id;
    r = clBuildProgram(p, 1, &device_id, options, /*notify:*/ null, // sync
        /* user_data: */null);
    if (r != 0) {
        char log_on_stack[16 * 1024];
        if (log == null || log_capacity <= 0) {
            log_capacity = countof(log_on_stack);
            log = log_on_stack;
        }
        log[0] = 0;
        int e = 0; // result of clGetProgramBuildInfo()
        if (log_capacity > 1024) {
            char* s = log;
            int64_t k = snprintf(s, log_capacity,
                "clBuildProgram() failed %s\n", ocl.error(r));
            s += k; log_capacity -= k;
            e = clGetProgramBuildInfo(p, device_id, CL_PROGRAM_BUILD_LOG,
                log_capacity, s, null);
            k = strlen(s);
            s += k; log_capacity -= k;
            if (e != 0 && log_capacity > 128) {
                k = strlen(log);
                snprintf(s, log_capacity - k,
                    "clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG) "
                    "failed %s\n", ocl.error(e));
            }
        }
        if (log == log_on_stack) {
            println("%s", log);
            if (e != 0) {
                println("WARNING: clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG) "
                        "failed %s", ocl.error(e));
            }
            fatal_if(clBuildProgram, "clBuildProgram() failed %s", ocl.error(r));
        }
        call(clReleaseProgram((cl_program)p));
        p = null; // no reason to hold on to the program that did not build
    }
    return (ocl_program_t)p;
}

static void ocl_release_program(ocl_program_t p) {
    call(clReleaseProgram((cl_program)p));
}

static ocl_kernel_t ocl_create_kernel(ocl_program_t p, const char* name) {
    cl_int r = 0;
    cl_kernel k = clCreateKernel((cl_program)p, name, &r);
    not_null(k, r);
    return (ocl_kernel_t)k;
}

static ocl_event_t ocl_enqueue_args(ocl_context_t* c,
        ocl_kernel_t k, int64_t n, int argc, ocl_arg_t argv[]) {
    assert(n > 0);
    for (int i = 0; i < argc; i++) {
        call(clSetKernelArg((cl_kernel)k, i, argv[i].bytes, argv[i].p));
    }
    size_t global_work_size = n;
    cl_event done = null;
    call(clEnqueueNDRangeKernel((cl_command_queue)c->q, (cl_kernel)k,
            1, null, &global_work_size, null, 0, null, &done));
    return (ocl_event_t)done;
}

static ocl_event_t ocl_enqueue(ocl_context_t* c, ocl_kernel_t k, int64_t n,
        ...) {
    va_list vl;
    va_start(vl, n);
    int argc = 0;
    for (;;) {
        void* p = va_arg(vl, void*);
        size_t bytes = va_arg(vl, size_t);
        if (p == null && bytes == 0) { break; }
        argc++;
    }
    va_end(vl);
    typedef struct { void* p; size_t bytes; } argv_t;
    ocl_arg_t* argv = (ocl_arg_t*)alloca(sizeof(argv_t) * argc);
    va_start(vl, n);
    for (int i = 0; i < argc; i++) {
        argv[i].p = va_arg(vl, void*);
        argv[i].bytes = va_arg(vl, size_t);
    }
    va_end(vl);
    return ocl.enqueue_args(c, k, n, argc, argv);
}

static ocl_profiling_t* ocl_profile_add(ocl_context_t* c, ocl_event_t e) {
    fatal_if(!ocl.is_profiling(c));
    fatal_if(c->ov->profiling_count == c->ov->max_profiling_count,
            "profiling[%lld] is too small", c->ov->max_profiling_count);
    ocl_profiling_t* p = &c->ov->profiling[c->ov->profiling_count++];
    memset(p, 0, sizeof(*p));
    ocl.retain_event(e); // increment reference count
    p->e = e;
    return p;
}

static void ocl_profile(ocl_profiling_t* p) {
    #pragma push_macro("get_info")
    #define get_info(n, v) do {                                                \
        call(clGetEventProfilingInfo((cl_event)p->e, n, sizeof(v), &v, null)); \
    } while (0)
    get_info(CL_PROFILING_COMMAND_QUEUED, p->queued);
    get_info(CL_PROFILING_COMMAND_SUBMIT, p->submit);
    get_info(CL_PROFILING_COMMAND_START, p->start);
    get_info(CL_PROFILING_COMMAND_END, p->end);
    #pragma pop_macro("get_info")
    static double sum;
    p->time = (p->end - p->start) / (double)NSEC_IN_SEC;
    if (p->count != 0) {
        double seconds_per_kernel = p->time / p->count;
        double invocations_per_second = 1.0 / seconds_per_kernel;
        double gops = invocations_per_second / (1000 * 1000 * 1000);
        p->gflops = p->fops * gops;
        p->g32ops = p->i32ops * gops;
        p->g64ops = p->i64ops * gops;
    }
    ocl.release_event(p->e); // decrement reference count
    p->e = null;
}

static void ocl_wait(ocl_event_t* events, int count) {
    call(clWaitForEvents(count, (cl_event*)events));
}

static void ocl_retain_event(ocl_event_t e) {
    call(clRetainEvent((cl_event)e));
}

static void ocl_release_event(ocl_event_t e) {
    call(clReleaseEvent((cl_event)e));
}

static void ocl_release_kernel(ocl_kernel_t k) {
    call(clReleaseKernel((cl_kernel)k));
}

static void ocl_kernel_info(ocl_context_t* c, ocl_kernel_t kernel,
        ocl_kernel_info_t* info) {
    cl_kernel k = (cl_kernel)kernel;
    cl_device_id device_id = (cl_device_id)ocl.devices[c->ix].id;
    #pragma push_macro("get_val")
    #define get_val(n, v) do { \
        call(clGetKernelWorkGroupInfo(k, device_id, n, sizeof(v), &v, null)); \
    } while (0)
    get_val(CL_KERNEL_WORK_GROUP_SIZE, info->work_group);
    get_val(CL_KERNEL_LOCAL_MEM_SIZE, info->local_memory);
    get_val(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        info->preferred_work_group_multiple);
    get_val(CL_KERNEL_PRIVATE_MEM_SIZE, info->private_mem_size);
    get_val(CL_KERNEL_GLOBAL_WORK_SIZE, info->global_work_size);
    #pragma pop_macro("get_val")
}

static void ocl_close(ocl_context_t* c) {
    ocl_dispose_queue(c);
    call(clReleaseContext((cl_context)c->c));
    c->c = null;
}

static const char* ocl_error(int r) {
    static char error[128];
    #define case_(x) case x: snprintf(error, countof(error), "%d " #x, r); break
    switch (r) {
        case_(CL_DEVICE_NOT_FOUND);
        case_(CL_DEVICE_NOT_AVAILABLE);
        case_(CL_COMPILER_NOT_AVAILABLE);
        case_(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        case_(CL_OUT_OF_RESOURCES);
        case_(CL_OUT_OF_HOST_MEMORY);
        case_(CL_PROFILING_INFO_NOT_AVAILABLE);
        case_(CL_MEM_COPY_OVERLAP);
        case_(CL_IMAGE_FORMAT_MISMATCH);
        case_(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        case_(CL_BUILD_PROGRAM_FAILURE);
        case_(CL_MAP_FAILURE);
        case_(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        case_(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        case_(CL_COMPILE_PROGRAM_FAILURE);
        case_(CL_LINKER_NOT_AVAILABLE);
        case_(CL_LINK_PROGRAM_FAILURE);
        case_(CL_DEVICE_PARTITION_FAILED);
        case_(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        case_(CL_INVALID_VALUE);
        case_(CL_INVALID_DEVICE_TYPE);
        case_(CL_INVALID_PLATFORM);
        case_(CL_INVALID_DEVICE);
        case_(CL_INVALID_CONTEXT);
        case_(CL_INVALID_QUEUE_PROPERTIES);
        case_(CL_INVALID_COMMAND_QUEUE);
        case_(CL_INVALID_HOST_PTR);
        case_(CL_INVALID_MEM_OBJECT);
        case_(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        case_(CL_INVALID_IMAGE_SIZE);
        case_(CL_INVALID_SAMPLER);
        case_(CL_INVALID_BINARY);
        case_(CL_INVALID_BUILD_OPTIONS);
        case_(CL_INVALID_PROGRAM);
        case_(CL_INVALID_PROGRAM_EXECUTABLE);
        case_(CL_INVALID_KERNEL_NAME);
        case_(CL_INVALID_KERNEL_DEFINITION);
        case_(CL_INVALID_KERNEL);
        case_(CL_INVALID_ARG_INDEX);
        case_(CL_INVALID_ARG_VALUE);
        case_(CL_INVALID_ARG_SIZE);
        case_(CL_INVALID_KERNEL_ARGS);
        case_(CL_INVALID_WORK_DIMENSION);
        case_(CL_INVALID_WORK_GROUP_SIZE);
        case_(CL_INVALID_WORK_ITEM_SIZE);
        case_(CL_INVALID_GLOBAL_OFFSET);
        case_(CL_INVALID_EVENT_WAIT_LIST);
        case_(CL_INVALID_EVENT);
        case_(CL_INVALID_OPERATION);
        case_(CL_INVALID_GL_OBJECT);
        case_(CL_INVALID_BUFFER_SIZE);
        case_(CL_INVALID_MIP_LEVEL);
        case_(CL_INVALID_GLOBAL_WORK_SIZE);
        case_(CL_INVALID_PROPERTY);
        case_(CL_INVALID_IMAGE_DESCRIPTOR);
        case_(CL_INVALID_COMPILER_OPTIONS);
        case_(CL_INVALID_LINKER_OPTIONS);
        case_(CL_INVALID_DEVICE_PARTITION_COUNT);
        case_(CL_INVALID_PIPE_SIZE);
        case_(CL_INVALID_DEVICE_QUEUE);
        case_(CL_INVALID_SPEC_ID);
        case_(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
        default: snprintf(error, countof(error), "%d Unknown error", r);
    }
    error[countof(error) - 1] = 0;
    return error;
}

static void ocl_check_fp16_support(int dix) {
    ocl_context_t c = ocl.open(dix, null);
    static const char* sc = // .cl source code
    "#pragma OPENCL EXTENSION cl_khr_fp16: enable          \n"
    "                                                      \n"
    "#define gcfp16p_t __global const half*                \n"
    "#define gfp32p_t  __global float*                     \n"
    "                                                      \n"
    "__kernel                                              \n"
    "void mul_fp16(gcfp16p_t x, gcfp16p_t y, gfp32p_t r) { \n"
    "    *r = vload_half(0, x) * vload_half(0, y);         \n"
    "}                                                     \n"
    "                                                      \n"
    "__kernel                                              \n"
    "void dot_fp16(gcfp16p_t x, gcfp16p_t y, gfp32p_t r) { \n"
    "    *r = dot(vload_half4(0, x), vload_half4(0, y));   \n"
    "}";
    char log[16 * 1024];
    ocl_program_t p = ocl.compile(&c, sc, strlen(sc), null, log, countof(log));
    bool b = p != null;
    if (b) {
        ocl.devices[dix].fp16_config |= CL_FP_DENORM; // assume at least that
        ocl.release_program(p);
    } else {
//      println("code: ---------\n%s\nlog: ---------\n%s\n", sc, log);
    }
    ocl.close(&c);
}

static cl_platform_id cl_platforms[16];
static cl_uint cl_platform_count;

static void ocl_enumerate_devices() {
    #pragma push_macro("get_str")
    #pragma push_macro("get_val")
    #pragma push_macro("get_opt")
    #pragma push_macro("ext")
    #define get_str(name, s) do {                             \
        call(clGetDeviceInfo(id, name, countof(s), s, null)); \
    } while (0)
    #define get_val(name, v) do {                             \
        call(clGetDeviceInfo(id, name, sizeof(v), &v, null)); \
    } while (0)
    #define get_opt(name, v) /*optional*/                     \
        (clGetDeviceInfo(id, name, sizeof(v), &v, null) == 0)
    #define ext(s) (strstr(d->extensions, (s)) != null)
    // Get device information
    for (cl_uint i = 0; i < cl_platform_count; i++) {
        cl_device_id device_ids[16] = {0};
        cl_uint devids_count = 0;
	    if (clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, countof(device_ids),
                device_ids, &devids_count) == 0) {
            for (cl_uint j = 0; j < devids_count; j++) {
                ocl_device_t* d = &ocl.devices[ocl.count];
                cl_device_id id = device_ids[j];
                d->id = (ocl_device_id_t)id;
                d->platform = cl_platforms[i];
                get_str(CL_DEVICE_NAME, d->name);
                get_str(CL_DEVICE_VENDOR, d->vendor);
                char text[4096];
                get_str(CL_DEVICE_VERSION, text); // e.g. "OpenCL 3.0 CUDA"
                int minor = 0; // sscanf wants type "int" not "int32_t"
                int major = 0;
                call(sscanf(text, "OpenCL %d.%d", &major, &minor) != 2);
                d->version_major = major;
                d->version_minor = minor;
                get_str(CL_DEVICE_OPENCL_C_VERSION, text);
                call(sscanf(text, "OpenCL C %d.%d", &major, &minor) != 2);
                d->c_version_major = major;
                d->c_version_minor = minor;
                get_str(CL_DEVICE_EXTENSIONS, d->extensions);
                get_val(CL_DEVICE_MAX_CLOCK_FREQUENCY,       d->clock_frequency);
                get_val(CL_DEVICE_ADDRESS_BITS,              d->address_bits);
                get_val(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,     d->global_cache);
                get_val(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, d->global_cacheline);
                get_val(CL_DEVICE_GLOBAL_MEM_SIZE,           d->global_memory);
                get_val(CL_DEVICE_LOCAL_MEM_SIZE,            d->local_memory);
                get_val(CL_DEVICE_MAX_CONSTANT_ARGS,         d->max_const_args);
                get_val(CL_DEVICE_MAX_COMPUTE_UNITS,         d->compute_units);
                get_val(CL_DEVICE_MAX_WORK_GROUP_SIZE,       d->max_groups);
                get_val(CL_DEVICE_SINGLE_FP_CONFIG,          d->fp32_config);
                get_val(CL_DEVICE_DOUBLE_FP_CONFIG,          d->fp64_config);
                get_val(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,  d->dimensions);
                // Intel extensions
                get_opt(CL_DEVICE_MAX_NUM_SUB_GROUPS,        d->max_subgroups);
                get_opt(CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
                                                             d->subgroup_ifp);
                call(d->dimensions > countof(d->max_items));
                get_val(CL_DEVICE_MAX_WORK_ITEM_SIZES, d->max_items);
                d->flavor = 0;
                d->flavor |= ext("_intel_") ? ocl_intel  : 0;
                d->flavor |= ext("_nv_")    ? ocl_nvidia : 0;
                d->flavor |= ext("_amd_")   ? ocl_amd    : 0;
                bool has_fp16_config =
                    get_opt(CL_DEVICE_HALF_FP_CONFIG, d->fp16_config);
                ocl.count++;
                if (!has_fp16_config || d->fp16_config == 0) {
                    ocl_check_fp16_support(ocl.count - 1);
                }
            }
        }
    }
    #pragma pop_macro("ext")
    #pragma pop_macro("get_opt")
    #pragma pop_macro("get_val")
    #pragma pop_macro("get_str")
}

static void ocl_init(void) {
    // Get platforms ids
    cl_platform_count = countof(cl_platforms);
    int r = clGetPlatformIDs(countof(cl_platforms), cl_platforms, &cl_platform_count);
	if (r == 0) {
        ocl_enumerate_devices();
    } else {
        println("clGetPlatformIDs() failed: %s", ocl.error(r));
    }
}

// Intel(R) UHD Graphics does not support ocl_fp64
// manifesting it by saying
//    "use of type 'double' requires cl_khr_fp64 extension to be enabled"
// while NOT having 'cl_khr_fp64' among it's extensions
// but double_fp_config == 0
//
// Mass confusion in the cl_khr_fp* use and abuse of reporting and enablement:
//
// Since OpenCL C 1.1 it is NOT required to enable cl_khr_fp64 extension
// to use "double" type in .cl code.
// However some Intel CPU/Integrated Graphics GPU silicon
// does not implement double.
// OpenCL is not clear on reporting the device float/double
// capabilities and clang .cl compiler struggles with it.
// Real problem is the cl_khr_* are [ab]used for both reporting
// and enablement/disablement which is muddy.
// For now both cl_khr_fp64 and cl_intel_accelerator are checked
// until OpenCL achieve clarity on #pragma extension
// enabling/disabling and reporting
// https://github.com/KhronosGroup/OpenCL-Docs/issues/82
// https://github.com/KhronosGroup/OpenCL-Docs/pull/355

static const char* ocl_fp_config_to_string(int64_t config) {
    static char s[1024];
    s[0] = 0;
    #pragma push_macro("append")
    #define append(text) do { strcat(s, ", " text); } while (0)
    if (config & CL_FP_DENORM)           { append("CL_FP_DENORM");           }
    if (config & CL_FP_INF_NAN)          { append("CL_FP_INF_NAN");          }
    if (config & CL_FP_ROUND_TO_NEAREST) { append("CL_FP_ROUND_TO_NEAREST"); }
    if (config & CL_FP_ROUND_TO_ZERO)    { append("CL_FP_ROUND_TO_ZERO");    }
    if (config & CL_FP_ROUND_TO_INF)     { append("CL_FP_ROUND_TO_INF");     }
    if (config & CL_FP_FMA)              { append("CL_FP_FMA");              }
    if (config & CL_FP_SOFT_FLOAT)       { append("CL_FP_SOFT_FLOAT");       }
    if (config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
        append("CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT");
    }
    #pragma pop_macro("append")
    return s[0] == 0 ? s : &s[2]; // skip first ", "
}

static void ocl_dump(int ix) {
    const ocl_device_t* d = &ocl.devices[ix];
    println("Device name:     %s OpenCL %d.%d C %d.%d", d->name,
        d->version_major, d->version_minor, d->c_version_major, d->c_version_minor);
    println("compute_units:    %lld @ %lldMHz (intptr_t %d bits)",
                                        d->compute_units,
                                        d->clock_frequency, (int)d->address_bits);
    println("global_cache:     %lldMB", d->global_cache / MB);
    println("global_cacheline: %lld",   d->global_cacheline);
    println("global_memory:    %lldMB", d->global_memory / MB);
    println("local_memory:     %lld bytes", d->local_memory);
    println("max_const_args:   %lld", d->max_const_args);
    println("max_groups:       %lld", d->max_groups);
    println("max_subgroups:    %lld", d->max_subgroups);

    println("subgroup_ifp:     %lld", d->subgroup_ifp);
    println("dimensions:       %lld", d->dimensions);
    const int64_t* wi = d->max_items;
    println("max_items[]:     {%lld %lld %lld}", wi[0], wi[1], wi[2]);
    println("fp16_config:      %s", ocl_fp_config_to_string(d->fp16_config));
    println("fp32_config:      %s", ocl_fp_config_to_string(d->fp32_config));
    println("fp64_config:      %s", ocl_fp_config_to_string(d->fp64_config));
    println("extensions:       %s", d->extensions);
}

// run with args: compile kernel.cl options...

static void ocl_compiler(int argc, const char* argv[]) {
    fatal_if(argc < 3);
    enum { source_max = 256 * 1024 };
    char* source = malloc(source_max);
    fatal_if(source == null);
    FILE* f = fopen(argv[2], "r");
    fatal_if(f == null, "failed to open file: %s", argv[2]);
    int64_t source_bytes = (int64_t)fread(source, 1, source_max, f);
    if (f != null) { fclose(f); }
    size_t optc = 0; // characters in options
    for (int i = 3; i < argc; i++) { optc += strlen(argv[i]) + 1; }
    char* opt = optc > 0 ? alloca(optc + 1) : null;
    if (opt != null) {
        opt[0] = 0;
        for (int i = 3; i < argc; i++) {
            strcat(opt, argv[i]);
            strcat(opt, "\x20"); // space
        }
    }
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
        int from = d->fp16_config != 0 ? ocl_fpp16 : ocl_fpp32;
        int to   = d->fp64_config == 0 ? ocl_fpp32 : ocl_fpp64;
        for (int fpp = from; fpp <= to; fpp++) {
            println("compile: %s for %s @ %s", argv[2], ocl_fpp_names[fpp], d->name);
            println("");
            ocl_program_t p = ocl.compile(&c, source, source_bytes, opt, null, 0);
            if (p == null) {
                println("failed to compile for %s: %s", ocl_fpp_names[fpp], argv[2]);
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
                    const char* fns = strrchr(argv[2], '\\'); // file name start
                    if (fns == null) { fns = strrchr(argv[2], '/'); }
                    if (fns == null) { fns = argv[2]; } else { fns++; }
                    // argv[2]="foo/bar/kernel.cl" fns == "kernel.cl"
                    const char* fne = strrchr(fns, '.');
                    if (fne == null) { fne = fns + strlen(fns); }
                    // fne == ".cl"
                    int fnc = (int)(fne - fns); // number of characters in filename
                    char fn[256]; // file name
                    snprintf(fn, countof(fn), "%.*s%d.%s.bin", fnc, fns,
                        ocl_fpp_bytes[fpp] * 8, dn);
                    f = fopen(fn, "wb");
                    fatal_if(f == null, "failed to create file: %s", fn);
                    int64_t written = (int64_t)fwrite(binary, 1, binary_sizes[0], f);
                    fatal_if(written != binary_sizes[0]);
                    fclose(f);
                    free(binary);
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
        println("failed to read: %s", argv[2]);
    }
    free(source);
}


ocl_if ocl = {
    .init = ocl_init,
    .dump = ocl_dump,
    .open = ocl_open,
    .is_profiling = ocl_is_profiling,
    .has_fpp = ocl_has_fpp,
    .error = ocl_error,
    .alloc = ocl_alloc,
    .allocate = ocl_allocate,
    .deallocate = ocl_deallocate,
    .access_to_map = ocl_access_to_map,
    .map = ocl_map,
    .unmap = ocl_unmap,
    .alloc_shared = ocl_alloc_shared,
    .map_shared = ocl_map_shared,
    .unmap_shared = ocl_unmap_shared,
    .free_shared = ocl_free_shared,
    .compile = ocl_compile,
    .create_kernel = ocl_create_kernel,
    .kernel_info = ocl_kernel_info,
    .enqueue_args = ocl_enqueue_args,
    .enqueue = ocl_enqueue,
    .wait = ocl_wait,
    .profile_add = ocl_profile_add,
    .profile = ocl_profile,
    .retain_event = ocl_retain_event,
    .release_event = ocl_release_event,
    .release_kernel = ocl_release_kernel,
    .release_program = ocl_release_program,
    .flush = ocl_flush,
    .finish = ocl_finish,
    .close = ocl_close,
    .compiler = ocl_compiler,
    .devices = ocl_devices
};


