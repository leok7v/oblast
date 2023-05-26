#include "rt.h"
#include "ocl.h"

#ifdef OCL_USE_NVIDIA_12_LIB_BINDINGS
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64\OpenCL.lib
#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64/OpenCL.lib")
#else // dynamic bindings with OpenCL.dll on the path
#include <CL/cl_bind.inc> // dynamically bind everything
#endif

static_assert(ocl_fpp16 == 0 && ocl_fpp32 == 1 && ocl_fpp64 == 2, "order");

const char* ocl_fpp_names[3] = {"fp16", "fp32", "fp64"};

const int ocl_fpp_bytes[3] = {
    (int)sizeof(fp16_t), (int)sizeof(fp32_t), (int)sizeof(fp64_t)
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
    traceln("ERROR: %*s", errinfo);
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
    ocl_context_t c;
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
    if (ov != null) {
        ov->max_groups_restore = d->max_groups;
        ov->max_items_restore  = d->max_items[0];
        if (0 < ov->max_groups && ov->max_groups < d->max_groups) {
            d->max_groups = ov->max_groups;
        }
        if (0 < ov->max_items && ov->max_items < d->max_items[0]) {
            d->max_items[0] = ov->max_items;
        }
    }
    return c;
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

static ocl_memory_t ocl_allocate(ocl_context_t* c, int access, size_t bytes) {
    cl_int r = 0;
    cl_mem m = clCreateBuffer(c->c, access|CL_MEM_ALLOC_HOST_PTR, bytes, null, &r);
    not_null(m, r);
    return (ocl_memory_t)m;
}

static void  ocl_deallocate(ocl_memory_t m) {
    call(clReleaseMemObject((cl_mem)m));
}

static void* ocl_map(ocl_context_t* c, int mapping, ocl_memory_t m, size_t offset,
        size_t bytes) {
    cl_int r = 0;
    // blocking_map: true sync mapping
    void* a = clEnqueueMapBuffer((cl_command_queue)c->q, (cl_mem)m,
        /*blocking_map: */ true, mapping, offset, bytes, 0, null, null, &r);
    not_null(a, r);
    return a;
}

static void  ocl_unmap(ocl_context_t* c, ocl_memory_t m, const void* a) {
    call(clEnqueueUnmapMemObject((cl_command_queue)c->q, (cl_mem)m, (void*)a,
        0, null, null));
}

static ocl_program_t ocl_compile_program(ocl_context_t* c,
        const char* code, size_t bytes, const char* options) {
    cl_int r = 0;
    cl_program p = clCreateProgramWithSource(c->c, 1, &code, &bytes, &r);
    not_null(p, r);
    // Build the program
    cl_device_id device_id = (cl_device_id)ocl.devices[c->ix].id;
    r = clBuildProgram(p, 1, &device_id, options, /*notify:*/ null, // sync
        /* user_data: */null);
    if (r != 0) {
        static char log[16 * 1024];
        log[0] = 0;
        // clGetProgramBuildInfo() returns invalid param if compiler crash
        (void)clGetProgramBuildInfo(p, device_id, CL_PROGRAM_BUILD_LOG,
            countof(log), log, null); // ignore errors
        traceln("%s", log);
    }
    fatal_if(r != 0, "clBuildProgram() failed %s", ocl.error(r));
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

static ocl_event_t ocl_enqueue_range_kernel(ocl_context_t* c,
        ocl_kernel_t k, size_t groups, size_t items_per_group,
        int argc, ocl_arg_t argv[]) {
    for (int i = 0; i < argc; i++) {
        call(clSetKernelArg((cl_kernel)k, i, argv[i].bytes, argv[i].p));
    }
    cl_event completion = null;
    size_t total = groups * items_per_group;
    ocl_device_t* d = &ocl.devices[c->ix]; (void)d;
    assert((int64_t)groups <= d->max_groups);
    assert((int64_t)items_per_group <= d->max_items[0]);
    call(clEnqueueNDRangeKernel((cl_command_queue)c->q, (cl_kernel)k,
            1, null, &total, &items_per_group, 0, null, &completion));
    return (ocl_event_t)completion;
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
//  sum += p->time;
//  traceln("time: %.6fus sum: %.6fus", p->time * USEC_IN_SEC, sum * USEC_IN_SEC);
    if (p->count != 0) {
        double seconds_per_kernel = p->time / p->count;
        double invocations_per_second = 1.0 / seconds_per_kernel;
        double gops = invocations_per_second / (1000 * 1000 * 1000);
        p->gflops = p->fops * gops;
        p->g32ops = p->i32ops * gops;
        p->g64ops = p->i64ops * gops;
    }
    ocl.release_event(p->e); // decrement refernce count
    p->e = null;

//  uint64_t ema_samples = p->ema_samples == 0 ? 128 : p->ema_samples;
//  // exponential moving average of 128 samples
//  const double ema_alpha = 1.0 / (double)ema_samples;
//  p->time = (p->end - p->start) / (double)NSEC_IN_SEC;
//  if (items != 0) {
//      double seconds_per_item = p->time / items;
//      double ops_per_second = 1.0 / seconds_per_item;
//      p->gops = ops_per_second / (1000 * 1000 * 1000);
//      p->ema.gops = p->ema.gops * (1.0 - ema_alpha) + p->gops * ema_alpha;
//  } else {
//      p->gops = 0; // cannot determine for unknown number of items
//  }
//  p->ema.time = p->ema.time * (1.0 - ema_alpha) + p->time * ema_alpha;
    // client is responsible updating and calculating .user fields
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
    if (c->ov != null) {
        ocl_device_t* d = &ocl.devices[c->ix];
        d->max_groups   = c->ov->max_groups_restore;
        d->max_items[0] = c->ov->max_items_restore;
    }
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

// TODO: need two test kernels: one for half (including half4 and half16) another for double
//       because both features are completely misreported by AMD, Intel and NVIDIA
//       or at least there is no accurate way to report them that is found
//       in OpenCL specification (I did not find it, ChatGPT didn't find it,
//       GPU f/w engineers who implemented OpenCL bindings clearly didn't find it too).
//       Take sample kernel from add.c

static void ocl_init(void) {
    #pragma push_macro("get_str")
    #pragma push_macro("get_val")
    #pragma push_macro("ext")
    #define get_str(name, s) do { \
        call(clGetDeviceInfo(id, name, countof(s), s, null)); \
    } while (0)
    #define get_val(name, v) do { \
        call(clGetDeviceInfo(id, name, sizeof(v), &v, null)); \
    } while (0)
    #define ext(s) (strstr(d->extensions, (s)) != null)
    // Get platform and device information
    cl_platform_id platforms[16] = {0};
    cl_uint platform_count = countof(platforms);
	call(clGetPlatformIDs(countof(platforms), platforms, &platform_count));
    for (cl_uint i = 0; i < platform_count; i++) {
        cl_device_id device_ids[16] = {0};
        cl_uint devids_count = 0;
	    if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, countof(device_ids),
                device_ids, &devids_count) == 0) {
            for (cl_uint j = 0; j < devids_count; j++) {
                ocl_device_t* d = &ocl.devices[ocl.count];
                cl_device_id id = device_ids[j];
                d->id = (ocl_device_id_t)id;
                d->platform = platforms[i];
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
                get_val(CL_DEVICE_MAX_CLOCK_FREQUENCY,      d->clock_frequency);
                get_val(CL_DEVICE_GLOBAL_MEM_SIZE,          d->global_memory);
                get_val(CL_DEVICE_LOCAL_MEM_SIZE,           d->local_memory);
                get_val(CL_DEVICE_MAX_CONSTANT_ARGS,        d->max_const_args);
                get_val(CL_DEVICE_MAX_COMPUTE_UNITS,        d->compute_units);
                get_val(CL_DEVICE_MAX_WORK_GROUP_SIZE,      d->max_groups);
                get_val(CL_DEVICE_DOUBLE_FP_CONFIG,         d->double_fp_config);
                get_val(CL_DEVICE_SINGLE_FP_CONFIG,         d->float_fp_config);
                get_val(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, d->dimensions);
                call(d->dimensions > countof(d->max_items));
                get_val(CL_DEVICE_MAX_WORK_ITEM_SIZES, d->max_items);
                d->fp_config = 0;
                d->fp_config |= ext("cl_khr_fp64") ? ocl_fp64 : 0;
                d->fp_config |= ext("cl_khr_fp16") ? ocl_fp16 : 0;
                d->flavor = 0;
                d->flavor |= ext("_intel_") ? ocl_intel  : 0;
                d->flavor |= ext("_nv_")    ? ocl_nvidia : 0;
                d->flavor |= ext("_amd_")   ? ocl_amd    : 0;
                if ((d->fp_config & ocl_fp16) == 0) {
                    // NVIDIA does not report cl_khr_fp16 extension
                    d->fp_config |= ocl_fp16; // but supports it
                }
//              if (ext("cl_intel_accelerator")) { // see note: ***
//                  d->fp_config &= ~ocl_fp64;
//              }
                ocl.count++;
            }
        }
    }
    #pragma pop_macro("ext")
    #pragma pop_macro("get_val")
    #pragma pop_macro("get_str")
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
// until OpenCL achive clarity on #pragma extension
// enabling/disabling and reporting
// https://github.com/KhronosGroup/OpenCL-Docs/issues/82
// https://github.com/KhronosGroup/OpenCL-Docs/pull/355

static const char* ocl_fp_config_to_string(int64_t config) {
    static char s[1024];
    s[0] = 0;
    #pragma push_macro("append")
    #define append(text) do { strcat(s, ", " text); } while (0)
    if (config & ocl_fp_denorm)           { append("ocl_fp_denorm");           }
    if (config & ocl_fp_inf_nan)          { append("ocl_fp_inf_nan");          }
    if (config & ocl_fp_round_to_nearest) { append("ocl_fp_round_to_nearest"); }
    if (config & ocl_fp_round_to_zero)    { append("ocl_fp_round_to_zero");    }
    if (config & ocl_fp_round_to_inf)     { append("ocl_fp_round_to_inf");     }
    if (config & ocl_fp_fma)              { append("ocl_fp_fma");              }
    if (config & ocl_fp_soft_float)       { append("ocl_fp_soft_float");       }
    if (config & ocl_fp_correctly_rounded_divide_sqrt) {
        append("ocl_fp_correctly_rounded_divide_sqrt");
    }
    #pragma pop_macro("append")
    return s;
}

static void ocl_dump(int ix) {
    const ocl_device_t* d = &ocl.devices[ix];
    traceln("Device name:     %s OpenCL %d.%d C %d.%d", d->name,
        d->version_major, d->version_minor, d->c_version_major, d->c_version_minor);
    traceln("compute_units:    %lld @ %lldMHz", d->compute_units, d->clock_frequency);
    traceln("global_memory:    %lldMB", d->global_memory / MB);
    traceln("local_memory:     %lld bytes", d->local_memory);
    traceln("max_const_args:   %lld", d->max_const_args);
    traceln("max_groups:       %lld", d->max_groups);
    traceln("dimensions:       %lld", d->dimensions);
    const int64_t* wi = d->max_items;
    traceln("max_items[]:     {%lld %lld %lld}", wi[0], wi[1], wi[2]);
    traceln("float_fp_config:  %s", ocl_fp_config_to_string(d->float_fp_config));
    traceln("double_fp_config: %s", ocl_fp_config_to_string(d->double_fp_config));
    traceln("extensions:       %s", d->extensions);
}

ocl_if ocl = {
    .init = ocl_init,
    .dump = ocl_dump,
    .open = ocl_open,
    .is_profiling = ocl_is_profiling,
    .error = ocl_error,
    .allocate = ocl_allocate,
    .deallocate = ocl_deallocate,
    .map = ocl_map,
    .unmap = ocl_unmap,
    .compile_program = ocl_compile_program,
    .create_kernel = ocl_create_kernel,
    .kernel_info = ocl_kernel_info,
    .enqueue_range_kernel = ocl_enqueue_range_kernel,
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
    .devices = ocl_devices
};

// C:\Windows\System32\OpenCL.dll
// File description OpenCL Client DLL
// Type Application extension
// File version 3.0.3.0
// Productname Khronos OpenCL ICD Loader
// Product version 3.0.3.0
// Copyright Copyright © The Khronos Group Inc 2016-2023
// Date modified 1/20/2023 3:20 PM
// Size: 1.41 MB (1,487,336 bytes)
// Signed by
// NVIDIA Sunday, January 15, 2023 11:18:06 AM
// and
// Microsoft Friday, January 20, 023 4:20:45 PM

#define OpenCL_dll "C:/Windows/System32/OpenCL.dll"

// TODO: alternatives for Intel only GPU and AMD GPU? Do we need them?
// e.g.:
// C:\Windows\system32\Intel_OpenCL_ICD64.dll
// C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_9eaeaf7bfb6c744b
// #define OpenCL_dll "Intel_OpenCL_ICD64.dll"

static void* ocl_dl; // OpenCL dynamic library

void* clBindFunction(const char* name) {
    static bool inititialized;
    if (!inititialized) {
        ocl_dl = load_dl(OpenCL_dll);
        // if not found, try anywheere else on the path:
        if (ocl_dl == null) { ocl_dl = load_dl(OpenCL_dll); }
        inititialized = true;
    }
    return ocl_dl != null ? find_symbol(ocl_dl, name) : null;
}

static_assertion(CL_FP_DENORM                        == ocl_fp_denorm);
static_assertion(CL_FP_INF_NAN                       == ocl_fp_inf_nan);
static_assertion(CL_FP_ROUND_TO_NEAREST              == ocl_fp_round_to_nearest);
static_assertion(CL_FP_ROUND_TO_ZERO                 == ocl_fp_round_to_zero);
static_assertion(CL_FP_ROUND_TO_INF                  == ocl_fp_round_to_inf);
static_assertion(CL_FP_FMA                           == ocl_fp_fma);
static_assertion(CL_FP_SOFT_FLOAT                    == ocl_fp_soft_float);
static_assertion(CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT == ocl_fp_correctly_rounded_divide_sqrt);

static_assertion(CL_MEM_READ_WRITE == ocl_allocate_rw);
static_assertion(CL_MEM_WRITE_ONLY == ocl_allocate_write);
static_assertion(CL_MEM_READ_ONLY  == ocl_allocate_read);

static_assertion(CL_MAP_READ == ocl_map_read);
static_assertion((CL_MAP_WRITE|CL_MAP_READ) == ocl_map_rw);
static_assertion(CL_MAP_WRITE_INVALIDATE_REGION == ocl_map_write);

