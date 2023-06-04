#pragma once
#include <stdint.h>
#include <stdbool.h>
// ocl.h exposes opencl.h and by design is just set of
// convenience fail fast wrappers to make code more
// readable and easier to maintain. It does NOT `abstract`
// OpenCL API
#include <CL/opencl.h>

#ifdef __cplusplus
extern "C" {
#endif

// float point precision index
enum {
    ocl_fpp_first = 0,
    ocl_fpp16 = 0, ocl_fpp32 = 1, ocl_fpp64 = 2, ocl_bfp16 = 3,
    ocl_fpp_last = 3
};

extern const char* ocl_fpp_names[4]; // "fp16", "fp32", "fp64", "bf16"
extern const int   ocl_fpp_bytes[4]; // { 2, 4, 8, 2 }

typedef cl_device_id ocl_device_id_t;
typedef cl_mem       ocl_memory_t;
typedef cl_program   ocl_program_t;
typedef cl_kernel    ocl_kernel_t;
typedef cl_event     ocl_event_t;

enum { // flavor (bitset because of collaboration and mixed solutions)
    ocl_nvidia    = (1 << 0),
    ocl_amd       = (1 << 1),
    ocl_intel     = (1 << 2),
    ocl_apple     = (1 << 3),
    ocl_adreno    = (1 << 4), // Qualcomm
    ocl_videocore = (1 << 5), // Broadcom
    ocl_powervr   = (1 << 6), // IBM
    ocl_vivante   = (1 << 7), // Imagination
    ocl_mali      = (1 << 8)  // ARM
    // to be continued...
};

// fp16_config, fp32_config, fp64_config bits
// CL_FP_DENORM .. CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT

// __kernel can use
// #pragma OPENCL SELECT_ROUNDING_MODE rte // rte rtz rtp rtn
// and
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
// #pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef struct ocl_device_s {
    void* platform;
    ocl_device_id_t id; // device id
    char  name[128];
    char  vendor[128];
    int32_t version_major;    // OpenCL version
    int32_t version_minor;
    int32_t c_version_major;  // OpenCL kernel .cl C language version
    int32_t c_version_minor;  // see: note ** below
    int64_t clock_frequency;  // MHz
    int64_t address_bits;     // 32 or 64 for uintptr_t and intptr_t
    int64_t global_cache;     // size in bytes
    int64_t global_cacheline;
    int64_t global_memory;
    int64_t local_memory;
    int64_t max_const_args;   // maximum number of constant args
    int64_t compute_units;    // max compute units, see: *** below
    int64_t max_groups;       // max number of work groups, see: ** below
    int64_t max_subgroups;    // max number of subgroups
    int64_t dimensions;       // dimensionality of work items
    int64_t max_items[3];     // max work items in a group per dimension
    int32_t flavor;           // GPU manufacturer - tricky, could be a mix
    int64_t fp16_config;
    int64_t fp32_config;
    int64_t fp64_config;
    int64_t subgroup_ifp;     // bool: independent forward progress
    char    extensions[4096]; // use strstr(extensions, "cl_khr_fp16")
} ocl_device_t;

// ** confusion between OpenCL devices and CL_C versions:
// https://stackoverflow.com/questions/67371938/nvidia-opencl-device-version

// *** NVIDIA GeForce RTX 3080 Laptop reports 48 units, but
// These units are often referred to as "compute cores", or
// "streaming multiprocessors" (SMs) or "CUDA cores"
// For the GeForce RTX 3080, the CUDA core count is 8704.

typedef struct ocl_kernel_info_s { // CL_KERNEL_*
    int64_t work_group;      // max kernel work group size
    int64_t compile_work_group;
    int64_t local_memory;
    int64_t preferred_work_group_multiple;
    int64_t private_mem_size;
    int64_t global_work_size;
} ocl_kernel_info_t;

// ** work groups and items:
// https://stackoverflow.com/questions/62236072/understanding-cl-device-max-work-group-size-limit-opencl
// https://registry.khronos.org/OpenCL/sdk/2.2/docs/man/html/clEnqueueNDRangeKernel.html
// usage of "size" "max" is confusing in OpenCL docs this avoided here

typedef struct ocl_profiling_s {
    ocl_event_t e;
    uint64_t queued; // in nanoseconds
    uint64_t submit; // in nanoseconds
    uint64_t start;  // in nanoseconds
    uint64_t end;    // in nanoseconds
    uint64_t count;  // number of time kernel was invoked
    uint64_t i32ops; // guestimate of int32_t operations per kernel
    uint64_t i64ops; // guestimate of int64_t operations per kernel
    uint64_t fops;   // guestimate of fpXX_t  operations per kernel
    // derivatives:
    double  time; // seconds: end - start
    double  gflops; // GFlops 1,000,000,000 float point operations
    double  g32ops; // Giga int32 ops
    double  g64ops; // Giga int64 ops
    double  user; // seconds: host time (to be filled by client)
} ocl_profiling_t;

typedef struct ocl_override_s {
    ocl_profiling_t* profiling;  // null - no profiling
    int64_t max_profiling_count; // number of elements in profiling[] array
    int64_t profiling_count;     // number of profiled kernel invocation
} ocl_override_t;

typedef struct ocl_context_s {
    int32_t ix; // device index
    void*   c; // OpenCL context
    void*   q; // OpenCL command queue
    ocl_override_t* ov;
} ocl_context_t;

typedef struct ocl_arg_s {
    void* p;
    size_t bytes;
} ocl_arg_t;

// If client need anything more complex from host/device shared memory
// model it can use direct clAPI calls:

typedef struct ocl_shared_s {  // TODO: if double mapping is not necessary delete .p
    void* p;        // only valid between map_shared()/unmap_shared()
    void* a;        // allocated host address or null if clSVMAlloc() failed
    ocl_memory_t m; // OpenCL memory handle with CL_MEM_USE_HOST_PTR
    ocl_context_t* c;
    int64_t  bytes;
    int32_t access; // CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
} ocl_shared_t;

// alloc/allocate/alloc_shared access flags:
// CL_MEM_READ_WRITE .. CL_MEM_KERNEL_READ_AND_WRITE
// map/map_shared mapping flags
// CL_MAP_READ, CL_MAP_WRITE, CL_MAP_WRITE_INVALIDATE_REGION

// single device single queue OpenCL interface

typedef struct ocl_if {
    void (*init)(void); // initializes devices[count] array
    void (*dump)(int ix); // dumps device info
    ocl_context_t (*open)(int32_t ix, ocl_override_t* ocl_override);
    bool (*is_profiling)(ocl_context_t* c);
    bool (*has_fpp)(ocl_context_t* c, int fpp);
    // pinned memory with CL_MEM_ALLOC_HOST_PTR
    ocl_memory_t (*alloc)(ocl_context_t* c, int access, size_t bytes);
    ocl_memory_t (*allocate)(ocl_context_t* c, int access, size_t bytes);
    // alloc() may return null, allocate() fatal if null
    void (*deallocate)(ocl_memory_t m);
    // CL_MEM_WRITE_ONLY -> CL_MAP_WRITE_INVALIDATE_REGION ...
    int (*access_to_map)(int access);
    // ocl_map_read  - host will read data written by GPU
    // ocl_map_write - host will write data that GPU will read
    void* (*map)(ocl_context_t* c, int mapping, ocl_memory_t m,
        size_t offset, size_t bytes); // may return null
    // memory must be unmapped before the kernel is executed
    void (*unmap)(ocl_context_t* c, ocl_memory_t m, const void* address);
    void (*migrate)(ocl_context_t* c, ocl_memory_t m);
    void (*migrate_undefined)(ocl_context_t* c, ocl_memory_t m);
    // device/host shared memory (w/o fine-grained access/atomics)
    // alloc_shared().a and .m will be null if failed
    // experimentally NVIDIA GPU only allows 1GB mapping... :(
    ocl_shared_t (*alloc_shared)(ocl_context_t* c, int access, size_t bytes);
    void* (*map_shared)(ocl_shared_t* sm);
    void (*unmap_shared)(ocl_shared_t* sm);
    void (*migrate_shared)(ocl_shared_t* sm);
    void (*migrate_shared_undefined)(ocl_shared_t* sm);
    void (*free_shared)(ocl_shared_t* sm);
    // compile() with log != null may return null, with log == null
    // any error is fatal.
    ocl_program_t (*compile)(ocl_context_t* c, const char* code,
        size_t bytes, const char* options, char log[], int64_t log_capacity);
    ocl_kernel_t (*create_kernel)(ocl_program_t p, const char* name);
    void (*kernel_info)(ocl_context_t* c, ocl_kernel_t kernel,
        ocl_kernel_info_t* info);
    // 1-dimensional range kernel: if items_in_work_group is 0 max is used
    ocl_event_t (*enqueue)(ocl_context_t* c, ocl_kernel_t k,
        int64_t n, ...); // void*, size_t bytes, ... terminate with null, 0
    ocl_event_t (*enqueue_args)(ocl_context_t* c, ocl_kernel_t k,
        int64_t n, int argc, ocl_arg_t argv[]);
    // appends queued event to array of profiling events;
    ocl_profiling_t* (*profile_add)(ocl_context_t* c, ocl_event_t e);
    void (*wait)(ocl_event_t* events, int count);
    void (*flush)(ocl_context_t* c); // all queued command to GPU
    void (*finish)(ocl_context_t* c); // waits for all commands to finish
    // must wait(&p->e, 1) or call .finish() before calling profile(p)
    void (*profile)(ocl_profiling_t* p);
    void (*retain_event)(ocl_event_t e);  // reference counter++
    void (*release_event)(ocl_event_t e); // reference counter--
    const char* (*error)(int result);
    void  (*release_program)(ocl_program_t p);
    void  (*release_kernel)(ocl_kernel_t k);
    void (*close)(ocl_context_t* c);
    // compiler("filename.cl") produces filename.intel|nvidia.bin file
    // args: filename.cl -D option=value ....
    void (*compiler)(int argc, const char* argv[]);
    ocl_device_t* devices;
    int32_t count;
} ocl_if;

extern ocl_if ocl;

#ifdef __cplusplus
}
#endif

/*
    In OpenCL, a work-item is a single unit of work that can be executed in
  parallel by a processing element. Each work-item is assigned a unique
  identifier within its work-group. Work-groups are collections of
  work-items that are executed together on a single processing element,
  such as a GPU core.

    The total number of work-items is determined by the global work-size,
  which is specified when the kernel is launched using clEnqueueNDRangeKernel.
  The global work-size is divided into work-groups of a fixed size specified
  by the local work-size. The number of work-groups is equal to the global
  work-size divided by the local work-size.

    For example, if the global work-size is (1024, 1024) and the local
  work-size is (8, 8), there will be 128 x 128 work-groups, each
  consisting of 8 x 8 = 64 work-items.

    Within a work-group, work-items can communicate with each other using
  local memory. Local memory is a shared memory space that is accessible
  only to work-items within the same work-group. Work-items within a
  work-group can synchronize their execution using barriers, which ensure
  that all work-items have completed their previous work-items before
  continuing execution.

  enqueue_range_kernel is 1-dimensional version of clEnqueueNDRangeKernel.
  enqueue_range_kernel_2D/3D can be exposed if needed.
*/