#pragma once
#include "ocl.h"
#include "fp16.h"

#ifdef __cplusplus
extern "C"
#endif

// float point precision index
enum { blast_fpp16 = 0, blast_fpp32 = 1, blast_fpp64 = 2 };

extern const char* blast_fpp_names[3];
extern const int   blast_fpp_bytes[3]; // { 2, 4, 8 }

enum { // .allocate()/.map() flags
    blast_access_read  = 0, // not a bitset!
    blast_access_write = 1,
    blast_access_rw    = 2
};

typedef struct blast_s blast_t;

typedef struct blast_memory_s { // treat as read only, will change don't cache
    void*   m; // mapped memory address in virtual memory. TODO: can be eliminated?
    void*   h; // handle
    int64_t s; // size in bytes
    blast_t* b;
} blast_memory_t;

typedef struct blast_s {
    ocl_context_t* c;
    // BLAS like operations
    // The offset parameters could be useful when multiple tensors reside in
    // a single memory region.
    // The function pointers below can be null if fp16 or fp64 is not supported:
    // dot()
    fp64_t (*dot[3])(
        blast_memory_t* v0, int64_t offset0, int64_t stride0,
        blast_memory_t* v1, int64_t offset1, int64_t stride1, int64_t n);
    // gemv()
    void (*gemv[3])(
        blast_memory_t* matrix/*[m][n]*/, int64_t offset_m, int64_t stride_m,
        blast_memory_t* vector/*[n]*/,    int64_t offset_v, int64_t stride_v,
        blast_memory_t* result/*[m]*/, int64_t m, int64_t n);
    // kernels are properties of c.c ocl_context:
    ocl_kernel_t dot_c[3];   // compact
    ocl_kernel_t dot_os[3];  // offset + stride
    ocl_kernel_t sum_odd[3];
    ocl_kernel_t sum_odd_os[3];
    ocl_kernel_t sum_even[3];
    ocl_kernel_t sum_even_os[3];
    ocl_kernel_t gemv_c[3];
    ocl_kernel_t gemv_os[3];
    // TODO:
    // TODO:
    ocl_kernel_t copy[3]; // for performance measurements
    ocl_kernel_t fma_os[3];
    ocl_kernel_t mad_c[3];
    ocl_kernel_t mad_os[3];
} blast_t;

typedef struct blast_if {
    void (*init)(blast_t* b, ocl_context_t* c);
   // Only the memory allocated by blast.allocate() can be used as an arguments.
    // Caller MUST unmap that memory to allow access to it by the GPU.
    // and will remap it back when done. The address WILL CHANGE!
    blast_memory_t (*allocate)(blast_t* b, int access, int64_t bytes);
    void  (*deallocate)(blast_memory_t* gm);
    // Client must map blast_memory to host memory before accessing it
    // and unmap before invocation of any other blast operation
    void* (*map)(blast_memory_t* gm, int access, int64_t offset, int64_t bytes);
    void  (*unmap)(blast_memory_t* gm);
    void (*fini)(blast_t* b);
} blast_if;

extern blast_if blast;

#ifdef __cplusplus
}
#endif

#ifdef TODO // (on "as needed" basis)
    Level 1 BLAS (14 subprograms):
    [ ] asum
        axpy
        copy
    [x] dot
        iamax
        nrm2
        rot
        rotg
        rotm
        rotmg
        scal
        swap
        sdsdot
        dsdot

    Level 2 BLAS (6 subprograms):
    [ ] gemv
        gbmv
        hemv
        hbmv
        symv
        sbmv

    Level 3 BLAS (4 subprograms)
        gemm
        symm
        hemm
        syrk
        herk
        syr2k
        her2k
        trmm
        trsm
#endif