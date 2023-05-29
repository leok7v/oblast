#include "rt.h"
#include "ocl.h"

typedef struct gemv_s {
    ocl_context_t* c;
    ocl_kernel_t kernel[3]; // gemv kernels ocl_fpp16, ocl_fpp32, ocl_fpp64
    ocl_kernel_t kernel4x[3];
    ocl_kernel_t kernel16x[3];
} gemv_t;

typedef struct gemv_if {
    void (*init)(gemv_t* g, ocl_context_t* c);
    void (*ocl_gemv16)(gemv_t* g, int64_t offset, ocl_memory_t mx/*[m][n]*/,
                       ocl_memory_t vc/*[n]*/, ocl_memory_t rs/*[m]*/,
                       int64_t n, int64_t m);
    void (*ocl_gemv32)(gemv_t* g, int64_t offset, ocl_memory_t mx/*[m][n]*/,
                       ocl_memory_t vc/*[n]*/, ocl_memory_t rs/*[m]*/,
                       int64_t n, int64_t m);
    void (*ocl_gemv64)(gemv_t* g, int64_t offset, ocl_memory_t mx/*[m][n]*/,
                       ocl_memory_t vc/*[n]*/, ocl_memory_t rs/*[m]*/,
                       int64_t n, int64_t m);
    // testing convenience (slow performs copy to/from GPU memory):
    void (*gemv16)(gemv_t* g, fp16_t mx[/*m][n*/], fp32_t vc[/*n*/],
        fp32_t rs[/*m*/], int64_t n, int64_t m);
    void (*gemv32)(gemv_t* g, fp32_t mx[/*m][n*/], fp32_t vc[/*n*/],
        fp32_t rs[/*m*/], int64_t n, int64_t m);
    void (*gemv64)(gemv_t* g, fp64_t mx[/*m][n*/], fp64_t vc[/*n*/],
        fp64_t rs[/*m*/], int64_t n, int64_t m);
    void (*fini)(gemv_t* g);
} gemv_if;

extern gemv_if gemv;

