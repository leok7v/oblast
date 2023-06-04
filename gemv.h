#include "rt.h"
#include "ocl.h"

typedef struct gemv_s {
    ocl_context_t* c;
    // gemv kernels ocl_fpp16, ocl_fpp32, ocl_fpp64, ocl_bfp16
    ocl_kernel_t kernel[ocl_fpp_last - ocl_fpp_first + 1];
    ocl_kernel_t kernel4x[ocl_fpp_last - ocl_fpp_first + 1];   // vec4 kernel
    ocl_kernel_t kernel16x[ocl_fpp_last - ocl_fpp_first + 1];  // 4 x vec4
} gemv_t;

typedef struct gemv_if {
    void (*init)(gemv_t* g, ocl_context_t* c);
    void (*gemv)(gemv_t* g, int fpp, int64_t offset, ocl_memory_t mx/*[m][n]*/,
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

