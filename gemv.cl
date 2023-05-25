#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#ifdef fp16_t
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define _concat_(first, last)  first ## last
#define name(first, last)      _concat_(first, last)


#define gpro_t __global const fp_t* // global read only elements
#define gpwr_t __global fp_t*       // global write only elements

inline fp_t vector_dot_product(gpro_t const x, gpro_t const y, const int32_t n) {
    fp_t sum = 0;
    for (int i = 0; i < n; i++) { sum += x[i] * y[i]; }
    return sum;
}

__kernel void name(gemv, fpp)(gpro_t const mx, gpro_t const vc,
        const int32_t n, gpwr_t rs) {
    const int32_t i = get_global_id(0);
    rs[i] = vector_dot_product(mx + i * n, vc, n);
}

