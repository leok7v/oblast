#if __OPENCL_VERSION__ <= CL_VERSION_1_1 && fpp == 64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#if fpp == 16
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

// #pragma OPENCL EXTENSION cl_amd_printf : enable
// #pragma OPENCL EXTENSION cl_intel_printf : enable
// #pragma OPENCL EXTENSION cl_nvidia_printf : enable

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define local_fence() barrier(CLK_LOCAL_MEM_FENCE)

#if fpp != 16

inline void concat(gemv_, fpp)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global       fp_t* restrict rs,
        const int32_t n, const int32_t m) {
    const uint y = get_global_id(0);
    const __global fp_t* row = mx + y * n;
    prefetch(vc, n);
    fp_t sum = 0;
    for (int i = 0; i < n; i++) { sum += row[i] * vc[i]; }
//  printf("rs[%d] = %g\n", y, sum);
    rs[y] = sum;
}

inline void concat(concat(gemv_, fpp), x4)(
        __global const vec4* restrict mx,
        __global const vec4* restrict vc,
        __global       fp_t* restrict rs,
        const int32_t n, const int32_t m) {

    const uint y = get_global_id(0);
    const __global vec4* row = mx + y * n;
    fp_t sum = 0;
    for (int i = 0; i < n; i++) { sum += dot(row[i], vc[i]); }
//  printf("rs[%d] = %g\n", y, sum);
    rs[y] = sum;
}

__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __global       fp_t rs[/*m*/],
        const int32_t n, const int32_t m) {
//  __local fp_t wc[1024];
//  event_t e = async_work_group_copy(wc, vc, n);
//  wait_group_events(1, &e);
    concat(gemv_, fpp)(mx, vc, rs, n, m);
}

__kernel
void concat(concat(gemv, fpp), x4)( // gemv32x4 "float4" version
        __global const vec4 mx[/*m][n*/],
        __global const vec4 vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
    concat(concat(gemv_, fpp), x4)(mx, vc, rs, n, m);
}

#else

inline void gemv_16(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global       fp_t* restrict rs,
        const int32_t n, const int32_t m) {


    const uint y = get_global_id(0);
    const __global fp_t* row = mx + y * n;
    fp32_t sum = 0;
    for (int i = 0; i < n; i++) { sum += vload_half(x, row) * vload_half(x, vc); }
//  printf("rs[%d] = %g\n", y, sum);
    vstore_half(sum, y, rs); // rs[y] = sum;
}

inline void gemv_16x4(
        __global const vec4* restrict mx,
        __global const vec4* restrict vc,
        __global       fp_t* restrict rs,
        const int32_t n, const int32_t m) {

    const uint y = get_global_id(0);
    const __global vec4* row = mx + y * n;
    fp32_t sum = 0;
    for (int i = 0; i < n; i++) { sum += dot(vload_half(x, row), vload_half(x, vc)); }
//  printf("rs[%d] = %g\n", y, sum);
    vstore_half(sum, y, rs); // rs[y] = sum;
}

__kernel
void gemv16( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp32_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
    concat(gemv_, fpp)(mx, vc, rs, n, m);
}

__kernel
void gemv16x4( // gemv32x4 "half4" version
        __global const vec4 mx[/*m][n*/],
        __global const vec4 vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp32_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
    gemv_16x4(mx, vc, rs, n, m);
}

#endif

// uncomment to force error here to see the warnings