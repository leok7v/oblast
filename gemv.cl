#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#ifdef fp16_t
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define syncwarp() barrier(CLK_LOCAL_MEM_FENCE)

inline fp16_t dot16x4(half4 x, half4 y) {
    const half* a = (half*)&x;
    const half* b = (half*)&y;
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

inline void concat(gemv_, fpp)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global       fp_t* restrict rs,
        __local        fp_t* restrict wc,
        const int32_t n, const int32_t m) {
    enum { warp = 32 };
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
//  printf("gemv_32() *************\n");
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fp_t sum = 0;
        // TODO: vec4 optimization
        for (uint x = lid; x < n; x += items) {
            sum += row[x] * vc[x];
//          printf("sum: %g row[%d]: %g vc[%d]: %g\n", sum, x, row[x], x, vc[x]);
        }
        wc[lid] = sum;
        syncwarp();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            syncwarp();
        }
        syncwarp();
        if (lid == 0) {
            rs[y] = wc[0];
        }
        syncwarp();
    }
}

// TODO: dot(half4, half4) is not supported at the time
// of the implementation, needs to be simulated by hand

inline void concat(concat(gemv_, fpp), x4)(
        __global const vec4* restrict mx,
        __global const vec4* restrict vc,
        __global       fp_t* restrict rs,
        __local        fp_t* restrict wc,
        const int32_t n, const int32_t m) {
    enum { warp = 32 };
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
//  printf("gemv_32() *************\n");
    for (uint y = gid; y < m; y += groups) {
        const __global vec4* row = mx + y * n;
        fp_t sum = 0;
        // TODO: vec4 optimization
        for (uint x = lid; x < n; x += items) {
            sum += dot(row[x], vc[x]);
//          printf("sum: %g row[%d]: %g vc[%d]: %g\n", sum, x, row[x], x, vc[x]);
        }
        wc[lid] = sum;
        syncwarp();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            syncwarp();
        }
        syncwarp();
        if (lid == 0) {
            rs[y] = wc[0];
        }
        syncwarp();
    }
}

__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
    concat(gemv_, fpp)(mx, vc, rs, wc, n, m);
}

__kernel
void concat(concat(gemv, fpp), x4)( // gemv32x4 "float4" version
        __global const vec4 mx[/*m][n*/],
        __global const vec4 vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
    concat(concat(gemv_, fpp), x4)(mx, vc, rs, wc, n, m);
}

// uncomment to force error here to see the warnings