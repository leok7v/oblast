#if __OPENCL_VERSION__ <= CL_VERSION_1_1 && fpp == 64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#if fpp == 16
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define syncwarp() barrier(CLK_LOCAL_MEM_FENCE)

#if fpp != 16

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
    for (uint y = gid; y < m; y += groups) {
        const __global vec4* row = mx + y * n;
        fp_t sum = 0;
        for (uint x = lid; x < n; x += items) {
            sum += dot(row[x], vc[x]);
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

#else

inline void gemv_16(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global       fp_t* restrict rs,
        __local      fp32_t* restrict wc,
        const int32_t n, const int32_t m) {
    enum { warp = 32 };
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fp32_t sum = 0;
        for (uint x = lid; x < n; x += items) {
//          sum += row[x] * vc[x];  // no half arithmetics
            sum += vload_half(x, row) * vload_half(x, vc);
        }
        wc[lid] = sum;
        syncwarp();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            syncwarp();
        }
        syncwarp();
        if (lid == 0) {
//          rs[y] = wc[0];
            vstore_half(wc[0], y, rs);
        }
        syncwarp();
    }
}

inline void gemv_16x4(
        __global const vec4* restrict mx,
        __global const vec4* restrict vc,
        __global       fp_t* restrict rs,
        __local      fp32_t* restrict wc,
        const int32_t n, const int32_t m) {
    enum { warp = 32 };
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global vec4* row = mx + y * n;
        fp32_t sum = 0;
        for (uint x = lid; x < n; x += items) {
//          sum += row[x] * vc[x];  // no half arithmetics
            sum += dot(vload_half4(x, row), vload_half4(x, vc));
        }
        wc[lid] = sum;
        syncwarp();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            syncwarp();
        }
        syncwarp();
        if (lid == 0) {
//          rs[y] = wc[0];
            vstore_half(wc[0], y, rs);
        }
        syncwarp();
    }
}

__kernel
void gemv16( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp32_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
    concat(gemv_, fpp)(mx, vc, rs, wc, n, m);
}

__kernel
void gemv16x4( // gemv32x4 "half4" version
        __global const vec4 mx[/*m][n*/],
        __global const vec4 vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp32_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
    gemv_16x4(mx, vc, rs, wc, n, m);
}

#endif


// uncomment to force error here to see the warnings