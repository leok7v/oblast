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

#define local_fence()    barrier(CLK_LOCAL_MEM_FENCE)
#define subgroup_fence() sub_group_barrier(CLK_LOCAL_MEM_FENCE);


#if fpp != 16

inline void concat(gemv_, fpp)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict wc,
        const int32_t n, const int32_t m) {
    // gemv_32|gemv_64
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
//      printf("y: %d [gid:%d .. m:%d]\n", y, gid, m);
        const __global fp_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) { s += row[x] * vc[x]; }
        wc[lid] = s;
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) { rs[y] = wc[0]; }
    }
}

#if max_subgroups > 0

inline void concat(concat(gemv_, fpp), _subgroups)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global     fpmx_t* restrict rs,
        __local      fpmx_t* restrict wc,
        const int32_t n, const int32_t m) {
    // gemv_32_subgroups|gemv_64_subgroups
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint subgroup_size = get_sub_group_size();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) { s += row[x] * vc[x]; }
        subgroup_fence()
        wc[lid + sub_group_id] = sub_group_reduce_add(s);
        local_fence();
        for (uint s = num_sub_groups >> 1; s > 0; s >>= 1) {
            if (sub_group_id < s) { wc[sub_group_id] += wc[sub_group_id + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) { rs[y] = wc[0]; }
    }
}

#endif

inline void concat(concat(gemv_, fpp), x4)(
        __global const vec4_t* restrict mx,
        __global const vec4_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict wc,
        const int32_t n, const int32_t m) {
    // gemv_32x4 gemv_64x4. Important "n" is 1/4 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
//  printf("n: %d m:%d\n", n, m);
    for (uint y = gid; y < m; y += groups) {
        const __global vec4_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) {
            vec4_t rx = row[x];
            vec4_t vx = vc[x];
            s += dot(row[x], vc[x]);
//          printf("s: %g rx: %g %g %g %g vx: %g %g %g %g\n", s,
//              rx.x, rx.y, rx.z, rx.w, vx.x, vx.y, vx.z, vx.w);
        }
        wc[lid] = s;
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) {
//          printf("r[%d] = %d\n", y, wc[0]);
            rs[y] = wc[0];
        }
    }
}

__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp_t wc[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_, fpp), _subgroups)(mx, vc, rs, wc, n, m);
#else // wc[max_items * max_groups] must be allocated by host
    concat(gemv_, fpp)(mx, vc, rs, wc, n, m);
#endif
}

__kernel
void concat(concat(gemv, fpp), x4)( // gemv32x4 "float4" version
        __global const vec4_t mx[/*m][n*/],
        __global const vec4_t vc[/*n*/],
        __global       fp_t   rs[/*m*/],
        __local        fp_t   wc[/*work_group_items*/],
        const int32_t n, const int32_t m) {
    concat(concat(gemv_, fpp), x4)(mx, vc, rs, wc, n, m);
}

#else

inline void gemv_16(
        __global const fp16_t* restrict mx,
        __global const fpmx_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict wc,
        const int32_t n, const int32_t m) {
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) {
            s += vload_half(x, row) * vc[x];
//          printf("lid: %d y: %d s: %g row[%d]: %g vc[%d] %g\n",
//              lid, y, s, x, vload_half(x, row), x, vc[x]);
        }
        wc[lid] = s;
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) { rs[y] = wc[0]; }
    }
}

inline void gemv_16x4(
        __global const fp16_t*  restrict mx,
        __global const fpmx4_t* restrict vc,
        __global       fpmx_t*  restrict rs,
        __local        fpmx_t*  restrict wc,
        const int32_t n, const int32_t m) {
    // Important "n" is 1/4 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp16_t* row = mx + y * n * 4;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) {
            // vload_half4 reads sizeof(halfn) bytes of data from
            // address (p + (offset * n))
//          const float4 rx4 = vload_half4(x, row);
//          const float4 vx4 = vc[x];
//          s += dot(rx4, vx4);
            s += dot(vload_half4(x, row), vc[x]);
        }
        wc[lid] = s;
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) { rs[y] = wc[0]; }
    }
}

__kernel
void gemv16( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp16_t mx[/*m][n*/],
        __global const fpmx_t vc[/*n*/],
        __global       fpmx_t rs[/*m*/],
        __local        fpmx_t wc[/*work_group_items*/],
        const int32_t n, const int32_t m) {
    concat(gemv_, fpp)(mx, vc, rs, wc, n, m);
}

__kernel
void gemv16x4( // gemv32x4 "half4" version
        __global const fp16_t  mx[/*m][n*/],
        __global const fpmx4_t vc[/*n*/],
        __global       fpmx_t  rs[/*m*/],
        __local        fpmx_t  wc[/*work_group_items*/],
        const int32_t n, const int32_t m) {
    gemv_16x4(mx, vc, rs, wc, n, m);
}

#endif

// uncomment to force error here to see the warnings