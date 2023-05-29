#if __OPENCL_VERSION__ <= CL_VERSION_1_1 && fpp == 64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#if fpp == 16
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable
#pragma OPENCL EXTENSION cl_nvidia_printf : enable

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define local_fence() barrier(CLK_LOCAL_MEM_FENCE)

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
    for (uint y = gid; y < m; y += groups) {
//      printf("y: %d [gid:%d .. m:%d]\n", y, gid, m);
        const __global fp_t* row = mx + y * n;
        fp_t sum = 0;
        for (uint x = lid; x < n; x += items) { sum += row[x] * vc[x]; }
        wc[lid] = sum;
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) {
            rs[y] = wc[0];
        }
        local_fence();
    }
}

//#define USE_SUB_GROUPS 1

#if max_subgroups > 0 && defined(USE_SUB_GROUPS)

//  https://registry.khronos.org/OpenCL/sdk/2.0/docs/man/xhtml/sub_group_barrier.html
//  memory_order_relaxed
//  memory_order_acquire
//  memory_order_release
//  memory_order_acq_rel
//  memory_order_seq_cst
//  void sub_group_barrier(cl_mem_fence_flags flags)
//  void sub_group_barrier(cl_mem_fence_flags flags, memory_scope scope)
//  https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
//  https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html
//  subgroup benefits from sub_group_reduce_add() but
//  1. wc[items] must be wc[items * max_subgroups]
//  2. wc[lid] = sum; becomes wc[lid + get_sub_group_id()] = sum;
//  3. for get_sub_group_id() we need to do another parallel reduction step

inline void concat(concat(gemv_, fpp), _subgroups)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global       fp_t* restrict rs,
        __local        fp_t* restrict wc,
        const int32_t n, const int32_t m) {

    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint subgroup_size = get_sub_group_size();
    const uint num_sub_groups = get_num_sub_groups();
//  get_enqueued_num_sub_groups() not supported and actually not needed
//  if (gid == 0 && lid == 0 && sub_group_id == 0) {
//      printf("num_sub_groups: %d\n", num_sub_groups);
//      printf("sub_group_id: %d\n", sub_group_id);
//      printf("sub_group_local_id: %d\n", get_sub_group_local_id());
//      printf("subgroup_size %d\n", subgroup_size);
//      printf("groups %d\n", groups);
//  }
    for (uint y = gid; y < m; y += groups) {
//      printf("y: %d [gid:%d .. m:%d]\n", y, gid, m);
        const __global fp_t* row = mx + y * n;
        // Compute partial sum within subgroup
        fp_t sum = 0;
        for (uint x = lid; x < n; x += items) {
            sum += row[x] * vc[x];
        }
        sum = sub_group_reduce_add(sum);
        wc[lid + sub_group_id] = sum;
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        local_fence();
        for (uint s = num_sub_groups >> 1; s > 0; s >>= 1) {
            if (sub_group_id < s) { wc[sub_group_id] += wc[sub_group_id + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) {
//          printf("y: %d wc[0]: %g\n lid: %d "
//              "subgroup_id: %d subgroup_local_id: %d\n",
//              y, wc[0], lid, sub_group_id, get_sub_group_local_id());
            rs[y] = wc[0];
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#endif

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
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) {
            rs[y] = wc[0];
        }
        local_fence();
    }
}

__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m) {
#if max_subgroups > 0 && defined(USE_SUB_GROUPS)
    concat(concat(gemv_, fpp), _subgroups)(mx, vc, rs, wc, n, m);
#else
    concat(gemv_, fpp)(mx, vc, rs, wc, n, m);
#endif
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
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) {
//          rs[y] = wc[0];
            vstore_half(wc[0], y, rs);
        }
        local_fence();
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
//          sum += row[x] * vc[x];  // no half arithmetic
            sum += dot(vload_half4(x, row), vload_half4(x, vc));
        }
        wc[lid] = sum;
        local_fence();
        for (uint s = items >> 1; s > 0; s >>= 1) {
            if (lid < s) { wc[lid] += wc[lid + s]; }
            local_fence();
        }
        local_fence();
        if (lid == 0) {
            vstore_half(wc[0], y, rs); // rs[y] = wc[0];
        }
        local_fence();
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