#if __OPENCL_VERSION__ <= CL_VERSION_1_1 && fpp == 64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#if fpp == 16
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define local_fence()    barrier(CLK_LOCAL_MEM_FENCE)
#define subgroup_fence() sub_group_barrier(CLK_LOCAL_MEM_FENCE);

#define reduce_add concat(reduce_add, fpmx)

// shared memory "sm" is accessible for all work items in a group

// TODO: dot16bf

inline void reduce_add(const uint lid, uint i, fpmx_t s,
        __local fpmx_t* restrict sm) {
    sm[lid] = s; // (*1*)
    while (i > 1) {
        // this fence guarantees all (*1*) memory write are complete
        // in this group and s = ... (*2*) reads coherent values
        uint i2 = i >> 1;
        local_fence();
        if (lid < i2) {
            s = sm[lid] + sm[lid + i2] + // (*2*)
                (((lid == 0) & (i & 1)) ? sm[lid + i - 1] : 0.0f);
        }
        // this fence guarantees that all (*2*) reads are done and summed
        // in register "s".
        local_fence();
        // and now work items in the group can write to shared memory
        // at their locations without racing with (*2*) reads above:
        if (lid < i2) { sm[lid] = s; } // (*3*)
        i = i2;
    }
    // this fence guarantees that all (*3*) writes of the last iteration
    // has been completed:
    local_fence();
}

#if fpp != 16

inline void concat(gemv_, fpp)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict sm, // "sm" shared memory
        const int32_t n, const int32_t m) {
    // gemv_32|gemv_64
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) { s += row[x] * vc[x]; }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_, fpp), x4)(
        __global const vec4_t* restrict mx,
        __global const vec4_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    // gemv_32x4 gemv_64x4. "n" is 1/4 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global vec4_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) { s += dot(row[x], vc[x]); }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_, fpp), x16)(
        __global const vec4_t* restrict mx,
        __global const vec4_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    // gemv_32x16 gemv_64x16. "n" is 1/16 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global vec4_t* row = mx + y * n * 4;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) {
            uint x4 = x << 2;
            s +=
                dot(row[x4 + 0], vc[x4 + 0]) +
                dot(row[x4 + 1], vc[x4 + 1]) +
                dot(row[x4 + 2], vc[x4 + 2]) +
                dot(row[x4 + 3], vc[x4 + 3]);
        }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_, fpp), x32)(
        __global const vec4_t* restrict mx,
        __global const vec4_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    // gemv_32x16 gemv_64x16. "n" is 1/32 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global vec4_t* row = mx + y * n * 8;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) {
            uint x8 = x << 3;
            s +=
                dot(row[x8 + 0], vc[x8 + 0]) +
                dot(row[x8 + 1], vc[x8 + 1]) +
                dot(row[x8 + 2], vc[x8 + 2]) +
                dot(row[x8 + 3], vc[x8 + 3]) +
                dot(row[x8 + 4], vc[x8 + 4]) +
                dot(row[x8 + 5], vc[x8 + 5]) +
                dot(row[x8 + 6], vc[x8 + 6]) +
                dot(row[x8 + 7], vc[x8 + 7]);
        }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

#if max_subgroups > 0

inline void concat(concat(gemv_, fpp), _subgroups)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __global     fpmx_t* restrict rs,
        __local      fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    // gemv_32_subgroups|gemv_64_subgroups
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) { s += row[x] * vc[x]; }
        subgroup_fence()
        s = sub_group_reduce_add(s);
        reduce_add(sub_group_id, num_sub_groups, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_, fpp), x4_subgroups)(
        __global const vec4_t* restrict mx,
        __global const vec4_t* restrict vc,
        __global     fpmx_t* restrict rs,
        __local      fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 or row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        const __global vec4_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) { s += dot(row[x], vc[x]); }
        subgroup_fence()
        reduce_add(sub_group_id, num_sub_groups, sub_group_reduce_add(s), sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_, fpp), x16_subgroups)(
        __global const vec4_t* restrict mx,
        __global const vec4_t* restrict vc,
        __global     fpmx_t* restrict rs,
        __local      fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 or row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        const __global vec4_t* row = mx + y * n * 4;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) {
            uint x4 = x << 2;
            s +=
                dot(row[x4 + 0], vc[x4 + 0]) +
                dot(row[x4 + 1], vc[x4 + 1]) +
                dot(row[x4 + 2], vc[x4 + 2]) +
                dot(row[x4 + 3], vc[x4 + 3]);
        }
        subgroup_fence();
        reduce_add(sub_group_id, num_sub_groups, sub_group_reduce_add(s), sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_, fpp), x32_subgroups)(
        __global const vec4_t* restrict mx,
        __global const vec4_t* restrict vc,
        __global     fpmx_t* restrict rs,
        __local      fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 or row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        const __global vec4_t* row = mx + y * n * 8;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) {
            uint x8 = x << 3;
            s +=
                dot(row[x8 + 0], vc[x8 + 0]) +
                dot(row[x8 + 1], vc[x8 + 1]) +
                dot(row[x8 + 2], vc[x8 + 2]) +
                dot(row[x8 + 3], vc[x8 + 3]) +
                dot(row[x8 + 4], vc[x8 + 4]) +
                dot(row[x8 + 5], vc[x8 + 5]) +
                dot(row[x8 + 6], vc[x8 + 6]) +
                dot(row[x8 + 7], vc[x8 + 7]);
        }
        subgroup_fence()
        reduce_add(sub_group_id, num_sub_groups, sub_group_reduce_add(s), sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

#endif

__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __global       fp_t rs[/*m*/],
        __local        fp_t sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_, fpp), _subgroups)(mx, vc, rs, sm, n, m);
#else // sm[max_items * max_groups] must be allocated by host
    concat(gemv_, fpp)(mx, vc, rs, sm, n, m);
#endif
}

__kernel
void concat(concat(gemv, fpp), x4)( // gemv32x4 "float4" version
        __global const vec4_t mx[/*m][n*/],
        __global const vec4_t vc[/*n*/],
        __global       fp_t   rs[/*m*/],
        __local        fp_t   sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_, fpp), x4_subgroups)(mx, vc, rs, sm, n, m);
#else
    concat(concat(gemv_, fpp), x4)(mx, vc, rs, sm, n, m);
#endif
}

__kernel
void concat(concat(gemv, fpp), x16)( // gemv32x4 "float4" version
        __global const vec4_t mx[/*m][n*/],
        __global const vec4_t vc[/*n*/],
        __global       fp_t   rs[/*m*/],
        __local        fp_t   sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_, fpp), x16_subgroups)(mx, vc, rs, sm, n, m);
#else
    concat(concat(gemv_, fpp), x16)(mx, vc, rs, sm, n, m);
#endif
}

__kernel
void concat(concat(gemv, fpp), x32)( // gemv32x4 "float4" version
        __global const vec4_t mx[/*m][n*/],
        __global const vec4_t vc[/*n*/],
        __global       fp_t   rs[/*m*/],
        __local        fp_t   sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_, fpp), x32_subgroups)(mx, vc, rs, sm, n, m);
#else
    concat(concat(gemv_, fpp), x32)(mx, vc, rs, sm, n, m);
#endif
}

#else

inline void gemv_16(
        __global const fp16_t* restrict mx,
        __global const fpmx_t* restrict vc,
        __global       fpmx_t* restrict rs,
        __local        fpmx_t* restrict sm,
        const int32_t n, const int32_t m) {
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fpmx_t s = 0;
        for (uint x = lid; x < n; x += items) { s += vload_half(x, row) * vc[x]; }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void gemv_16x4(
        __global const fp16_t*  restrict mx,
        __global const fpmx4_t* restrict vc,
        __global       fpmx_t*  restrict rs,
        __local        fpmx_t*  restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/4 of row width
    // vload_half4 reads sizeof(halfn) bytes of data from
    // address (p + (offset * n))
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp16_t* row = mx + y * n * 4;
        fpmx_t s = 0; // ^^^ * 4 because mx is fp16_t*
        for (uint x = lid; x < n; x += items) { s += dot(vload_half4(x, row), vc[x]); }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void gemv_16x16(
        __global const fp16_t*  restrict mx,
        __global const fpmx4_t* restrict vc,
        __global       fpmx_t*  restrict rs,
        __local        fpmx_t*  restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp16_t* row = mx + y * n * 16;
        fpmx_t s = 0; // ^^^ * 16 because mx is fp16_t*
        for (uint x = lid; x < n; x += items) {
            uint x4 = x << 2;
            s +=
                dot(vload_half4(x4 + 0, row), vc[x4 + 0]) +
                dot(vload_half4(x4 + 1, row), vc[x4 + 1]) +
                dot(vload_half4(x4 + 2, row), vc[x4 + 2]) +
                dot(vload_half4(x4 + 3, row), vc[x4 + 3]);
        }
        reduce_add32(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void gemv_16x32(
        __global const fp16_t*  restrict mx,
        __global const fpmx4_t* restrict vc,
        __global       fpmx_t*  restrict rs,
        __local        fpmx_t*  restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp16_t* row = mx + y * n * 32;
        fpmx_t s = 0; // ^^^ * 16 because mx is fp16_t*
        for (uint x = lid; x < n; x += items) {
            uint x8 = x << 3;
            s +=
                dot(vload_half4(x8 + 0, row), vc[x8 + 0]) +
                dot(vload_half4(x8 + 1, row), vc[x8 + 1]) +
                dot(vload_half4(x8 + 2, row), vc[x8 + 2]) +
                dot(vload_half4(x8 + 3, row), vc[x8 + 3]) +
                dot(vload_half4(x8 + 4, row), vc[x8 + 4]) +
                dot(vload_half4(x8 + 5, row), vc[x8 + 5]) +
                dot(vload_half4(x8 + 6, row), vc[x8 + 6]) +
                dot(vload_half4(x8 + 7, row), vc[x8 + 7]);
        }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

__kernel
void gemv16(
        __global const fp16_t mx[/*m][n*/],
        __global const fpmx_t vc[/*n*/],
        __global       fpmx_t rs[/*m*/],
        __local        fpmx_t sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
    concat(gemv_, fpp)(mx, vc, rs, sm, n, m);
}

__kernel
void gemv16x4(
        __global const fp16_t  mx[/*m][n*/],
        __global const fpmx4_t vc[/*n*/],
        __global       fpmx_t  rs[/*m*/],
        __local        fpmx_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
    gemv_16x4(mx, vc, rs, sm, n, m);
}

__kernel
void gemv16x16(
        __global const fp16_t  mx[/*m][n*/],
        __global const fpmx4_t vc[/*n*/],
        __global       fpmx_t  rs[/*m*/],
        __local        fpmx_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
    gemv_16x16(mx, vc, rs, sm, n, m);
}

__kernel
void gemv16x32(
        __global const fp16_t  mx[/*m][n*/],
        __global const fpmx4_t vc[/*n*/],
        __global       fpmx_t  rs[/*m*/],
        __local        fpmx_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
    gemv_16x32(mx, vc, rs, sm, n, m);
}

#endif

// uncomment to force error here to see the warnings