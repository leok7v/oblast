#if __OPENCL_VERSION__ <= CL_VERSION_1_1 && fpp == 64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#if fpp == 16 && !defined(bfp16)
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

// This .cl code expects some compile time definitions:
// fpp    -  float point precision 16,32,64
// bfp16  -  "brain float 16" aka bfloat16 undefined or 1
// fp_t   -  fp16_t fp32_t or fp64_t
// fpv4_t -  fp16x4_t fp32x4_t or fp64x4_t for dot(vec4, vec4)
// accumulator types:
// accu   -  32|64 bits in accu_t
// accu_t -  accumulator type for dot products.
//           fp64_t for fpp==64 and fp32_t for all others.
// acc4_t -  float4|fp32x4_t or double4|fp64x4_t

// #include <stdint.h>-like definitions:
typedef char    int8_t;
typedef uchar   uint8_t;
typedef uchar   byte_t;
typedef short   int16_t;
typedef ushort  uint16_t;
typedef int     int32_t;
typedef uint    uint32_t;
typedef long    int64_t;
typedef ulong   uint64_t;
// OpenCL does have uintptr_t and intptr_t which
// are expected to be 64 bits on modern GPUs.

// Every GPU is expected to support float fp32_t and float4
typedef float   fp32_t;
typedef float4  fp32x4_t;

#if fpp == 64
typedef double   fp64_t;
typedef double4  fp64x4_t;
#elif fpp == 16 && !defined(bfp16)
typedef half     fp16_t;
typedef half4    fp16x4_t;
#elif defined(bfp16)
typedef struct __attribute__((packed)) { uint16_t bits; } bf16_t;
#endif

// memory attributes
#define work   __local
#define read   __global const
#define write  __global

#define wr_offsetof(type, offset, p) ((__global type*)((__global byte_t*)(p) + offset))
#define rd_offsetof(type, offset, p) ((read type*)((read byte_t*)(p) + offset))

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define local_fence()    barrier(CLK_LOCAL_MEM_FENCE)
#define subgroup_fence() sub_group_barrier(CLK_LOCAL_MEM_FENCE);

#define reduce_add concat(reduce_add, accu)

// work memory "sm" is accessible for all work items in a group

inline void reduce_add(const uint lid, uint i, accu_t s,
        work accu_t* restrict sm) {
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
        // and now work items in the group can write to work memory
        // at their locations without racing with (*2*) reads above:
        if (lid < i2) { sm[lid] = s; } // (*3*)
        i = i2;
    }
    // this fence guarantees that all (*3*) writes of the last iteration
    // has been completed:
    local_fence();
}

#if fpp != 16 && !defined(bfp16) //     *** fp32_t and fp64_t

inline void concat(gemv_fp, fpp)(
        read  fp_t* restrict mx,
        read  fp_t* restrict vc,
        write accu_t* restrict rs,
        work accu_t* restrict sm, // "sm" work memory
        const int32_t n, const int32_t m) {
    // gemv_fp32|gemv_fp64
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read fp_t* row = mx + y * n;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) { s += row[x] * vc[x]; }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_fp, fpp), x4)(
        read  fpv4_t* restrict mx,
        read  fpv4_t* restrict vc,
        write accu_t* restrict rs,
        work  accu_t* restrict sm,
        const int32_t n, const int32_t m) {
    // gemv_fp32x4 gemv_fp64x4. "n" is 1/4 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read fpv4_t* row = mx + y * n;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) { s += dot(row[x], vc[x]); }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_fp, fpp), x16)(
        read  fpv4_t* restrict mx,
        read  fpv4_t* restrict vc,
        write accu_t* restrict rs,
        work  accu_t* restrict sm,
        const int32_t n, const int32_t m) {
    // gemv_fp32x16 gemv_fp64x16. "n" is 1/16 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read fpv4_t* row = mx + y * n * 4;
        accu_t s = 0;
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

#if max_subgroups > 0

inline void concat(concat(gemv_fp, fpp), _subgroups)(
        read  fp_t*   restrict mx,
        read  fp_t*   restrict vc,
        write accu_t* restrict rs,
        work  accu_t* restrict sm,
        const int32_t n, const int32_t m) {
    // gemv_fp32_subgroups|gemv_fp64_subgroups
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        read fp_t* row = mx + y * n;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) { s += row[x] * vc[x]; }
        subgroup_fence()
        s = sub_group_reduce_add(s);
        reduce_add(sub_group_id, num_sub_groups, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_fp, fpp), x4_subgroups)(
        read fpv4_t*  restrict mx,
        read fpv4_t*  restrict vc,
        write accu_t* restrict rs,
        work accu_t*  restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 or row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        read fpv4_t* row = mx + y * n;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) { s += dot(row[x], vc[x]); }
        subgroup_fence()
        reduce_add(sub_group_id, num_sub_groups, sub_group_reduce_add(s), sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void concat(concat(gemv_fp, fpp), x16_subgroups)(
        read fpv4_t*  restrict mx,
        read fpv4_t*  restrict vc,
        write accu_t* restrict rs,
        work accu_t*  restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 or row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    const uint sub_group_id = get_sub_group_id();
    const uint num_sub_groups = get_num_sub_groups();
    for (uint y = gid; y < m; y += groups) {
        read fpv4_t* row = mx + y * n * 4;
        accu_t s = 0;
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

#endif

// Kernels do not support "restrict" but inline functions do.
// The untested belief is the "restrict" may help compiler
// to generate code that can avoid unnecessary local/global
// RAM access and cache values on the registers more freely.

__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        const int64_t  mx_offset, // cannot be intptr_t
        read fp_t mx[/*m][n*/],
        const int64_t  vc_offset,
        read fp_t vc[/*n*/],
        const int64_t  rs_offset,
        write fp_t rs[/*m*/],
        work  fp_t sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_fp, fpp), _subgroups)(
        rd_offsetof(fp_t, mx_offset, mx),
        rd_offsetof(fp_t, vc_offset, vc),
        wr_offsetof(fp_t, rs_offset, rs),
        sm, n, m);
#else // sm[max_items * max_groups] must be allocated by host
    concat(gemv_fp, fpp)(
        rd_offsetof(fp_t, mx_offset, mx),
        rd_offsetof(fp_t, vc_offset, vc),
        wr_offsetof(fp_t, rs_offset, rs),
        sm, n, m);
#endif
}

__kernel
void concat(concat(gemv, fpp), x4)( // gemv32x4 "float4" version
        const int64_t mx_offset,
        read  fpv4_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  fpv4_t  vc[/*n*/],
        const int64_t rs_offset,
        write fp_t    rs[/*m*/],
        work  fp_t    sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_fp,  fpp), x4_subgroups)(
        rd_offsetof(fpv4_t, mx_offset, mx),
        rd_offsetof(fpv4_t, vc_offset, vc),
        wr_offsetof(fp_t,   rs_offset, rs),
        sm, n, m);
#else
    concat(concat(gemv_fp, fpp), x4)(
        rd_offsetof(fpv4_t, mx_offset, mx),
        rd_offsetof(fpv4_t, vc_offset, vc),
        wr_offsetof(fp_t,   rs_offset, rs),
        sm, n, m);
#endif
}

__kernel
void concat(concat(gemv, fpp), x16)( // gemv32x4 "float4" version
        const int64_t mx_offset,
        read  fpv4_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  fpv4_t  vc[/*n*/],
        const int64_t rs_offset,
        write fp_t    rs[/*m*/],
        work  fp_t    sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
#if max_subgroups > 0
    concat(concat(gemv_fp, fpp), x16_subgroups)(
        rd_offsetof(fpv4_t, mx_offset, mx),
        rd_offsetof(fpv4_t, vc_offset, vc),
        wr_offsetof(fp_t,   rs_offset, rs),
        sm, n, m);
#else
    concat(concat(gemv_fp, fpp), x16)(
        rd_offsetof(fpv4_t, mx_offset, mx),
        rd_offsetof(fpv4_t, vc_offset, vc),
        wr_offsetof(fp_t,   rs_offset, rs),
        sm, n, m);
#endif
}

#elif defined(bfp16) //         *** bf16_t ***

inline fp32_t read_bf(read bf16_t* a) {
    read uint16_t* pbf16 = (__global uint16_t*)a;
    uint32_t v = ((uint32_t)*pbf16) << 16;
    return *(fp32_t*)&v;
}

inline fp32_t load_bf(const intptr_t offset, read bf16_t* a) {
    return read_bf(a + offset);
}

inline void gemv_bf16( // bf16_t
        read  bf16_t* restrict mx,
        read  accu_t* restrict vc,
        write accu_t* restrict rs,
        work  accu_t* restrict sm,
        const  int32_t n, const int32_t m) {
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read bf16_t* row = mx + y * n;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) { s += load_bf(x, row) * vc[x]; }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void gemv_bf16x4( // vec4 of bf16_t
        read  bf16_t* restrict mx,
        read  accu_t* restrict vc,
        write accu_t* restrict rs,
        work  accu_t* restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/4 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read bf16_t* row = mx + y * n * 4;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) {
            read bf16_t* mp = row + (x << 2);
            read accu_t* vp = vc  + (x << 2);
            #pragma unroll 4
            for (uint i = 0; i < 4; i++) {
                s += read_bf(mp++) * *vp++;
            }
        }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void gemv_bf16x16( // 16 elements of bf16_t
        read  bf16_t* restrict mx,
        read  accu_t* restrict vc,
        write accu_t* restrict rs,
        work  accu_t* restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read bf16_t* row = mx + y * n * 16;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) {
            read bf16_t* mp = row + (x << 4);
            read accu_t* vp = vc  + (x << 4);
            #pragma unroll 16
            for (uint i = 0; i < 16; i++) {
                s += read_bf(mp++) * *vp++;
            }
        }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

__kernel
void bfmv16(
        const int64_t mx_offset,
        read  bf16_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  accu_t  vc[/*n*/],
        const int64_t rs_offset,
        write accu_t  rs[/*m*/],
        work  accu_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
//  printf("gemv_bf16\n");
    gemv_bf16(
        rd_offsetof(bf16_t, mx_offset, mx),
        rd_offsetof(accu_t, vc_offset, vc),
        wr_offsetof(accu_t, rs_offset, rs),
        sm, n, m);
}

__kernel
void bfmv16x4(
        const int64_t mx_offset,
        read  bf16_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  accu_t  vc[/*n*/],
        const int64_t rs_offset,
        write accu_t  rs[/*m*/],
        work  accu_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
//  printf("gemv_bf16x4\n");
    gemv_bf16x4(
        rd_offsetof(bf16_t, mx_offset, mx),
        rd_offsetof(accu_t, vc_offset, vc),
        wr_offsetof(accu_t, rs_offset, rs),
        sm, n, m);
}

__kernel
void bfmv16x16(
        const int64_t mx_offset,
        read  bf16_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  accu_t  vc[/*n*/],
        const int64_t rs_offset,
        write accu_t  rs[/*m*/],
        work  accu_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
 // printf("gemv_bf16x16\n");
    gemv_bf16x16(
        rd_offsetof(bf16_t, mx_offset, mx),
        rd_offsetof(accu_t, vc_offset, vc),
        wr_offsetof(accu_t, rs_offset, rs),
        sm, n, m);
}

#else //                        *** fp_t fp16_t ***

inline void gemv_fp16(
        read  fp16_t* restrict mx,
        read  accu_t* restrict vc,
        write accu_t* restrict rs,
        work  accu_t* restrict sm,
        const  int32_t n, const int32_t m) {
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read fp16_t* row = mx + y * n;
        accu_t s = 0;
        for (uint x = lid; x < n; x += items) { s += vload_half(x, row) * vc[x]; }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void gemv_fp16x4(
        read  fp16_t*  restrict mx,
        read  acc4_t* restrict vc,
        write accu_t*  restrict rs,
        work  accu_t*  restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/4 of row width
    // vload_half4 reads sizeof(halfn) bytes of data from
    // address (p + (offset * n))
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read fp16_t* row = mx + y * n * 4;
        accu_t s = 0; // ^^^ * 4 because mx is fp16_t*
        for (uint x = lid; x < n; x += items) { s += dot(vload_half4(x, row), vc[x]); }
        reduce_add(lid, items, s, sm);
        if (lid == 0) { rs[y] = sm[0]; }
    }
}

inline void gemv_fp16x16(
        read  fp16_t*  restrict mx,
        read  acc4_t* restrict vc,
        write accu_t*  restrict rs,
        work  accu_t*  restrict sm,
        const int32_t n, const int32_t m) {
    // "n" is 1/16 of row width
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        read fp16_t* row = mx + y * n * 16;
        accu_t s = 0; // ^^^ * 16 because mx is fp16_t*
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

__kernel
void gemv16(
        const int64_t mx_offset,
        read  fp16_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  accu_t  vc[/*n*/],
        const int64_t rs_offset,
        write accu_t  rs[/*m*/],
        work  accu_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
//  printf("gemv_fp16\n");
    gemv_fp16(
        rd_offsetof(fp16_t, mx_offset, mx),
        rd_offsetof(accu_t, vc_offset, vc),
        wr_offsetof(accu_t, rs_offset, rs),
        sm, n, m);
}

__kernel
void gemv16x4(
        const int64_t mx_offset,
        read  fp16_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  acc4_t vc[/*n*/],
        const int64_t rs_offset,
        write accu_t  rs[/*m*/],
        work  accu_t  sm[/*work_group_items*/],
        const int32_t n, const int32_t m) {
//  printf("gemv_fp16x4\n");
    gemv_fp16x4(
        rd_offsetof(fp16_t,  mx_offset, mx),
        rd_offsetof(acc4_t, vc_offset, vc),
        wr_offsetof(accu_t,  rs_offset, rs),
        sm, n, m);
}

__kernel
void gemv16x16(
        const int64_t mx_offset,
        read  fp16_t  mx[/*m][n*/],
        const int64_t vc_offset,
        read  acc4_t vc[/*n*/],
        const int64_t rs_offset,
        write accu_t  rs[/*m*/],
        work  accu_t  sm[/*work_group_items*/],
        const  int32_t n, const int32_t m) {
//  printf("gemv_fp16x16\n");
    gemv_fp16x16(
        rd_offsetof(fp16_t,  mx_offset, mx),
        rd_offsetof(acc4_t, vc_offset, vc),
        wr_offsetof(accu_t,  rs_offset, rs),
        sm, n, m);
}

#endif

// uncomment to force error here to see the warnings