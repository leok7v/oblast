// expectes to be prepended with
// #define suffix fp16|fp32|fp64
// #define fp_t float
// or
// #define fp_t double
// #define fp_t half

// for gemv() optimizations vec4, vec8, vec16 must be defined as:
// #define vec4 type4
// and only for half
// #define vec8 type8
// and
// #define vec16 type16
// there <type> is one of half|float|double

// this is deprecated in favor of mad() and fma()
// #pragma OPENCL SELECT_ROUNDING_MODE rte // rte rtz rtp rtn

// Because half4, half8 and half16 has limited support at the
// time of writing there is workaround for surrogate kernels fp16_t.
// is the special case.
// #define fp16_surrogate 1
// or
// #define fp16_surrogate 2
// to use alternative implementations of:
// dot_fp16x16_1()
// or
// dot_fp16x16_2()
// TODO: measure which one is faster

#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#ifdef fp16_t
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define _concat_(first, last)  first ##_## last
#define name(first, last)      _concat_(first, last)

// name(foo, suffix) generate names like
//   foo_fp16() foo_fp32() foo_fp64()
//   offset/strided (_os_) versions are much slower - upto 5 alu cycles
// for offset + i * stride (reportedly).                TODO: measure
//   foo_os_fp16() foo_os_fp32() foo_os_fp64()
//
// Further speed up:
// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_fp16
// dot(half16, half16) is capable of dot product of 16 fp16_t values

// experimental evidence on NVIDIA GeForce RTX 3080 Laptop
// 1. not defined: FP_FAST_FMA_HALF not defined
// 2. dot(halfN, halfN) not defined for N=2,3,5,16

// below are substitutes for dot(half4, half4) and dot(half16, half16)

#define fp_ro_t __global const fp_t* // pointer to read only elements
#define fp_wr_t __global fp_t*       // pointer to write only elements

__kernel void name(sum_odd, suffix)(fp_ro_t const v, fp_wr_t r) {
    const int32_t i = get_global_id(0);
    const int32_t m = get_global_size(0); // middle
    const int32_t e = get_global_size(0) * 2; // end
    if (i == 0) { // extra one for odd for first element only
        r[0] = v[0] + v[m] + v[e];
    } else {
        r[i] = v[i] + v[i + m];
    }
}

__kernel void name(sum_even, suffix)(fp_ro_t const v, fp_wr_t r) {
    const int32_t i = get_global_id(0);
    const int32_t m = get_global_size(0);
    r[i] = v[i] + v[(i + m)];
}

__kernel void name(sum_odd_os, suffix)(fp_ro_t v,
        const int32_t offset, const int32_t stride, fp_wr_t r) {
    const int32_t i = get_global_id(0);
    const int32_t m = get_global_size(0); // middle
    const int32_t e = get_global_size(0) * 2; // end
    if (i == 0) {
        r[i] = v[offset + i * stride] + v[offset + (i + m) * stride] +
               v[offset + e * stride]; // extra one for odd
    } else {
        r[i] = v[offset + i * stride] + v[offset + (i + m) * stride];
    }
}

__kernel void name(sum_even_os, suffix)(fp_ro_t const v,
        const int32_t offset, const int32_t stride, fp_wr_t r) {
    const int32_t i = get_global_id(0);
    const int32_t m = get_global_size(0);
    r[i] = v[offset + i * stride] + v[offset + (i + m) * stride];
}

// for n = groups * items:
// dot(x, y, r) does not do summation and to complete operation
// sum_xxx(r, s[n / 2]), sum_xxx(r, n / 4) ...
// must be chained after initial dot()
// _xxx must be _odd or _even depending on oddness of "n" not "n / 2"

__kernel void name(dot, suffix)(fp_ro_t const v0, fp_ro_t const v1, fp_wr_t r) {
    const int32_t i = get_global_id(0); // (0) of dimension zero out of 3
    r[i] = v0[i] * v1[i];
}

__kernel void name(dot_os, suffix)(
        fp_ro_t const v0, const int32_t offset0, const int32_t stride0,
        fp_ro_t const v1, const int32_t offset1, const int32_t stride1,
        fp_wr_t r) {
    const int32_t i = get_global_id(0);
    r[i] = v0[offset0 + i * stride0] * v1[offset1 + i * stride1];
}

// TODO: dot16_fp16(), dot4_fp32(), dot4_fp4() future optimization

// gemv General Matrix Multiplication by Vector
// for k = groups * items:
// v[n] sequential memory addresses
// mx[rows][columns] rows == k sequential memory addresses
// r[k]

__kernel void name(gemv, suffix)(fp_ro_t const mx, fp_ro_t const v,
        fp_wr_t r, const int32_t n) {
    const int32_t i = get_global_id(0);
    fp_ro_t const m = mx + i * n;
    fp_t s = 0;
    for (int32_t j = 0; j < n; j++) { s += v[j] * m[j]; }
    r[i] = s;
}

#ifndef fp16_surrogate

__kernel void name(gemv4, suffix)(fp_ro_t mx, fp_ro_t v, fp_wr_t r, int32_t n) {
    const int32_t i = get_global_id(0);
    fp_ro_t m = mx + i * n;
    fp_t s = 0;
    while (n > 4) {
        s += dot(*(__global vec4*)v, *(__global vec4*)m);
        v += 4; m += 4; n -= 4;
    }
    while (n > 0) {
        s += *v++ * *m++;
        n--;
    }
    r[i] = s;
}

#endif

// example, for:
// mx = [ m00 m01 m02 m03 ]
//      [ m10 m11 m12 m13 ]
//      [ m20 m21 m22 m23 ]
//      [ m30 m31 m32 m33 ]
//      [ m40 m41 m42 m43 ]
//      [ m50 m51 m52 m53 ]
// v = [v0 v1 v2 v3 v4 v5 v6 v7 v8]
// dot_fp??(mx, 6, 5, 1
//          v, 1, 2,
//          r) with groups * items = 5
// will multiply submatrix M11 to M43 (in CAPS):
// mx = [ m00 m01 m02 m03 ]
//      [ m10[M11_M12_M13 ]
//      [ m20 M21_M22_M23 ]
//      [ m30 M31_M32_M33 ]
//      [ m40 M41_M42_M43 ]
//      [ m50 M51_M52_M53]]
// to the vector [v1, v3, v5]
// producing r[4]
//   r[0] = [v1, v3, v5] dot  [M11 M12 M13]
//   r[1] = [v1, v3, v5] dot  [M21 M22 M23]
//   r[2] = [v1, v3, v5] dot  [M31 M32 M33]
//   r[3] = [v1, v3, v5] dot  [M41 M42 M43]
//   r[4] = [v1, v3, v5] dot  [M51 M52 M53]

__kernel void name(gemv_os, suffix)(
        fp_ro_t mx, const int32_t mx_offset,
        int32_t row_stride, int32_t column_stride,
        fp_ro_t vc,
        const int32_t offset, const int32_t stride,
        fp_wr_t r, const int32_t n) {
    const int32_t i = get_global_id(0);
    fp_ro_t m = mx + mx_offset + i * row_stride;
    fp_ro_t v = vc + offset;
    fp_t s = 0;
    for (int32_t j = 0; j < n; j++) {
        s += v[j * stride] * mx[j * column_stride];
    }
    r[i] = s;
}

#if defined(fp16_t) && defined(fp16_surrogate)

#define fp16ro_t __global const fp16_t*
#define fp16wr_t __global fp16_t*

inline float dot_fp16x4(fp16ro_t const a, fp16ro_t const b) {
    return dot((float4)*a, (float4)*b);
}

inline float dot_fp16x8(fp16ro_t a, fp16ro_t b) {
    return dot_fp16x4(a + 0, b + 0) + dot_fp16x4(a +  4, b +  4);
}

inline float dot_fp16x16(fp16ro_t a, fp16ro_t b) {
    return dot_fp16x8(a + 0, b + 0) + dot_fp16x8(a +  8, b +  8);
}

__kernel void gemv4_fp16(fp16ro_t mx, fp16ro_t v, fp16wr_t r, int32_t n) {
    const int32_t i = get_global_id(0);
    fp16ro_t m = mx + i * n;
    fp32_t s = 0;
    while (n >= 4) {
        s += dot_fp16x4(v, m);
        n -= 4; v += 4; m += 4;
    }
    while (n >= 0) {
        s += *v++ * *m++; n--;
    }
    r[i] = (fp16_t)s;
}

__kernel void gemv16_fp16(fp16ro_t mx, fp16ro_t v, fp16wr_t r, int32_t n) {
    const int32_t i = get_global_id(0);
    fp16ro_t m = mx + i * n;
    fp32_t s = 0;
    while (n >= 16) {
        s += dot_fp16x16(v, m);
        n -= 16; v += 16; m += 16;
    }
    while (n >= 8) {
        s += dot_fp16x8(v, m);
        n -= 8; v += 8; m += 8;
    }
    while (n >= 4) {
        s += dot_fp16x4(v, m);
        n -= 4; v += 4; m += 4;
    }
    while (n >= 0) {
        s += *v++ * *m++; n--;
    }
    r[i] = (fp16_t)s;
}

#endif // fp16_surrogate
