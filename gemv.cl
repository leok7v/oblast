#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#ifdef fp16_t
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define syncwarp() barrier(CLK_LOCAL_MEM_FENCE)

inline void concat(gemv_, fpp)(
        __global const fp_t* restrict mx,
        __global const fp_t* restrict vc,
        __local        fp_t* restrict wc,
        const int32_t n, const int32_t m,
        __global       fp_t* restrict rs) {
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

__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __local        fp_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m,
        __global fp_t rs[/*m*/]) {
    concat(gemv_, fpp)(mx, vc, wc, n, m, rs);
}
