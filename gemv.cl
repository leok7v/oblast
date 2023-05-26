#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#ifdef fp16_t
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define _concat_(first, last)  first ## last
#define concat(first, last) _concat_(first, last)

#define sync() barrier(CLK_LOCAL_MEM_FENCE)


__kernel
void concat(gemv, fpp)( // gemv32 gemv16 gemv64; fpp float point precision
        __global const fp_t mx[/*m][n*/],
        __global const fp_t vc[/*n*/],
        __local        fp_t wc[/*items*/], // work copy, local dot product
        const int32_t n, const int32_t m,
        __global fp_t rs[/*m*/]) {
    enum { warp = 32 };
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint items = get_local_size(0);
    const uint groups = get_num_groups(0);
    for (uint y = gid; y < m; y += groups) {
        const __global fp_t* row = mx + y * n;
        fp_t sum = 0;
        // TODO: vec4 optimization
        for (uint x = lid; x < n; x += items) {
            sum += row[x] * vc[x];
        }
        wc[lid] = sum;
        sync();
        uint id = lid & (warp - 1);
        fp_t wr = 0.0f; // each warp reduces 64 consecutive elements
        if (lid < items / 2) {
            volatile __local fp_t* p = wc + 2 * lid - id;
            p[0] += p[32];
            p[0] += p[16];
            p[0] += p[8];
            p[0] += p[4];
            p[0] += p[2];
            p[0] += p[1];
            wr = p[0];
        }
        sync();
        if (id == 0) {
            wc[lid / warp] = wr;
        }
        sync();
        uint size = items / (2 * warp);
        if (lid < size / 2) {
            volatile __local fp_t* p = wc + lid;
//          // cannot asume that get_local_size(0) is constant
//          // the code below assumes get_local_size(0) <= 2048
//          if (/* items > 1024 && */ size >= 32) { p[0] += p[16]; }
//          if (/* items > 512  && */ size >= 16) { p[0] += p[8]; }
//          if (/* items > 256  && */ size >= 8)  { p[0] += p[4]; }
//          if (/* items > 128  && */ size >= 4)  { p[0] += p[2]; }
//          if (/* items > 64   && */ size >= 2)  { p[0] += p[1]; }
            #pragma unroll
            while (size >= 2) {
                size >>= 1;
                p[0] += p[size];
            }

        }
        if (lid == 0) { rs[y] = wc[0]; }
        sync();
    }
}

// https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclMatVecMul/oclMatVecMul.cl
