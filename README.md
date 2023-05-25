# o`blast - Oh Basic Linear Algebra Subrotines/Subprograms/Functions TINY

![Build](https://github.com/leok7v/oblast/workflows/build/badge.svg)

### goal

- [ ] implement gemv(matrix[m][n], vector[n]) in a most effecient manner on OpenCL for fp32_t

### progres

- [x] OpenCL header used from: 
   https://github.com/KhronosGroup/OpenCL-Headers/tree/main/CL
- [x] OpenCL.dll exists on Windows and routes to both Intel and Nvidia drivers.
- [x] Not using any OpenCL.lib
- [x] Generated dynamic bindings using GetProcAddress and trivial heared parsing in generate.exe.
- [x] ocl.* interface is simplified fail fast shim on top of OpenCL
- [x] Trivial host fp16_t support just to verify GPU fp16 (not bfloat16!) results
- [x] AVX2/AVX512 dot() vector product
- [ ] implement gemv()

### references

https://github.com/leok7v/OpenCL
