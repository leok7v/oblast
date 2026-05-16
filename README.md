# o`blast - Oh Basic Linear Algebra Subrotines/Subprograms/Functions TINY

[![build](https://github.com/leok7v/oblast/actions/workflows/build.yml/badge.svg)](https://github.com/leok7v/oblast/actions/workflows/build.yml)
[![test](https://github.com/leok7v/oblast/actions/workflows/test.yml/badge.svg)](https://github.com/leok7v/oblast/actions/workflows/test.yml)

### goal

- [ ] implement gemv(matrix[m][n], vector[n]) in a most effecient manner on OpenCL for fp32_t

### progress

- [x] OpenCL header used from: 
   https://github.com/KhronosGroup/OpenCL-Headers/tree/main/CL
- [x] OpenCL.dll exists on Windows and routes to both Intel and Nvidia drivers.
- [x] Not using any OpenCL.lib
- [x] Generated dynamic bindings using GetProcAddress and trivial header parsing in generate.exe.
- [x] ocl.* interface is simplified fail fast shim on top of OpenCL
- [x] Trivial host fp16_t support just to verify GPU fp16 (not bfloat16!) results
- [x] AVX2/AVX512 dot() vector product
- [ ] implement gemv()

### references

https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html

https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_fp16.html

https://github.com/leok7v/OpenCL

https://github.com/KhronosGroup/SPIR/issues/54

OpenCL C 2.0 spec, section 6.9 paragraph k.

"k. Arguments to kernel functions in a program cannot be declared with the built-in scalar
types bool, half, size_t, ptrdiff_t, intptr_t, and uintptr_t or a struct
and/or union that contain fields declared to be one of these built-in scalar types. The size
in bytes of these types except half are implementation-defined and in addition can also
be different for the OpenCL device and the host processor making it difficult to allocate
buffer objects to be passed as arguments to a kernel declared as pointer to these types.
half is not supported as half can be used as a storage format47 only and is not a data
type on which floating-point arithmetic can be performed."

--

half is difficult:
https://chromium.googlesource.com/external/llvm.org/clang/+/google/stable/test/SemaOpenCL/half.cl


https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/mathFunctions.html

We use the generic type name gentype to indicate that the function can take 
    ```float, float2, float3, float4, float8, float16,``` 
    ```double [1], double2, double3, double4, double8 or double16``` 
as the type for the arguments:

```
gentype fma(gentype a, gentype b, gentype c)
c + a * b
gentype mad(gentype a, gentype b, gentype c)
a * b + c
gentype half_divide(gentype x, gentype y)
gentype native_divide(gentype x, gentype y)
gentype half_recip(gentype x) 1 / x
```

https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html

5.2.4 Geometric Functions
```
half and half{2|3|4}
half dot (gentype p0, gentype p1)

...

fma() vs mad()

fma() (Fused Multiply-Add):
* Syntax: fma(a, b, c)
* Computes the fused multiply-add operation: (a * b) + c
* Supports both floating-point and integer data types.
* Provides higher accuracy and precision for floating-point operations.
* Handles special cases like infinities, NaNs, and denormalized numbers 
  in a specified manner according to the IEEE 754 standard.
* Performs rounding according to the rounding mode set using set_rounding_mode().

mad() (Multiply-Add):
* Syntax: mad(a, b, c)
* Computes the multiply-add operation: (a * b) + c
* Supports only floating-point data types.
* Provides a basic multiply-add operation without additional features like handling special cases or precise rounding.
* Suitable for general-purpose arithmetic calculations.
```

In summary, fma() is more powerful and versatile, supporting both 
floating-point and integer types with better accuracy and rounding control. 
On the other hand, mad() is limited to floating-point types and provides 
a basic multiply-add operation without any specific handling of special 
cases or precise rounding.

Subgroups:

https://github.com/KhronosGroup/OpenCL-Docs/blob/main/ext/cl_khr_subgroups.asciidoc

---

## simd.c — portable SIMD dispatcher

Single-file dispatcher that compiles to one binary per (OS, arch) and
selects the best available ISA tier at runtime via CPUID/HWCAP. The
quantized dot products (Q8_0 / Q4_K / Q5_K / Q6_K against Q8_K
activations) are line-for-line ports of the llama.cpp `ggml-cpu`
reference; the tiled SGEMM kernel is a verbatim port of
kittens.cpu/cpu/tensor.c.

### Files

| File         | Role                                                         |
|--------------|--------------------------------------------------------------|
| `quants.h`   | Block layouts, fp16 helpers, scalar reference dots           |
| `neon.c`     | ARM NEON tiers: `+dotprod` (Apple Silicon, A76+) and baseline (Cortex-A53/A72/A73) |
| `avx.c`      | x86 tiers: AVX-VNNI, AVX2-FMA, AVX1 + F16C HGEMM             |
| `simd.c`     | Dispatcher + tiled SGEMM/HGEMM + self-tests (`-DSIMD_TEST`)  |

Per-kernel ISA tier is selected via `__attribute__((target("...")))`;
no per-host compile flags needed beyond ISA baseline.

### Build

```
# macOS / Linux  (arm64 or x86_64)
clang -O3 -std=c11 -DSIMD_TEST simd.c -o simd_test && ./simd_test

# Windows via MSYS2 (clang from mingw-w64-x86_64-clang)
/mingw64/bin/clang -O3 -std=c11 -DSIMD_TEST -static simd.c -o simd_test.exe

# Android via NDK
$NDK/toolchains/llvm/prebuilt/<host>/bin/clang \
  --target=aarch64-linux-android24 -O3 -DSIMD_TEST simd.c -o simd_test
```

### Benchmarks (median GFlops across 5 fresh runs, M=N=K=512 for GEMMs)

All numerics cross-validated against the scalar reference; max delta
1.5e-05 from accumulation-order rounding (identical across SIMD tiers).

| Chip                         | OS / arch        | ISA tier        | q8_0 | q4k | q5k | q6k | SGEMM | HGEMM |
|------------------------------|------------------|-----------------|-----:|----:|----:|----:|------:|------:|
| Apple M-series               | macOS arm64      | NEON + dotprod  |  102 | 117 |  74 |  63 |    13 |    15 |
| AMD Zen 5 (Strix Halo)       | Linux x86_64     | AVX-VNNI        |   80 |  96 |  74 |  70 |    40 |    33 |
| Intel Haswell i7-4578U       | macOS x86_64     | AVX2 + FMA      |   29 |  26 |  20 |  21 |    19 |     8 |
| Intel Ivy Bridge i7-3615QM   | macOS x86_64     | AVX1            |   15 |  18 |  15 |  15 |    14 |    11 |
| Intel Ivy Bridge i7-3720QM   | macOS x86_64     | AVX1            |   12 |  14 |  12 |  12 |    15 |    12 |
| Intel Ivy Bridge i7-3667U    | Windows x86_64   | AVX1            |    9 |  22 |  17 |  16 |    14 |    11 |
| Qualcomm SD 765G (A76 + A55) | Android arm64    | NEON + dotprod  |   28 |  30 |  19 |  14 |     4 |     2 |
| Amlogic A311D (A73 + A53)    | Ubuntu arm64     | NEON baseline   |    9 |   6 |   6 |   6 |     2 |     2 |

Q-quant numbers are the dispatcher's pick on each chip; older silicon
without dot-product instructions falls back to vmull+vpadalq (NEON) /
maddubs+madd (AVX2) / 128-bit SSE (AVX1) automatically.

**Apple Silicon variance note.** macOS classifies short CPU bursts by
QoS; brief bench loops after I/O (e.g. the sync step in the sweep
script) can be parked on E-cores for the first ~100ms and report
~30% lower throughput. The harness calls
`pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0)` and
warms up the CPU for 300ms before timing to keep the scheduler
honest, but run-to-run variance of ±15% remains. The Apple M row
above is the median of 5 fresh runs; peak observed q4k was 137
GFlops, low 91. Other chips on the list are within ~5% run-to-run.

