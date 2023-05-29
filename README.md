# o`blast - Oh Basic Linear Algebra Subrotines/Subprograms/Functions TINY

![Build](https://github.com/leok7v/oblast/workflows/build/badge.svg)

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
