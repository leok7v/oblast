#include <stdbool.h>
#include "CL/cl_platform.h"
#include "CL/opencl.h"
#include "CL/cl_d3d11.h"
#include "CL/cl_d3d10.h"
#include "CL/cl_dx9_media_sharing.h"
#include "CL/cl_egl.h"
#include "CL/cl_ext.h"
#include "CL/cl_function_types.h"
#include "CL/cl_gl.h"
#include "CL/cl_half.h"
#include "CL/cl_icd.h"
#include "CL/cl_layer.h"

#ifdef OCL_USE_NVIDIA_12_LIB_BINDINGS
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64\OpenCL.lib
#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64/OpenCL.lib")
#else // dynamic bindings with OpenCL.dll on the path
#include "CL/cl_bind.inc" // dynamically bind everything
#endif

// C:\Windows\System32\OpenCL.dll
// File description OpenCL Client DLL
// Type Application extension
// File version 3.0.3.0
// Product name Khronos OpenCL ICD Loader
// Product version 3.0.3.0
// Copyright Copyright © The Khronos Group Inc 2016-2023
// Date modified 1/20/2023 3:20 PM
// Size: 1.41 MB (1,487,336 bytes)
// Signed by
// NVIDIA Sunday, January 15, 2023 11:18:06 AM
// and
// Microsoft Friday, January 20, 023 4:20:45 PM

#define OpenCL_dll "C:/Windows/System32/OpenCL.dll"

// TODO: alternatives for Intel only GPU and AMD GPU? Do we need them?
// e.g.:
// C:\Windows\system32\Intel_OpenCL_ICD64.dll
// C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_9eaeaf7bfb6c744b
// #define OpenCL_dll "Intel_OpenCL_ICD64.dll"

static void* ocl_dl; // OpenCL dynamic library

#ifndef LoadLibrary
#pragma comment(lib, "kernel32")
void*    __stdcall LoadLibraryA(const char* pathname);
void*    __stdcall GetProcAddress(void* module, const char* pathname);
#endif

void* clBindFunction(const char* name) {
    typedef void* (*load_dl_t)(const char* pathname); // dlopen | LoadLibrary
    typedef void* (*find_symbol_t)(void* dl, const char* symbol); // dlsym | GetProcAddress
    load_dl_t load_dl = (load_dl_t)LoadLibraryA;
    find_symbol_t find_symbol = (find_symbol_t)GetProcAddress;
    static bool initialized;
    if (!initialized) {
        ocl_dl = load_dl(OpenCL_dll);
        // if not found, try anywhere else on the path:
        if (ocl_dl == null) { ocl_dl = load_dl(OpenCL_dll); }
        initialized = true;
    }
    return ocl_dl != null ? find_symbol(ocl_dl, name) : null;
}
