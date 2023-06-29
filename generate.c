#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "rt.h"
#if 0
clCreateImage2D
clCreateImage3D
clEnqueueMarker
clEnqueueWaitForEvents
clEnqueueBarrier
clUnloadCompiler
clGetExtensionFunctionAddress
clCreateCommandQueue
clCreateSampler
clEnqueueTask
clGetGLContextInfoKHR
clCreateFromGLBuffer
clCreateFromGLTexture
clCreateFromGLRenderbuffer
clGetGLObjectInfo
clGetGLTextureInfo
clEnqueueAcquireGLObjects
clEnqueueReleaseGLObjects
clCreateFromGLTexture2D
clCreateFromGLTexture3D
#endif


#define null ((void*)0)

#define strequ(s1, s2) (strcmp((s1), (s2)) == 0)

static bool starts_with(const char* s1, const char* s2) { return strstr(s1, s2) == s1; }

static char* skip_whitespace(char* p) {
    while (*p <= 0x20) { p++; }
    return p;
}

static char* next_token(char* *s) {
    char* p = skip_whitespace(*s);
    char* token = p;
    while (*p > 0x20) { p++; }
    *s = skip_whitespace(p);
    return token;
}

static char* trim(char* s) {
    char* p = skip_whitespace(s);
    char* q = p + strlen(p) - 1;
    while (*q <= 0x20) { *q = 0; q--; }
    return p;
}

static char* callback_name(char* start, char* *name) {
    char* n = *name;
    while (n >= start && !starts_with(n, "CL_CALLBACK *")) { n--; }
    if (starts_with(n, "CL_CALLBACK *")) {
        // foo (CL_CALLBACK * bar)(params)
        *name = n - 2;
        static char param[256];
        n += strlen("CL_CALLBACK * ");
        char* e = strchr(n, ')');
        memcpy(param, n, e - n);
        param[e - n] = 0;
        return param;
    }
    assert(false);
    return null;
}

static char* signature_to_args(char* signature) {
    static char args[1024];
    args[0] = 0;
    char* a = args;
    char* e = signature + strlen(signature) - 1;
    assert(signature[0] == '(' && *e == ')');
    char* p = signature + 1;
    for (;;) {
        char* q = strchr(p, ',');
        char* cb = strstr(p, ")("); // callback
        if (cb != null && cb < e && (q == null || cb < q)) {
            q = cb;
        }
        if (q == null || q > e) { q = e; }
        char* name = q - 1;
        while (*name != '(' && *name > 0x20) { name--; }
        int n = (int)strlen(args);
        if (n < sizeof(args) - (q - name)) {
            name++;
            size_t k = q - name;
            a = args + n;
            memcpy(a, name, q - name);
            if (k > 2 && starts_with(a + k - 2, "[]")) {
                k -= 2;
            }
            if (q == e) {
                a[k] = 0;
            } else {
                a[k    ] = ',';
                a[k + 1] = 0x20;
                a[k + 2] = 0;
            }
        }
        if (cb == q) {
            // "void (CL_CALLBACK * foo)(type p1, type p2, ...)"
            q += 3;
            while (q < e && *q != ')') { q++; }
            q++;
        }
        if (q == e) { break; }
        p = q + 1;
    }
    return trim(args);
}

static void generate_function(char* return_type, char* function) {
    return_type = trim(return_type);
    function = trim(function);
    char* p = strchr(function, '(');
    if (p != null) {
        static char type_def[2048];
        char* signature = p;
        static char name[256];
        if (p - function < sizeof(name) - 1) {
            memcpy(name, function, p - function);
            name[p - function] = 0;
//          println("%s", name);
            snprintf(type_def, sizeof(type_def), "typedef %s (*%s_t_)%s;",
                return_type, name, signature);
            char* args = signature_to_args(signature);
            char  _return_if_null[256];
            sprintf(_return_if_null,
                "return f == null ? (%s)null :\n        ", return_type);
            char* _return = strequ(return_type, "void") ? "" :
                           (strequ(return_type, "cl_int") ?
                            "return f == null ? CL_FUNCTION_NOT_IMPLEMENTED :\n        " :
                            _return_if_null);
            args = trim(args);
            printf("%s %s%s {\n"
                "    %s\n"
                "    static %s_t_ f;\n"
                "    if (f == null) { f = (%s_t_)clBindFunction(\"%s\"); }\n"
                "    %sf(%s);\n"
                "}\n\n",
                return_type, name, signature,
                type_def,
                name, name, name, _return, strequ(args, "void") ? "" : args
            );
        }
    }
}

static void parse_text(char* p) {
    for (;;) {
        p = strstr(p, "extern CL_API_ENTRY");
        if (p == null) { break; }
        char* next = next_token(&p);
        assert(starts_with(next, "extern"));
        next = next_token(&p);
        assert(starts_with(next, "CL_API_ENTRY"));
        if (starts_with(next, "CL_API_ENTRY")) {
            next = next_token(&p);
            if (starts_with(next, "CL_API_PREFIX_")) {
                // CL_API_PREFIX__*_DEPRECATED
                next = next_token(&p); // skip "CL_API_PREFIX__*_DEPRECATED"
            }
            char* return_type = next;
            char* call = strstr(next, "CL_API_CALL");
            if (call != null) {
                static char rt[256];
                if (call - return_type < sizeof(rt) - 1) {
                    memcpy(rt, return_type, call - return_type);
                    rt[call - return_type] = 0;
                    next_token(&call); // skip CL_API_CALL
                    p = call;
                    char* suffix = strstr(p, ")""\x20""CL_API_SUFFIX");
                    // some files have this instead of
                    // CL_API_SUFFIX__VERSION_*_*
                    char* close  = strstr(p, ")\x20;\n");
                    if (close != null && (suffix == null || close < suffix)) {
                        suffix = close + 2;
                    } else if (suffix != null) {
                        suffix += 2; // skip ") "
                    }
                    if (suffix != null) {
                        static char function[16 * 1024];
                        if (suffix - call < sizeof(function) - 1) {
                            memcpy(function, call, suffix - call - 1);
                            function[suffix - call - 1] = 0;
                            generate_function(rt, function);
                        }
                    } else {
                        assert(false); // need to read header and figure out new generation strategy
                    }
                }
            }
        }
    }
}

static int parse_file(const char* name) {
    char filename[256];
    sprintf(filename, "CL/%s", name);
    FILE* f = fopen(filename, "r");
    if (f == null) {
        sprintf(filename, "../CL/%s", name);
        f = fopen(filename, "r");
    }
    if (f != null) {
//      println("file: %s", filename);
        static char text[1024 * 1024];
        int k = (int)fread(text, 1, sizeof(text) - 1, f);
        text[k] = 0;
        parse_text(text);
        fclose(f);
        return 0;
    } else {
        fprintf(stderr, "CL/%s is not found\n", name);
        fprintf(stderr, "download from: https://github.com/KhronosGroup/OpenCL-Headers\n");
        fprintf(stderr, "and place into CL sufolder.\n");
        assert(f != null);
        return 1;
    }
}

int main(int argc, const char* argv[]) {
    (void)argc; (void)argv;
    printf("/* DO NOT EDIT. THIS FILE IS GENERATED BY generate.c */\n\n");
    printf("#define CL_FUNCTION_NOT_IMPLEMENTED -255\n\n");
    printf("extern void* clBindFunction(const char* name);\n\n");
    printf("#ifndef null\n");
    printf("#define null ((void*)0) // like nullptr a bit better than (0)\n");
    printf("#endif\n\n");
    // in case output is compiled by C++ compiler
    printf("#ifdef __cplusplus\n");
    printf("extern \"C\" {\n");
    printf("#endif\n\n");
    static const char* names[] = {
        "cl.h",
        "cl_d3d10.h",
        "cl_d3d11.h",
        "cl_dx9_media_sharing.h",
//      "cl_dx9_media_sharing_intel.h",
        "cl_egl.h",
        "cl_ext.h",
//      "cl_ext_intel.h",
//      "cl_function_types.h",
        "cl_gl.h",
        "cl_half.h",
        "cl_icd.h",
        "cl_layer.h",
//      "cl_va_api_media_sharing_intel.h",
    };
    int r = 0;
    for (int i = 0; i < countof(names) && r == 0; i++) {
        r = parse_file(names[i]);
    }
    printf("#ifdef __cplusplus\n");
    printf("} // extern \"C\"\n");
    printf("#endif\n\n");
    return r;
}
