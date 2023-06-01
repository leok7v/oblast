#pragma once
#include "rt.h"

#ifdef cplusplus
extern "C" {
#endif

typedef struct dot_if {
    fp64_t (*fp16)(const fp16_t* v0, int64_t s0, const fp16_t* v1, int64_t s1, int64_t n);
    fp64_t (*fp32)(const fp32_t* v0, int64_t s0, const fp32_t* v1, int64_t s1, int64_t n);
    fp64_t (*fp64)(const fp64_t* v0, int64_t s0, const fp64_t* v1, int64_t s1, int64_t n);
    fp64_t (*bf16)(const bf16_t* v0, int64_t s0, const bf16_t* v1, int64_t s1, int64_t n);
    void   (*test)(void); // can be null
} dot_if;

extern dot_if dot;

#ifdef cplusplus
} // extern "C"
#endif
