#pragma once
#include "rt.h"

#ifdef cplusplus
extern "C" {
#endif

fp64_t dot16(const fp16_t* v0, int64_t s0, const fp16_t* v1, int64_t s1, int64_t n);
fp64_t dot32(const fp32_t* v0, int64_t s0, const fp32_t* v1, int64_t s1, int64_t n);
fp64_t dot64(const fp64_t* v0, int64_t s0, const fp64_t* v1, int64_t s1, int64_t n);

void dot_init();
void dot_test();

#ifdef cplusplus
} // extern "C"
#endif
