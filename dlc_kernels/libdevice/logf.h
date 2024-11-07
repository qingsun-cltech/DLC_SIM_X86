#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _LOGF_H_X86_
#define _LOGF_H_X86_



inline float8_128 __dlc_logf(float8_128 a) {
    float8_128 result = v_f32_log(a);
    result = v_f32_mul_b(result, 0.6931472);
    result = v_f32_sel(v_f32_cmp(EQ, result, 0x80000000), result, 0xff800000);
    result = v_f32_sel(v_f32_cmp(EQ, a, 0), result, 0xff800000);
    return result;
}

#endif