#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FSQRT_RD_WITHOUT_UNARY_H_X86_
#define _FSQRT_RD_WITHOUT_UNARY_H_X86_

#include "function.h"


inline float8_128 __dlc_fsqrt_rd_without_unary(float8_128 a) {
    int8_128 valueNoSignBit = v_u32_and(*(int8_128*)(&a), 0x7fffffff);

    bool8_128 is_neg = v_f32_cmp(LSEQ, a, 0.0);
    bool8_128 is_zero = v_f32_cmp(EQ, a, 0);
    bool8_128 is_inf_or_nan = v_s32_cmp(GTEQ, valueNoSignBit, 0x7f800000);

    float8_128 half_a = v_f32_mul_b(*(float8_128*)(&valueNoSignBit), 0.5);
    valueNoSignBit = v_s32_sub(0x5f3759df, v_u32_shr(valueNoSignBit, 1));
    float8_128 result = *(float8_128*)(&valueNoSignBit);
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_a, v_f32_mul_b(result, result))));
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_a, v_f32_mul_b(result, result))));
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_a, v_f32_mul_b(result, result))));

    result = v_f32_mul_b(result, result);

    float8_128 half_result = v_f32_mul_b(result, 0.5);
    int8_128 i_result = v_s32_sub(0x5f3759df, v_u32_shr(*(int8_128*)(&result), 1));
    result = *(float8_128*)(&i_result);
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_result, v_f32_mul_b(result, result))));
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_result, v_f32_mul_b(result, result))));
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_result, v_f32_mul_b(result, result))));

    int8_128 signbit = v_u32_and(*(int8_128*)(&a), 0x80000000);
    int8_128 result_temp = v_u32_or(*(int8_128*)(&result), signbit);
    result = *(float8_128*)(&result_temp);

    int8_128 tmp_1 = v_u32_or(signbit, 0x7fc00000);
    int8_128 tmp_2 = v_u32_or(signbit, 0);

    result = v_f32_sel(is_zero, result, *(float8_128*)(&tmp_2));
    result = v_f32_sel(is_inf_or_nan, result, (a * a));
    result = v_f32_sel(is_neg, result, *(float8_128*)(&tmp_1));
    return result;
}


#endif