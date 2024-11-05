#ifndef _RSQRTF_WITHOUT_UNARY_H_
#define _RSQRTF_WITHOUT_UNARY_H_

#include "function.h"
#include "typehint.h"

inline float8_128 __dlc_rsqrtf_without_unary(float8_128 a) {
    int8_128 valueNoSignBit = v_u32_and(*(int8_128*)(&a), 0x7fffffff);
    bool8_128 is_neg = v_f32_cmp(LS, a, 0.0);
    bool8_128 is_zero = v_f32_cmp(EQ, a, 0.0);
    bool8_128 is_inf = v_s32_cmp(EQ, valueNoSignBit, 0x7f800000);
    bool8_128 is_nan = v_s32_cmp(GT, valueNoSignBit, 0x7f800000);

    float8_128 half_a = v_f32_mul_b(*(float8_128*)(&valueNoSignBit), 0.5);
    valueNoSignBit = v_s32_sub(0x5f3759df, v_u32_shr(valueNoSignBit, 1));
    
    float8_128 result = *(float8_128*)(&valueNoSignBit);
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_a, v_f32_mul_b(result, result))));
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_a, v_f32_mul_b(result, result))));
    result = v_f32_mul_b(result, v_f32_sub_b(1.5, v_f32_mul_b(half_a, v_f32_mul_b(result, result))));

    int8_128 signbit = v_u32_and(*(int8_128*)(&a), 0x80000000);
    int8_128 tmp_1 = v_u32_or(signbit, 0x7f800000);
    int8_128 tmp_2 = v_u32_or(signbit, 0);
    int8_128 tmp_3 = v_u32_or(signbit, 0x7fc00000);

    result = v_f32_sel(is_inf, result, *(float8_128*)(&tmp_2));
    result = v_f32_sel(is_zero, result, *(float8_128*)(&tmp_1));
    result = v_f32_sel(is_nan, result, *(float8_128*)(&tmp_3));
    result = v_f32_sel(is_neg, result, *(float8_128*)(&tmp_3));
    return result;
}

#endif