#ifndef __MATMUL_T_H_X86__
#define __MATMUL_T_H_X86__

#include "ldst.h"

#include "dlc-intrinsics.h"
#include "typehint.h"

inline float128_128 add128_128(float128_128 s, float128_128 a) {
    float8_128 r0 = v_f32_add_b(sub_vector(s, 0), sub_vector(a, 0));
    float8_128 r1 = v_f32_add_b(sub_vector(s, 1), sub_vector(a, 1));
    float8_128 r2 = v_f32_add_b(sub_vector(s, 2), sub_vector(a, 2));
    float8_128 r3 = v_f32_add_b(sub_vector(s, 3), sub_vector(a, 3));
    float8_128 r4 = v_f32_add_b(sub_vector(s, 4), sub_vector(a, 4));
    float8_128 r5 = v_f32_add_b(sub_vector(s, 5), sub_vector(a, 5));
    float8_128 r6 = v_f32_add_b(sub_vector(s, 6), sub_vector(a, 6));
    float8_128 r7 = v_f32_add_b(sub_vector(s, 7), sub_vector(a, 7));
    float8_128 r8 = v_f32_add_b(sub_vector(s, 8), sub_vector(a, 8));
    float8_128 r9 = v_f32_add_b(sub_vector(s, 9), sub_vector(a, 9));
    float8_128 r10 = v_f32_add_b(sub_vector(s, 10), sub_vector(a, 10));
    float8_128 r11 = v_f32_add_b(sub_vector(s, 11), sub_vector(a, 11));
    float8_128 r12 = v_f32_add_b(sub_vector(s, 12), sub_vector(a, 12));
    float8_128 r13 = v_f32_add_b(sub_vector(s, 13), sub_vector(a, 13));
    float8_128 r14 = v_f32_add_b(sub_vector(s, 14), sub_vector(a, 14));
    float8_128 r15 = v_f32_add_b(sub_vector(s, 15), sub_vector(a, 15));
    return v_concat_16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
}

#endif