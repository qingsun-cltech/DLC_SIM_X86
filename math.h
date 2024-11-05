#ifndef _MATH_H_X86_
#define _MATH_H_X86_

#include "dlc-intrinsics.h"
#include "typehint.h"

inline int pre_exp2(int x) { return (1 << x) - 1; }
inline int min(int a, int b) { return (a < b) ? a : b; }
inline int max(int a, int b) { return (a > b) ? a : b; }

inline int roundUp128(int a) { return (a + 127) / 128 * 128; }
inline int roundDown128(int a) { return a / 128 * 128; }

inline int soft_sdiv(int a, int b) {
    int q = 0;
    int s = 0;
    for (int i = 31; i >= 0; i--) {
        int tmp = (a >> i) & 1;
        q = (q << 1) | tmp;
        if (q >= b) {
            s += 1 << i;
            q -= b;
        }
    }
    return s;
}

inline int soft_sdiv_remainder(int a, int b, int *remainder) {
    int q = 0;
    int s = 0;
    for (int i = 31; i >= 0; i--) {
        int tmp = (a >> i) & 1;
        q = (q << 1) | tmp;
        if (q >= b) {
            s += 1 << i;
            q -= b;
        }
    }
    *remainder = q;
    return s;
}
// inline int8_128 soft_sdiv_1024(int8_128 a, int8_128 b) {
//     int8_128 q = 0;
//     int8_128 s = 0;
//     for (int i = 31; i >= 0; i--) {
//         int tmp = (a >> i) & 1;
//         q = (q << 1) | tmp;
//         if (q >= b) {
//             s += 1 << i;
//             q -= b;
//         }
//     }
//     return s;
// }


inline int8_128 soft_sdiv_1024(int8_128 a, int8_128 b) {
    int8_128 q = v_u32_move_i(0);
    int8_128 s = v_u32_move_i(0);
// #pragma clang loop unroll_count(8)
    for (int i = 31; i >= 0; i--) {
        int8_128 tmp = v_u32_and(v_u32_shr(a, v_u32_move_i(i)), v_u32_move_i(1));
        q = (v_u32_shl(q, v_u32_move_i(1))) | tmp;
        bool8_128 update = v_s32_cmp(GTEQ, q, b);
        s = v_s32_sel(update, s, v_s32_add(s, v_u32_move_i(1 << i)));
        q = v_s32_sel(update, q, q - b);
    }
    return s;
}

inline int8_128 soft_sdiv_remainder_1024(int8_128 a, int8_128 b, int8_128 *remainder) {
    int8_128 q = v_u32_move_i(0);
    int8_128 s = v_u32_move_i(0);
// #pragma clang loop unroll_count(8)
    for (int i = 31; i >= 0; i--) {
        int8_128 tmp = v_u32_and(v_u32_shr(a, v_u32_move_i(i)), v_u32_move_i(1));
        q = (v_u32_shl(q, v_u32_move_i(1))) | tmp;
        bool8_128 update = v_s32_cmp(GTEQ, q, b);
        s = v_s32_sel(update, s, v_s32_add(s, v_u32_move_i(1 << i)));
        q = v_s32_sel(update, q, q - b);
    }
    *remainder = q;
    return s;
}

#endif
