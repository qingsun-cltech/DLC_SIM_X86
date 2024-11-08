#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #pragma once
#ifndef _CURAND_H_X86_
#define _CURAND_H_X86_

/*

This `curand.h` has optimized for our kernel usage,
where only `x` in the return results of `curand_uniform4` and `curand_normal4` is valid.
Also, based on the PyTorch,
we made modifications to the `curand_normal4` to meet our needs.
If a consistent curand behavior with CUDA is required, please use `curand_cuda.h`.

*/

#include "../libdevice/fsqrt_rd_without_unary.h"
#include "../libdevice/logf_without_unary.h"
#include "../libdevice/sincosf.h"
#include "../libdevice/uint2float_rz.h"
// #include "typehint.h"

typedef struct _uint4 {
    int8_128 x;
    int8_128 y;
    int8_128 z;
    int8_128 w;
} uint4;

inline uint4 make_uint4(int8_128 x, int8_128 y, int8_128 z, int8_128 w) {
    uint4 r;
    r.x = x;
    r.y = y;
    r.z = z;
    r.w = w;
    return r;
}

typedef struct _uint2 {
    int8_128 x;
    int8_128 y;
} uint2;

inline uint2 make_uint2(int8_128 x, int8_128 y) {
    uint2 r;
    r.x = x;
    r.y = y;
    return r;
}

typedef struct _float4 {
    float8_128 x, y, z, w;
} float4;

typedef struct curandStatePhilox4_32_10 {
    uint4 ctr;
    uint2 key;
    int8_128 STATE;
} curandStatePhilox4_32_10_t;

const int kPhilox10A = 0x9E3779B9;
const int kPhilox10B = 0xBB67AE85;
const int kPhiloxSA = 0xD2511F53;
const int kPhiloxSB = 0xCD9E8D57;

// inline int8_128 u32mul(int8_128 a, int8_128 b, int8_128 *hi) {
//     // Extract bytes from input values
//     int8_128 /*__attribute__((address_space(2)))*/ a_bytes[4];
//     int8_128 /*__attribute__((address_space(2)))*/ b_bytes[4];
//     for (int i = 0; i < 4; ++i) {
//         a_bytes[i] = (a >> (i * 8)) & 0xFF;
//         b_bytes[i] = (b >> (i * 8)) & 0xFF;
//         Print("a_bytes[i] : %h\n",a_bytes[i]);
//         // Print("b_bytes[i] : %h\n",b_bytes[i]);
//     }

//     // Perform multiplications and accumulate partial results
//     int8_128 c[8] = {0}; // Array to store partial results and carry
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             c[i + j] += a_bytes[i] * b_bytes[j];
//             // Print("c[i + j] : %d\n",c[i + j]);
//         }
//     }

//     // Reduce partial results and compute final values
//     int8_128 p[8] = {0};
//     int8_128 carry = 0; // Initialize carry value
//     for (int i = 0; i < 7; ++i) {
//         c[i] += carry;      // Add carry from previous iteration
//         p[i] = c[i] & 0xFF; // Current reduced result
//         carry = c[i] >> 8;  // Calculate new carry value
//     }
//     p[7] = (c[7] + carry) & 0xFF; // Last reduced result with carry

//     // Combine reduced results into final output values
//     int8_128 lo = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
//     *hi = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);

//     // int8_128 hig = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);

//     return lo;
// }

inline int8_128 u32mullo(int8_128 a, int8_128 b) {
    // Extract bytes from input values
    int8_128 /*__attribute__((address_space(2)))*/ a_bytes[4];
    int8_128 /*__attribute__((address_space(2)))*/ b_bytes[4];
    for (int i = 0; i < 4; ++i) {
        a_bytes[i] = (a >> (i * 8)) & 0xFF;
        b_bytes[i] = (b >> (i * 8)) & 0xFF;
    }

    // Perform multiplications and accumulate partial results
    int8_128 c[4] = {0}; // Array to store partial results and carry

    c[0] = a_bytes[0] * b_bytes[0];
    c[1] = a_bytes[0] * b_bytes[1] + a_bytes[1] * b_bytes[0];
    c[2] = a_bytes[0] * b_bytes[2] + a_bytes[1] * b_bytes[1] + a_bytes[2] * b_bytes[0];
    c[3] = a_bytes[0] * b_bytes[3] + a_bytes[1] * b_bytes[2] + a_bytes[2] * b_bytes[1] + a_bytes[3] * b_bytes[0];

    // Reduce partial results and compute final values
    int8_128 p[4] = {0};
    int8_128 carry = 0; // Initialize carry value
    for (int i = 0; i < 4; ++i) {
        c[i] += carry;      // Add carry from previous iteration
        p[i] = c[i] & 0xFF; // Current reduced result
        carry = v_u32_shr(c[i], 8);  // Calculate new carry value
    }

    // Combine reduced results into final output values
    int8_128 lo = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
    return lo;
}

inline int8_128 u32mul(int8_128 a, int8_128 b, int8_128 *hi) {

    // Extract bytes from input values
    int8_128 /*__attribute__((address_space(2)))*/ a_bytes[3];
    int8_128 /*__attribute__((address_space(2)))*/ b_bytes[3];
    for (int i = 0; i < 3; ++i) {
        a_bytes[i] = (v_u32_shr(a, (i * 11))) & 0x7FF;
        b_bytes[i] = (v_u32_shr(b, (i * 11))) & 0x7FF;
    }

    // Perform multiplications and accumulate partial results
    int8_128 c[6] = {0}; // Array to store partial results and carry
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            c[i + j] += a_bytes[i] * b_bytes[j];
        }
    }
    // Reduce partial results and compute final values
    int8_128 p[6] = {0};
    int8_128 carry = 0; // Initialize carry value
    for (int i = 0; i < 5; ++i) {
        c[i] += carry; // Add carry from previous iteration
        p[i] = c[i] & 0x7FF; // Current reduced result
        carry = v_u32_shr(c[i] , 11); // Calculate new carry value
    }
    p[5] = (c[5] + carry) & 0x7FF; // Last reduced result with carry
    // Combine reduced results into final output values
    int8_128 lo = p[0] | (p[1] << 11) | (p[2] << 22) ;
    *hi = v_u32_shr(p[2] , 10) | (p[3] << 1) | (p[4] << 12) | (p[5] << 23);
    return lo;
}






// inline int8_128 u32mul(int8_128 a, int8_128 b, int8_128 *hi) {
//     int8_128 c = v_u32_move_i(0);
//     int8_128 d = v_u32_move_i(0);
// lo .u32add (b >> n) & 1 == 1 ? a << n : 0
// cf (a & b) | ((a ^ b) & !c)
// hi .u32add (b >> n) & 1 == 1 ? a >> (32 - n) : 0
// #define ITEM(n) \
//     { \
//         bool8_128 plus = \
//             v_s32_cmp(EQ, v_u32_and(v_u32_shr(b, v_u32_move_i(n)), v_u32_move_i(1)), v_u32_move_i(1)); \
//         int8_128 addlo = v_s32_sel(plus, v_u32_move_i(0), v_u32_shl(a, v_u32_move_i(n))); \
//         int8_128 addhi = v_s32_sel(plus, v_u32_move_i(0), v_u32_shr(a, v_u32_move_i(32 - n))); \
//         int8_128 lhi = v_u32_shr(c, v_u32_move_i(31)); \
//         int8_128 rhi = v_u32_shr(addlo, v_u32_move_i(31)); \
//         c = v_s32_add_non_clamp(c, addlo); \
//         int8_128 chi = v_u32_xor(v_u32_shr(c, v_u32_move_i(31)), v_u32_move_i(1)); \
//         int8_128 carry = (lhi & rhi) | ((lhi ^ rhi) & chi); \
//         d = v_s32_add_non_clamp(d, v_s32_add_non_clamp(addhi, carry)); \
//     }
// #define ITEM2(n) ITEM(n) ITEM(n + 1)
// #define ITEM4(n) ITEM2(n) ITEM2(n + 2)
// #define ITEM8(n) ITEM4(n) ITEM4(n + 4)
// #define ITEM16(n) ITEM8(n) ITEM8(n + 8)
// #define ITEM32(n) ITEM16(n) ITEM16(n + 16)
//     for (int n = 0; n < 32; n += 1) {
//         // unroll will cause not store yet error
//         ITEM(n)
//     }
// #undef ITEM32
// #undef ITEM16
// #undef ITEM8
// #undef ITEM4
// #undef ITEM2
// #undef ITEM
//     *hi = d;
//     return c;
// }

// inline int8_128 u32mullo(int8_128 a, int8_128 b) {
//     int8_128 c = v_u32_move_i(0);
// // int8_128 d = v_u32_move_i(0);
// // lo .u32add (b >> n) & 1 == 1 ? a << n : 0
// // cf (a & b) | ((a ^ b) & !c)
// // hi .u32add (b >> n) & 1 == 1 ? a >> (32 - n) : 0
// #define ITEM(n)                                                                                              \
//     {                                                                                                        \
//         bool8_128 plus =                                                                                     \
//             v_s32_cmp(EQ, v_u32_and(v_u32_shr(b, v_u32_move_i(n)), v_u32_move_i(1)), v_u32_move_i(1));       \
//         int8_128 addlo = v_s32_sel(plus, v_u32_move_i(0), v_u32_shl(a, v_u32_move_i(n)));                    \
//         c = v_s32_add_non_clamp(c, addlo);                                                                   \
//     }
// #define ITEM2(n) ITEM(n) ITEM(n + 1)
// #define ITEM4(n) ITEM2(n) ITEM2(n + 2)
// #define ITEM8(n) ITEM4(n) ITEM4(n + 4)
// #define ITEM16(n) ITEM8(n) ITEM8(n + 8)
// #define ITEM32(n) ITEM16(n) ITEM16(n + 16)
//     // // #pragma clang loop unroll_count(8)
//     for (int n = 0; n < 32; n += 1) {
//         // unroll will cause not store yet error
//         ITEM(n)
//     }
// #undef ITEM32
// #undef ITEM16
// #undef ITEM8
// #undef ITEM4
// #undef ITEM2
// #undef ITEM
//     return c;
// }

inline int8_128 murmurhash3_32(int8_128 key) {

    key = v_u32_xor(key, v_u32_shr(key, 16));

    key = u32mullo(v_u32_move_i(0x85ebca6b), key);

    key = v_u32_xor(key, v_u32_shr(key, 13));

    key = u32mullo(v_u32_move_i(0xc2b2ae35), key);

    key = v_u32_xor(key, v_u32_shr(key, 16));
    return key;
}


inline uint4 single_round(uint4 ctr, uint2 key) {
    int8_128 hi0 = v_u32_move_i(0);
    int8_128 lo0 = u32mul(v_u32_move_i(kPhiloxSA), ctr.x, &hi0);
    int8_128 hi1 = v_u32_move_i(0);
    int8_128 lo1 = u32mul(v_u32_move_i(kPhiloxSB), ctr.z, &hi1);

    uint4 ret;
    ret.x = v_u32_xor(v_u32_xor(hi1, ctr.y), key.x);
    ret.y = lo1;
    ret.z = v_u32_xor(v_u32_xor(hi0, ctr.w), key.y);
    ret.w = lo0;

    return ret;
}

inline uint4 curand_Philox4x32_10(uint4 c, uint2 k) {
    c = single_round(c, k); // 1
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 2
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 3
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 4
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 5
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 6
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 7
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 8
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 9
    k.x = v_s32_add_non_clamp(k.x, v_u32_move_i(kPhilox10A));
    k.y = v_s32_add_non_clamp(k.y, v_u32_move_i(kPhilox10B));
    c = single_round(c, k); // 10
    return c;
}

inline uint2 u64div4(uint2 n) {
    n.x = v_u32_or(v_u32_shr(n.x, v_u32_move_i(2)), v_u32_shl(n.y, v_u32_move_i(30)));
    n.y = v_u32_shr(n.y, v_u32_move_i(2));
    return n;
}

inline void curand_init(uint2 seed, uint2 subsequence, uint2 offset, curandStatePhilox4_32_10_t *state) {
    uint2 offd4 = u64div4(offset);
    state->ctr = make_uint4(offd4.x, offd4.y, subsequence.x, subsequence.y);
    state->key = seed;
    state->STATE = v_u32_and(offset.x, v_u32_move_i(3));
}

const float CURAND_2POW32_INV = 2.3283064e-10f;
const float CURAND_2POW32_INV_HALF = 1.1641532e-10f;

const float CURAND_2POW32_INV_2PI = (2.3283064e-10f * 6.2831855f);

inline float4 _curand_uniform4(uint4 x) {
    float4 y;
    float4 x_f;

    x_f.x = __dlc_uint2float_rz(x.x);

    y.x = v_f32_mul_b(x_f.x, v_u32_move_f(CURAND_2POW32_INV));

    float CURAND_2POW32_INV_Half = CURAND_2POW32_INV / 2.0;

    y.x = v_f32_add_b(y.x, v_u32_move_f(CURAND_2POW32_INV_Half));

    return y;
}

inline uint4 curand4(curandStatePhilox4_32_10_t *state) {
    uint4 out = curand_Philox4x32_10(state->ctr, state->key);
    bool8_128 cond1 = v_s32_cmp(EQ, state->STATE, v_u32_move_i(1));
    bool8_128 cond2 = v_s32_cmp(EQ, state->STATE, v_u32_move_i(2));
    bool8_128 cond3 = v_s32_cmp(EQ, state->STATE, v_u32_move_i(3));
    out.x = v_s32_sel(cond1, out.x, out.y);
    out.x = v_s32_sel(cond2, out.x, out.z);
    out.y = v_s32_sel(cond2, out.y, out.w);
    out.x = v_s32_sel(cond3, out.x, out.w);
    return out;
}

inline int8_128 curand(curandStatePhilox4_32_10_t *state) { return curand4(state).x; }

inline float4 curand_uniform4(curandStatePhilox4_32_10_t *state) { return _curand_uniform4(curand4(state)); }

inline float2 _curand_box_muller(int8_128 x, int8_128 y) {
    float8_128 u = __dlc_uint2float_rz(x);
    float8_128 v = __dlc_uint2float_rz(y);
    u = v_f32_mul_b(u, v_u32_move_f(CURAND_2POW32_INV));
    v = v_f32_mul_b(v, v_u32_move_f(CURAND_2POW32_INV_2PI));
    u = v_f32_add_b(u, v_u32_move_f(CURAND_2POW32_INV / 2.0f));
    v = v_f32_add_b(v, v_u32_move_f(CURAND_2POW32_INV_2PI / 2.0f));
    float8_128 s =
        __dlc_fsqrt_rd_without_unary(v_f32_mul_b(__dlc_logf_without_unary(u), v_u32_move_f(-2.0f)));
    float2 res = __dlc_sincosf(v);
    res.x = v_f32_mul_b(res.x, s);
    res.y = v_f32_mul_b(res.y, s);
    return res;
}

inline float4 curand_normal4(curandStatePhilox4_32_10_t *state) {
    bool8_128 chooseY = v_s32_cmp(EQ, v_u32_and(state->STATE, v_u32_move_i(1)), v_u32_move_i(1));
    state->STATE = v_u32_and(state->STATE, v_u32_move_i(0xfffffffe));
    uint4 x = curand4(state);
    float4 res;
    float2 res1 = _curand_box_muller(x.x, x.y);
    res.x = v_f32_sel(chooseY, res1.x, res1.y);
    return res;
}

#endif
