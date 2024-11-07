#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef __SINCOSF_PAYNE_H_X86ANEK_H_X86__
#define __SINCOSF_PAYNE_H_X86ANEK_H_X86__

// https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c



struct __sincosf_ph_dfloat {
    float8_128 hi;
    float8_128 lo;
};

struct __sincosf_ph_any8_128 {
    union {
        float8_128 f;
        int8_128 i;
    };
};

inline struct __sincosf_ph_dfloat __sincosf_ph_spilt2(float8_128 a) {
    struct __sincosf_ph_any8_128 hi, lo;
    hi.f = a;
    lo.f = a;
    hi.i = v_u32_and(hi.i, v_u32_move_i(0xffff0000));
    lo.f = v_f32_sub_b(a, hi.f);
    struct __sincosf_ph_dfloat res;
    res.hi = hi.f;
    res.lo = lo.f;
    return res;
}

// Veltkamp-Dekker product for accurate multiply add
inline float8_128 __sincosf_ph_soft_fmaf(float8_128 a, float8_128 b, float8_128 c) {
    struct __sincosf_ph_dfloat as = __sincosf_ph_spilt2(a);
    struct __sincosf_ph_dfloat bs = __sincosf_ph_spilt2(b);
    float8_128 hihi = v_f32_mul_b(as.hi, bs.hi);
    float8_128 hiz = v_f32_add_b(hihi, c);
    float8_128 hilo = v_f32_mul_b(as.hi, bs.lo);
    float8_128 lohi = v_f32_mul_b(as.lo, bs.hi);
    float8_128 lolo = v_f32_mul_b(as.lo, bs.lo);
    float8_128 resa = v_f32_add_b(hiz, hilo);
    float8_128 resb = v_f32_add_b(resa, lohi);
    float8_128 resc = v_f32_add_b(resb, lolo);
    return resc;
}

// happy vscode type hint
inline int8_128 __sincosf_ph_uor(int8_128 a, int8_128 b) { return a | b; }

// happy vscode type hint
inline int8_128 __sincosf_ph_ssub(int8_128 a, int8_128 b) { return a - b; }

/* Approximate sine on [-PI/4,+PI/4]. Maximum ulp error = 0.64721
   Returns -0.0f for an argument of -0.0f
   Polynomial approximation based on unpublished work by T. Myklebust
*/
inline float8_128 __sincosf_ph_sinf_poly(float8_128 a, float8_128 s) {
    float8_128 r = v_u32_move_f(0x1.7d3bbcp-19f);
    r = __sincosf_ph_soft_fmaf(r, s, v_u32_move_f(-0x1.a06bbap-13f));
    r = __sincosf_ph_soft_fmaf(r, s, v_u32_move_f(0x1.11119ap-07f));
    r = __sincosf_ph_soft_fmaf(r, s, v_u32_move_f(-0x1.555556p-03f));
    r = v_f32_add_b(v_f32_mul_b(r, s), v_u32_move_f(0.0f));
    r = __sincosf_ph_soft_fmaf(r, a, a);
    return r;
}

/* Approximate cosine on [-PI/4,+PI/4]. Maximum ulp error = 0.87531 */
inline float8_128 __sincosf_ph_cosf_poly(float8_128 s) {
    float8_128 r;

    r = v_u32_move_f(0x1.98e616p-16f);
    r = __sincosf_ph_soft_fmaf(r, s, v_u32_move_f(-0x1.6c06dcp-10f));
    r = __sincosf_ph_soft_fmaf(r, s, v_u32_move_f(0x1.55553cp-05f));
    r = __sincosf_ph_soft_fmaf(r, s, v_u32_move_f(-0x1.000000p-01f));
    r = __sincosf_ph_soft_fmaf(r, s, v_u32_move_f(0x1.000000p+00f));
    return r;
}

inline float8_128 __sincosf_ph_fastabsf(float8_128 a) {
    struct __sincosf_ph_any8_128 x;
    x.f = a;
    x.i = v_u32_and(x.i, v_u32_move_i(0x7fffffff));
    return x.f;
}

// get float exponent plus 1
inline int8_128 __sincosf_ph_fgetexpop1(float8_128 a) {
    struct __sincosf_ph_any8_128 x;
    x.f = a;
    int8_128 e =
        v_s32_add(v_u32_and(v_u32_shr(x.i, v_u32_move_i(23)), v_u32_move_i(0xff)), v_u32_move_i(-126));
    return e;
}

// float set exponent and abs
inline float8_128 __sincosf_ph_fabssetexpo(float8_128 a, int8_128 expo) {
    struct __sincosf_ph_any8_128 x;
    x.f = a;
    x.i = v_u32_and(x.i, v_u32_move_i(0x007fffff));
    x.i = __sincosf_ph_uor(x.i, (v_u32_shl(v_s32_add(expo, v_u32_move_i(127)), v_u32_move_i(23))));
    return x.f;
}

inline float8_128 __sincosf_ph_fastneg(float8_128 a) {
    struct __sincosf_ph_any8_128 x;
    x.f = a;
    x.i = v_u32_xor(x.i, v_u32_move_i(0x80000000));
    return x.f;
}

inline int8_128 __sincosf_ph_soft_mulu(int8_128 a, int8_128 b, int8_128 *hi) {
    int8_128 as0 = v_u32_and(a, v_u32_move_i(0x7ff));
    int8_128 bs0 = v_u32_and(b, v_u32_move_i(0x7ff));
    int8_128 as1 = v_u32_and(v_u32_shr(a, v_u32_move_i(11)), v_u32_move_i(0x7ff));
    int8_128 bs1 = v_u32_and(v_u32_shr(b, v_u32_move_i(11)), v_u32_move_i(0x7ff));
    int8_128 as2 = v_u32_shr(a, v_u32_move_i(2 * 11));
    int8_128 bs2 = v_u32_shr(b, v_u32_move_i(2 * 11));

    int8_128 c00 = as0 * bs0;
    int8_128 c01 = as0 * bs1;
    int8_128 c02 = as0 * bs2;
    int8_128 c10 = as1 * bs0;
    int8_128 c11 = as1 * bs1;
    int8_128 c12 = as1 * bs2;
    int8_128 c20 = as2 * bs0;
    int8_128 c21 = as2 * bs1;
    int8_128 c22 = as2 * bs2;

    int8_128 c0 = c00;
    int8_128 c1 = c01 + c10;
    int8_128 c2 = c02 + c11 + c20;
    int8_128 c3 = c12 + c21;
    int8_128 c4 = c22;

    c1 += v_u32_shr(c0, 11);
    c0 &= 0x7ff;

    c2 += v_u32_shr(c1, 11);
    c1 &= 0x7ff;

    c3 += v_u32_shr(c2, 11);
    c2 &= 0x7ff;

    c4 += v_u32_shr(c3, 11);
    c3 &= 0x7ff;

    int8_128 c5 = v_u32_shr(c4, 11);
    c4 &= 0x7ff;

    int8_128 lo = c0 | (c1 << 11) | (c2 << 22);
    *hi = v_u32_shr(c2, 10) | (c3 << 1) | (c4 << 12) | (c5 << 23);

    return lo;
}

inline int8_128 __sincosf_ph_soft_fmau(int8_128 a, int8_128 b, int8_128 z, int8_128 *hi) {
    int8_128 as0 = v_u32_and(a, v_u32_move_i(0x7ff));
    int8_128 bs0 = v_u32_and(b, v_u32_move_i(0x7ff));
    int8_128 zs0 = v_u32_and(z, v_u32_move_i(0x7ff));
    int8_128 as1 = v_u32_and(v_u32_shr(a, v_u32_move_i(11)), v_u32_move_i(0x7ff));
    int8_128 bs1 = v_u32_and(v_u32_shr(b, v_u32_move_i(11)), v_u32_move_i(0x7ff));
    int8_128 zs1 = v_u32_and(v_u32_shr(z, v_u32_move_i(11)), v_u32_move_i(0x7ff));
    int8_128 as2 = v_u32_shr(a, v_u32_move_i(2 * 11));
    int8_128 bs2 = v_u32_shr(b, v_u32_move_i(2 * 11));
    int8_128 zs2 = v_u32_shr(z, v_u32_move_i(2 * 11));

    int8_128 c00 = as0 * bs0;
    int8_128 c01 = as0 * bs1;
    int8_128 c02 = as0 * bs2;
    int8_128 c10 = as1 * bs0;
    int8_128 c11 = as1 * bs1;
    int8_128 c12 = as1 * bs2;
    int8_128 c20 = as2 * bs0;
    int8_128 c21 = as2 * bs1;
    int8_128 c22 = as2 * bs2;

    int8_128 c0 = c00 + zs0;
    int8_128 c1 = c01 + c10 + zs1;
    int8_128 c2 = c02 + c11 + c20 + zs2;
    int8_128 c3 = c12 + c21;
    int8_128 c4 = c22;

    c1 += v_u32_shr(c0, 11);
    c0 &= 0x7ff;

    c2 += v_u32_shr(c1, 11);
    c1 &= 0x7ff;

    c3 += v_u32_shr(c2, 11);
    c2 &= 0x7ff;

    c4 += v_u32_shr(c3, 11);
    c3 &= 0x7ff;

    int8_128 c5 = v_u32_shr(c4, 11);
    c4 &= 0x7ff;

    int8_128 lo = c0 | (c1 << 11) | (c2 << 22);
    *hi = v_u32_shr(c2, 10) | (c3 << 1) | (c4 << 12) | (c5 << 23);

    return lo;
}

inline int8_128 __sincosf_ph_soft_uadd(int8_128 a, int8_128 b) {
    int8_128 a0 = v_u32_and(a, v_u32_move_i(0x0000ffff));
    int8_128 b0 = v_u32_and(b, v_u32_move_i(0x0000ffff));
    int8_128 a1 = v_u32_shr(a, v_u32_move_i(16));
    int8_128 b1 = v_u32_shr(b, v_u32_move_i(16));

    int8_128 c0 = v_s32_add(a0, b0);
    int8_128 c1 = v_s32_add(v_s32_add(a1, b1), v_u32_shr(c0, v_u32_move_i(16)));
    return __sincosf_ph_uor(v_u32_shl(c1, v_u32_move_i(16)), v_u32_and(c0, v_u32_move_i(0x0000ffff)));
}

inline float8_128 __sincosf_ph_trig_red_slowpath_f(float8_128 a, int8_128 *quadrant) {
    int8_128 e = __sincosf_ph_fgetexpop1(a);
    float8_128 a2 = __sincosf_ph_fabssetexpo(a, v_u32_move_i(30));
    int8_128 ia = v_u32_shl(v_cvt_ftoi(a2, v_u32_move_b(0xffffffff)), v_u32_move_i(1));

    int8_128 i = v_u32_shr(e, v_u32_move_i(5));
    e = v_u32_and(e, v_u32_move_i(31));

    const unsigned int two_over_pi_f[] = {0x00000000, 0x28be60db, 0x9391054a, 0x7f09d5f4,
                                          0x7d4d3770, 0x36d8a566, 0x4f10e410};

    int8_128 pi0 = v_u32_move_i(0);
    int8_128 pi1 = v_u32_move_i(0);
    int8_128 pi2 = v_u32_move_i(0);
    int8_128 pi3 = v_u32_move_i(0);
    bool8_128 ieq0 = v_s32_cmp(EQ, i, v_u32_move_i(0));
    bool8_128 ieq1 = v_s32_cmp(EQ, i, v_u32_move_i(1));
    bool8_128 ieq2 = v_s32_cmp(EQ, i, v_u32_move_i(2));
    bool8_128 ieq3 = v_s32_cmp(EQ, i, v_u32_move_i(3));

    pi0 = v_s32_sel(ieq0, pi0, v_u32_move_i(two_over_pi_f[0]));
    pi1 = v_s32_sel(ieq0, pi1, v_u32_move_i(two_over_pi_f[1]));
    pi2 = v_s32_sel(ieq0, pi2, v_u32_move_i(two_over_pi_f[2]));
    pi3 = v_s32_sel(ieq0, pi3, v_u32_move_i(two_over_pi_f[3]));

    pi0 = v_s32_sel(ieq1, pi0, v_u32_move_i(two_over_pi_f[1 + 0]));
    pi1 = v_s32_sel(ieq1, pi1, v_u32_move_i(two_over_pi_f[1 + 1]));
    pi2 = v_s32_sel(ieq1, pi2, v_u32_move_i(two_over_pi_f[1 + 2]));
    pi3 = v_s32_sel(ieq1, pi3, v_u32_move_i(two_over_pi_f[1 + 3]));

    pi0 = v_s32_sel(ieq2, pi0, v_u32_move_i(two_over_pi_f[2 + 0]));
    pi1 = v_s32_sel(ieq2, pi1, v_u32_move_i(two_over_pi_f[2 + 1]));
    pi2 = v_s32_sel(ieq2, pi2, v_u32_move_i(two_over_pi_f[2 + 2]));
    pi3 = v_s32_sel(ieq2, pi3, v_u32_move_i(two_over_pi_f[2 + 3]));

    pi0 = v_s32_sel(ieq3, pi0, v_u32_move_i(two_over_pi_f[3 + 0]));
    pi1 = v_s32_sel(ieq3, pi1, v_u32_move_i(two_over_pi_f[3 + 1]));
    pi2 = v_s32_sel(ieq3, pi2, v_u32_move_i(two_over_pi_f[3 + 2]));
    pi3 = v_s32_sel(ieq3, pi3, v_u32_move_i(two_over_pi_f[3 + 3]));

    int8_128 hi = __sincosf_ph_uor(v_u32_shl(pi0, e), v_u32_shr(pi1, __sincosf_ph_ssub(v_u32_move_i(32), e)));
    int8_128 mid =
        __sincosf_ph_uor(v_u32_shl(pi1, e), v_u32_shr(pi2, __sincosf_ph_ssub(v_u32_move_i(32), e)));
    int8_128 lo = __sincosf_ph_uor(v_u32_shl(pi2, e), v_u32_shr(pi3, __sincosf_ph_ssub(v_u32_move_i(32), e)));

    int8_128 lop, hip;
    _UNUSED int8_128 tmplo = __sincosf_ph_soft_mulu(ia, lo, &lop);
    lop = __sincosf_ph_soft_fmau(ia, mid, lop, &hip);
    _UNUSED int8_128 tmphi;
    int8_128 hip1 = __sincosf_ph_soft_mulu(ia, hi, &tmphi);
    hip = __sincosf_ph_soft_uadd(hip, hip1);

    int8_128 q = v_u32_shr(hip, v_u32_move_i(30));
    hip = v_u32_and(hip, v_u32_move_i(0x3fffffff));

    bool8_128 hip2 = v_s32_cmp(NEQ, v_u32_and(hip, v_u32_move_i(0x20000000)), v_u32_move_i(0));
    bool8_128 hip4 = v_s32_cmp(NEQ, v_u32_and(hip, v_u32_move_i(0x40000000)), v_u32_move_i(0));
    int8_128 hipxor4 = v_u32_xor(hip, v_u32_move_i(0x40000000));
    int8_128 hipxorc = v_u32_xor(hip, v_u32_move_i(0xC0000000));
    int8_128 hipxor = v_s32_sel(hip4, hipxorc, hipxor4);
    hip = v_s32_sel(hip2, hip, hipxor);
    q = v_s32_sel(hip2, q, v_s32_add(q, v_u32_move_i(1)));

    bool8_128 sigp = v_s32_cmp(NEQ, v_u32_and(hip, v_u32_move_i(0x80000000)), v_u32_move_i(0));
    hip = v_s32_sel(sigp, hip, v_u32_xor(hip, v_u32_move_i(0xffffffff)));
    lop = v_s32_sel(sigp, lop, v_u32_xor(v_s32_add(lop, v_u32_move_i(-1)), v_u32_move_i(0xffffffff)));
    float8_128 d = v_cvt_itof(hip);
    int8_128 llop = v_u32_and(lop, v_u32_move_i(0x0000ffff));
    int8_128 hlop = v_u32_shr(lop, v_u32_move_i(16));
    float8_128 fhlop = v_f32_mul_b(v_cvt_itof(hlop), v_u32_move_f(0x1.0p-16));
    float8_128 fllop = v_f32_mul_b(v_cvt_itof(llop), v_u32_move_f(0x1.0p-32));
    d = v_f32_add_b(d, v_f32_add_b(fhlop, fllop));
    d = v_f32_sel(sigp, d, __sincosf_ph_fastneg(d));

    d = v_f32_mul_b(d, v_u32_move_f(0x1.921fb54442d18p-30));
    bool8_128 nega = v_f32_cmp(LS, a, v_u32_move_f(0.0));
    float8_128 r = d;
    r = v_f32_sel(nega, r, __sincosf_ph_fastneg(r));
    q = v_s32_sel(nega, q, __sincosf_ph_ssub(v_u32_move_i(0), q));

    *quadrant = q;
    return r;
}

/* Like rintf(), but -0.0f -> +0.0f, and |a| must be <= 0x1.0p+22 */
inline float8_128 __sincosf_ph_quick_and_dirty_rintf(float8_128 a) {
    float8_128 cvt_magic = v_u32_move_f(0x1.800000p+23f);
    return v_f32_sub_b(v_f32_add_b(a, cvt_magic), cvt_magic);
}

/* Argument reduction for trigonometric functions that reduces the argument
   to the interval [-PI/4, +PI/4] and also returns the quadrant. It returns
   -0.0f for an input of -0.0f
*/
inline float8_128 __sincosf_ph_trig_red_f(float8_128 a, float switch_over, int8_128 *q) {
    float8_128 r1, r2;
    int8_128 q1, q2;

    bool8_128 sel = v_f32_cmp(GT, __sincosf_ph_fastabsf(a), v_u32_move_f(switch_over));
    { r1 = __sincosf_ph_trig_red_slowpath_f(a, &q1); }
    {
        float8_128 j, r;
        /* FMA-enhanced Cody-Waite style reduction. W. J. Cody and W. Waite,
           "Software Manual for the Elementary Functions", Prentice-Hall 1980
        */
        j = v_f32_mul_b(v_u32_move_f(0x1.45f306p-1f), a);
        j = __sincosf_ph_quick_and_dirty_rintf(j);
        r = __sincosf_ph_soft_fmaf(j, v_u32_move_f(-0x1.921fb0p+00f), a); // pio2_high
        r = __sincosf_ph_soft_fmaf(j, v_u32_move_f(-0x1.5110b4p-22f), r); // pio2_mid
        r = __sincosf_ph_soft_fmaf(j, v_u32_move_f(-0x1.846988p-48f), r); // pio2_low
        q2 = v_cvt_ftoi(j, v_u32_move_b(0xffffffff));
        r2 = r;
    }
    *q = v_s32_sel(sel, q2, q1);
    return v_f32_sel(sel, r2, r1);
}

/* Map sine or cosine value based on quadrant */
inline float8_128 __sincosf_ph_sinf_cosf_core(float8_128 a, int8_128 i) {
    float8_128 r, s;

    s = v_f32_mul_b(a, a);
    float8_128 cosv = __sincosf_ph_cosf_poly(s);
    float8_128 sinv = __sincosf_ph_sinf_poly(a, s);
    bool8_128 sel1 = v_s32_cmp(NEQ, v_u32_and(i, v_u32_move_i(1)), v_u32_move_i(0));
    r = v_f32_sel(sel1, sinv, cosv);
    bool8_128 sel2 = v_s32_cmp(NEQ, v_u32_and(i, v_u32_move_i(2)), v_u32_move_i(0));
    r = v_f32_sel(sel2, r, v_f32_sub_b(v_u32_move_f(0.0), r)); // don't change "sign" of NaNs
    return r;
}

inline float8_128 __dlc_sinf(float8_128 a) {
    float8_128 r;
    int8_128 i;

    a = v_f32_add_b(v_f32_mul_b(a, v_u32_move_f(0.0)), a);
    r = __sincosf_ph_trig_red_f(a, 117435.992f, &i);
    r = __sincosf_ph_sinf_cosf_core(r, i);
    return r;
}

/* maximum ulp error = 1.49510 */
inline float8_128 __dlc_cosf(float8_128 a) {
    float8_128 r;
    int8_128 i;

    a = v_f32_add_b(v_f32_mul_b(a, v_u32_move_f(0.0)), a); // inf -> NaN
    r = __sincosf_ph_trig_red_f(a, 71476.0625f, &i);
    r = __sincosf_ph_sinf_cosf_core(r, v_s32_add(i, v_u32_move_i(1)));
    return r;
}

inline void __dlc_sincosf(float8_128 a, float8_128 *sin, float8_128 *cos) {
    float8_128 r;
    int8_128 i;

    a = v_f32_add_b(v_f32_mul_b(a, v_u32_move_f(0.0)), a); // inf -> NaN
    r = __sincosf_ph_trig_red_f(a, 71476.0625f, &i);
    *sin = __sincosf_ph_sinf_cosf_core(r, i);
    *cos = __sincosf_ph_sinf_cosf_core(r, v_s32_add(i, v_u32_move_i(1)));
}

#endif
