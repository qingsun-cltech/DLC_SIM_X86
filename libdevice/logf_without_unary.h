#ifndef _LOGF_WITHOUT_UNARY_H_
#define _LOGF_WITHOUT_UNARY_H_

#include "function.h"
#include "typehint.h"
#include "frcp_rd_without_unary.h"

inline float8_128 __dlc_logf_without_unary(float8_128 a) {
  #ifndef USE_DLC_INST
    const float ln2_hi = 6.9313812256e-01; /* 0x3f317180 */
    const float ln2_lo = 9.0580006145e-06; /* 0x3717f7d1 */
    const float two25 = 3.355443200e+07;   /* 0x4c000000 */
    const float Lg1 = 6.6666668653e-01;    /* 0x3F2AAAAB */
    const float Lg2 = 4.0000000596e-01;    /* 0x3ECCCCCD */
    const float Lg3 = 2.8571429849e-01;    /* 0x3E924925 */
    const float Lg4 = 2.2222198546e-01;    /* 0x3E638E29 */
    const float Lg5 = 1.8183572590e-01;    /* 0x3E3A3325 */
    const float Lg6 = 1.5313838422e-01;    /* 0x3E1CD04F */
    const float Lg7 = 1.4798198640e-01;    /* 0x3E178897 */

    float8_128 x = a;
    int8_128 ix = *(int8_128*)(&x);
    int8_128 k = 0;

    bool8_128 is_zero = v_s32_cmp(EQ, v_u32_and(ix, 0x7fffffff), 0);
    bool8_128 is_neg = v_s32_cmp(LS, ix, 0);
    bool8_128 is_inf = v_s32_cmp(GTEQ, ix, 0x7f800000);

    k = v_s32_sel(v_s32_cmp(LS, ix, 0x00800000), k, (k - 25));
    x = v_f32_sel(v_s32_cmp(LS, ix, 0x00800000), x, (x * two25));
    ix = *(int8_128*)(&x);
    k += ((ix >> 23) - 127);
    ix = ix & 0x007fffff;
    int8_128 i = (ix + (0x95f64 << 3)) & 0x800000;
    int8_128 temp = ix | (i ^ 0x3f800000);
    x = *(float8_128*)(&temp);

    k += (i >> 23);
    float8_128 f = x - 1.0;
    float8_128 dk = v_cvt_itof(k);
    float8_128 R = 0.0;

    bool8_128 is_small_fraction = v_s32_cmp(LS, v_u32_and(0x007fffff, (15 + ix)), 16);
    bool8_128 is_zero_fraction  = v_f32_cmp(EQ, f, 0.0);
    bool8_128 is_zero_exponent  = v_s32_cmp(EQ, k, 0);

    float8_128 result_1 = v_f32_sel(is_zero_exponent , (dk * ln2_hi + dk * ln2_lo), 0.0);
    R = f * f * ((float)0.5 - (float)0.33333333333333333 * f);
    float8_128 result_2 = v_f32_sel(is_zero_exponent , (dk * ln2_hi - ((R - dk * ln2_lo) - f)), (f - R));

    float8_128 result_3 = v_f32_sel(is_zero_fraction , result_2, result_1);

    float8_128 s = v_f32_mul_b(f, __dlc_frcp_rd_without_unary(v_f32_add_b(2.0, f)));
    float8_128 z = s * s;
    i = ix - (0x6147a << 3);
    float8_128 w = z * z;
    int8_128 j = (0x6b851 << 3) - ix;
    float8_128 t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    float8_128 t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    i |= j;
    R = t2 + t1;

    float8_128 hfsq = 0.5 * f * f;
    bool8_128 is_large_diff  = v_s32_cmp(GT, i, 0);

    float8_128 result_4 = v_f32_sel(is_zero_exponent , 
                                        (dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f)), 
                                        (f - (hfsq - s * (hfsq + R))));
    float8_128 result_5 = v_f32_sel(is_zero_exponent , 
                                        (dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f)), 
                                        (f - s * (f - R)));
    float8_128 result_6 = v_f32_sel(is_large_diff , result_5, result_4);

    float8_128 result = v_f32_sel(is_small_fraction, result_6, result_3);
    result = v_f32_sel(is_zero, result, v_u32_move_b(0xff800000));
    result = v_f32_sel(is_neg, result, v_u32_move_b(0xffc00000));
    result = v_f32_sel(is_inf, result, (a * a));
    return result;
  #else
    // v_f32_log is log2
    return v_f32_log(a) * 0.6931471805599453;
  #endif
}

#endif