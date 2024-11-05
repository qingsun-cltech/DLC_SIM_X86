#ifndef _EXPF_H_
#define _EXPF_H_
#include "frcp_rd_without_unary.h"
inline float8_128 __dlc_expf(float8_128  x_reg){
  #ifndef USE_DLC_INST
    const int ln2HI[2] = {0x3f317180, 0xbf317180};
    const int ln2LO[2] = {0x3717f7d1, 0xb717f7d1};
    const int invln2 = 0x3fb8aa3b;
    const int P1 = 0x3e2aaaab;
    const int P2 = 0xbb360b61;
    const int P3 = 0x388ab355;
    const int P4 = 0xb5ddea0e;
    const int P5 = 0x3331bb4c;

    //prepare
    int8_128 x_as_int_reg = *(int8_128*)(&x_reg);
    int8_128 hx_reg = x_as_int_reg & 0x7fffffff;
    float8_128 hi_reg = 0, lo_reg = 0;
    int8_128 xsb_reg = x_as_int_reg & 0x80000000;
    int8_128 k_reg = 0;
    /* argument reduction */ 
    bool8_128 cmp_pos_neg = v_s32_cmp(EQ, xsb_reg, v_u32_move_i(0x80000000));
    float8_128 hi_reg1 = v_f32_sel(cmp_pos_neg, v_u32_move_b(ln2HI[0]), v_u32_move_b(ln2HI[1]));
    float8_128 lo_reg1 = v_f32_sel(cmp_pos_neg, v_u32_move_b(ln2LO[0]), v_u32_move_b(ln2LO[1]));
    hi_reg1 = x_reg - hi_reg1;
    int8_128 k_reg1 = v_s32_sel(cmp_pos_neg, 1, -1);
    float8_128 k_reg2 = v_f32_sel(cmp_pos_neg, 0.5, -0.5);
    float8_128 t_reg1 = x_reg * v_u32_move_b(invln2);
    t_reg1 = t_reg1 + k_reg2;
    int8_128 k_reg2_as_int = v_cvt_ftoi(t_reg1, v_u32_move_b(0xffffffff));
    t_reg1 = v_cvt_itof(k_reg2_as_int);
    float8_128 hi_reg2 =  t_reg1 * v_u32_move_b(ln2HI[0]);
    float8_128 lo_reg2 =  t_reg1 * v_u32_move_b(ln2LO[0]);
    hi_reg2 = x_reg - hi_reg2;
    bool8_128 gt_05ln2 = v_s32_cmp(GT, hx_reg, v_u32_move_i(0x3eb17218)); 
    bool8_128 ls_15ln2 = v_s32_cmp(LS, hx_reg, v_u32_move_i(0x3F851592));
    hi_reg1 = v_f32_sel(ls_15ln2, hi_reg2, hi_reg1);
    lo_reg1 = v_f32_sel(ls_15ln2, lo_reg2, lo_reg1);
    k_reg1 = v_s32_sel(ls_15ln2, k_reg2_as_int, k_reg1);
    hi_reg = v_f32_sel(gt_05ln2, hi_reg, hi_reg1);
    lo_reg = v_f32_sel(gt_05ln2, lo_reg, lo_reg1);
    k_reg = v_s32_sel(gt_05ln2, k_reg, k_reg1);
    float8_128 x_reg1 = hi_reg - lo_reg;
    x_reg1 = v_f32_sel(gt_05ln2, x_reg, x_reg1);

  /* x is now in primary range VR13 = y*/
    float8_128 t_reg = x_reg1 * x_reg1;
    float8_128 c_tmp_reg = t_reg * v_u32_move_b(P5) +  v_u32_move_b(P4);
    c_tmp_reg = t_reg * c_tmp_reg +  v_u32_move_b(P3);
    c_tmp_reg = t_reg * c_tmp_reg +  v_u32_move_b(P2);
    c_tmp_reg = t_reg * c_tmp_reg +  v_u32_move_b(P1);
    float8_128 c_reg = x_reg1 - c_tmp_reg * c_tmp_reg;
    c_reg = x_reg1 - t_reg * c_tmp_reg;

    float8_128 xc_reg = x_reg1 * c_reg;
    float8_128 c2_reg = __dlc_frcp_rd_without_unary(c_reg - 2.0);
    float8_128 tmp_reg = xc_reg * c2_reg;
    //k != 0
    bool8_128 k_eq0 = v_s32_cmp(EQ, k_reg, 0);
    float8_128 k_Neg0_reg = 1.0 - (lo_reg + tmp_reg - hi_reg);
    float8_128 y_reg = v_f32_sel(k_eq0, k_Neg0_reg, 0);
    //k >= -125
    bool8_128 k_ge_ne125 = v_s32_cmp(GT, k_reg, -125);
    int8_128 kl23_reg = k_reg << 23;
    int8_128 y_kGeN125_reg = *(int8_128*)(&y_reg) + kl23_reg;
    int8_128 k100_reg = ((k_reg + 100) << 23) + *(int8_128*)(&y_reg);
    float8_128 k100_reg_f32 = *(float8_128*)(&k100_reg);
    k100_reg_f32 = k100_reg_f32 * v_u32_move_b(0x0d800000);
    y_reg = v_f32_sel(k_ge_ne125, k100_reg_f32, *(float8_128*)(&y_kGeN125_reg));
    // k  == 0 select_vmask0  y
    float8_128 k0_reg = 1.0 - (tmp_reg - x_reg);
    y_reg = v_f32_sel(k_eq0, y_reg, k0_reg);
    //hx < 0x34000000
    bool8_128 hx_ls = v_s32_cmp(LS, hx_reg, v_u32_move_i(0x34000000));
    float8_128 xAddone_reg = x_reg + 1.0;
    y_reg = v_f32_sel(hx_ls, y_reg, xAddone_reg);
    /* filter out non-finite argument */
    bool8_128 hx_FLT_UWORD_LOG_MAX = v_s32_cmp(GT, x_as_int_reg, v_u32_move_i(0x42b17217));
    bool8_128 hx_LS_ZERO = v_s32_cmp(LS, x_as_int_reg, v_u32_move_i(0));
    float8_128 underflow_reg = v_f32_sel(hx_LS_ZERO, y_reg, 0);
    bool8_128 hx_FLT_UWORD_LOG_MIN = v_s32_cmp(GT, hx_reg, v_u32_move_i(0x42cff1b5));
    y_reg = v_f32_sel(hx_FLT_UWORD_LOG_MIN, y_reg, underflow_reg);
    bool8_128 X_IS_INF_OR_NAN = v_f32_infnan(x_reg);
    y_reg =  v_f32_sel(hx_FLT_UWORD_LOG_MAX, y_reg, v_u32_move_b(0x7f800000));
    bool8_128 X_EQ_NEG_INF = v_s32_cmp(EQ, x_as_int_reg, v_u32_move_i(0xff800000));
    int8_128 non_inf_reg = 0;
    non_inf_reg = v_s32_sel(X_IS_INF_OR_NAN, non_inf_reg, v_u32_move_i(0x7fc00000));
    int8_128 nan_sign_reg = x_as_int_reg & 0x80000000;
    nan_sign_reg = nan_sign_reg | non_inf_reg;
    bool8_128 X_EQ_POS_INF = v_s32_cmp(EQ, x_as_int_reg, v_u32_move_i(0x7f800000));
    y_reg = v_f32_sel(X_IS_INF_OR_NAN, y_reg, *(float8_128*)(&nan_sign_reg));
    y_reg = v_f32_sel(X_EQ_NEG_INF, y_reg, 0);
    y_reg = v_f32_sel(X_EQ_POS_INF, y_reg, v_u32_move_b(0x7f800000));
    return y_reg;
  #else
    return v_f32_exp(x_reg);
  #endif 
}
#endif // _EXPF_H_
