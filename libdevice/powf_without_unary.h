#ifndef _POWF_WITHOUT_UNARY_H_
#define _POWF_WITHOUT_UNARY_H_
#include "fabsf.h"
#include "int2float_rn.h"
#include "frcp_rd_without_unary.h"
#include "powf.h"

inline float8_128 __math_oflowf(int8_128 sign) {
    float8_128 y = v_u32_move_f(0x1p97f);
    bool8_128 sign_eq_0 = v_s32_cmp(EQ, sign, 0);
    float8_128 temp = v_f32_sel(sign_eq_0, y, -y);
    y = temp * y;
    return y;
}

inline float8_128 __math_uflowf(int8_128 sign) {
    float8_128 y = v_u32_move_f(0x1p-95f);
    bool8_128 sign_eq_0 = v_s32_cmp(EQ, sign, 0);
    float8_128 temp = v_f32_sel(sign_eq_0, y, -y);
    y = temp * y;
    return y;
}
/*减少cycle，直接传进来的就是uint
# define IEEE_754_2008_SNAN 1
issignalingf_inline (float x)
{
  uint32_t ix = asuint (x);
  if (!IEEE_754_2008_SNAN)
    return (ix & 0x7fc00000) == 0x7fc00000;
  return 2 * (ix ^ 0x00400000) > 0xFF800000u;
}
暂时就认为IEEE_754_2008_SNAN为1
*/
inline bool8_128 issignalingf_inline(int8_128 ix) {
    int8_128 x_temp = 2 * (ix ^ 0x00400000);
    bool8_128 issignalingf_inline_v = v_s32_cmp(GT, x_temp, 0xFF800000);
    return issignalingf_inline_v;
}

inline float8_128 __dlc_powf_without_unary(float8_128 x, float8_128 y) {
  #ifndef USE_DLC_INST
    float8_128 z, ax, z_h, z_l, p_h, p_l;
    float8_128 y1, t1, t2 = 0, r, s, t, u, v, w;
    int8_128 i, j = 0, k = 0, yisint, n;
    int8_128 hx, hy, ix, iy, is;
    float8_128 result = 0;
    hx = *(int8_128 *)(&x);
    hy = *(int8_128 *)(&y);
    ix = hx & 0x7fffffff;
    iy = hy & 0x7fffffff;

    /* determine if y is an odd int when x < 0
     * yisint = 0	... y is not an integer
     * yisint = 1	... y is an odd int
     * yisint = 2	... y is an even int
     */

    yisint = 0;
    bool8_128 hx_ls_0 = v_s32_cmp(LS, hx, 0x0);
    bool8_128 iy_gteq_4b800000 = v_s32_cmp(GTEQ, iy, 0x4b800000);

    yisint = v_s32_sel(iy_gteq_4b800000, yisint, 2);
    bool8_128 iy_gteq_3f800000 = v_s32_cmp(GTEQ, iy, 0x3f800000);

    int8_128 k_ = v_s32_sel(hx_ls_0 & iy_gteq_3f800000, 0, v_u32_shr(iy, 23) - 127);
    int8_128 j_ = v_s32_sel(hx_ls_0 & iy_gteq_3f800000, 0, v_u32_shr(iy, 23 - k_));
    bool8_128 mask7 = v_s32_cmp(EQ, v_u32_shl(j_, 23 - k_), iy);
    bool8_128 mask4_6_7 = (hx_ls_0 & iy_gteq_3f800000) & mask7;
    yisint = v_s32_sel(mask4_6_7, yisint, 2 - (j_ & 1));

    /*这里在判断哪些寄存器会在上面用过的基础上给到之后的代码去使用*/
    k = v_s32_sel((hx_ls_0 & iy_gteq_3f800000), k, k_);
    j = v_s32_sel((hx_ls_0 & iy_gteq_3f800000), j, j_);

    /* |y| is huge */
    bool8_128 iy_gt_huge = v_s32_cmp(GT, iy, 0x4d000000);
    bool8_128 iy_lseq_huge = v_s32_cmp(LSEQ, iy, 0x4d000000);

    ax = __dlc_fabsf(x); /*这个ax在后面还会用到，所以不能改变它的值*/

    /* over/underflow if x is not close to one */
    bool8_128 ix_ls_3f7ffff4 = v_s32_cmp(LS, ix, 0x3f7ffff4);
    bool8_128 ix_gt_3f800007 = v_s32_cmp(GT, ix, 0x3f800007);
    bool8_128 hy_gt_0 = v_s32_cmp(GT, hy, 0);
    /*这里的输出判断放到最后去*/

    /* now |1-x| is tiny <= 2**-20, suffice to compute
    log(x) by x-x^2/2+x^3/3-x^4/4 */
    /* t has 20 trailing zeros */
    t = ax - 1; /*t，w, u, v, t1, is在后面都是先赋值再使用*/
    w = (t * t) * (0.5 - t * (0.333333333333 - t * 0.25));
    u = 1.4426879883e+00 * t;
    v = t * 7.0526075433e-06 - w * 1.4426950216e+00;
    t1 = u + v;
    is = *(int8_128 *)(&t1);
    int8_128 is_temp = v_u32_and(is, v_u32_move_i(0xfffff000));
    t1 = *(float8_128 *)(&is_temp);
    float8_128 t2_ = v - (t1 - u);

    float8_128 s2, s_h, s_l, t_h, t_l;
    n = 0; /*n后面不会用到*/
    /* take care subnormal number */
    bool8_128 FLT_UWORD_IS_SUBNORMAL = v_s32_cmp(LS, ix, 0x00800000);
    float8_128 ax_ = ax * 16777216.0;
    int8_128 n_ = n - 24;
    int8_128 ix_ = *(int8_128 *)(&ax);

    ax_ = v_f32_sel(FLT_UWORD_IS_SUBNORMAL & iy_lseq_huge, ax, ax_);
    n_ = v_s32_sel(FLT_UWORD_IS_SUBNORMAL & iy_lseq_huge, n, n_);
    ix_ = v_s32_sel(FLT_UWORD_IS_SUBNORMAL & iy_lseq_huge, ix, ix_);

    n_ = n_ + (v_u32_shr(ix, 23) - 127);
    j = ix & 0x007fffff; /*j后面会重新赋值*/
    /* determine interval */
    ix_ = j | 0x3f800000; /* normalize ix */
    /* |x|<sqrt(3/2) */
    bool8_128 j_lseq_1cc471 = v_s32_cmp(LSEQ, j, 0x1cc471);
    k_ = v_s32_sel(j_lseq_1cc471, k, 0);
    /* |x|<sqrt(3)   */
    bool8_128 j_ls_5db3d7 = v_s32_cmp(LS, j, 0x5db3d7);
    k_ = v_s32_sel(j_ls_5db3d7, k_, 1);
    bool8_128 j_gteq_5db3d7 = v_s32_cmp(GTEQ, j, 0x5db3d7);

    k_ = v_s32_sel(j_gteq_5db3d7, k_, 0);
    n_ = v_s32_sel(j_gteq_5db3d7, n_, n_ + 1);
    ix_ = v_s32_sel(j_gteq_5db3d7, ix_, ix_ - 0x00800000);

    ax_ = *(float8_128 *)(&ix_);

    /* compute s = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
    float8_128 bp = v_f32_sel(v_s32_cmp(EQ, k_, 0), 1.5, 1.0);
    float8_128 _u_ = ax_ - bp;
    float8_128 _v_ = __dlc_frcp_rd_without_unary(ax_ + bp);
    float8_128 s_ = _u_ * _v_;
    s_h = s_;
    int8_128 _is_ = *(int8_128 *)(&s_h);
    int8_128 _is_temp_ = v_u32_and(_is_, v_u32_move_i(0xfffff000));
    s_h = *(float8_128 *)(&_is_temp_);

    /* t_h=ax+bp[k] High */
    int8_128 ix_temp = ((v_u32_shr(ix_, 1) | 0x20000000) + 0x0040000) + v_u32_shl(k_, 21);

    t_h = *(float8_128 *)(&ix_temp);

    t_l = ax_ - (t_h - bp);
    s_l = _v_ * ((_u_ - s_h * t_h) - s_h * t_l);
    /* compute log(ax) */
    s2 = s_ * s_;
    float8_128 r_ = s2 * s2 *
                    (6.0000002384e-01 +
                     s2 * (4.2857143283e-01 +
                           s2 * (3.3333334327e-01 +
                                 s2 * (2.7272811532e-01 + s2 * (2.3066075146e-01 + s2 * 2.0697501302e-01)))));
    r_ += s_l * (s_h + s_);
    s2 = s_h * s_h;
    t_h = 3.0 + s2 + r_;
    _is_ = *(int8_128 *)(&t_h);
    _is_temp_ = v_u32_and(_is_, v_u32_move_i(0xfffff000));
    t_h = *(float8_128 *)(&_is_temp_);
    t_l = r_ - ((t_h - 3.0) - s2);
    /* u+v = s*(1+...) */
    _u_ = s_h * t_h;
    _v_ = s_l * t_h + t_l * s_;
    /* 2/(3log2)*(s+...) */
    float8_128 p_h_ = _u_ + _v_;
    _is_ = *(int8_128 *)(&p_h_);
    // _is_temp_ = v_u32_and(_is_, v_u32_move_i(0xfffff000));
    _is_temp_ = _is_ & 0xfffff000;
    p_h_ = *(float8_128 *)(&_is_temp_);
    float8_128 p_l_ = _v_ - (p_h_ - _u_);
    /* cp_h+cp_l = 2/(3*log2) */
    z_h = 9.6179199219e-01 * p_h_;
    float8_128 dp_l = v_f32_sel(v_s32_cmp(EQ, k_, 0), 1.56322085e-06, 0.0);
    z_l = 4.7017383622e-06 * p_h_ + p_l_ * 9.6179670095e-01 + dp_l;

    /* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
    float8_128 _t_ = __dlc_int2float_rn(n_);
    float8_128 dp_h = v_f32_sel(v_s32_cmp(EQ, k_, 0), 5.84960938e-01, 0.0);
    float8_128 _t1_ = (((z_h + z_l) + dp_h) + _t_);
    _is_ = *(int8_128 *)(&_t1_);
    _is_temp_ = v_u32_and(_is_, v_u32_move_i(0xfffff000));
    _t1_ = *(float8_128 *)(&_is_temp_);
    float8_128 _t2_ = z_l - (((_t1_ - _t_) - dp_h) - z_h);

    /*这里在判断哪些寄存器会在上面用过的基础上给到之后的代码去使用*/
    t1 = v_f32_sel(iy_lseq_huge, t1, _t1_);
    t1 = v_f32_sel(iy_gt_huge, t1, _t1_);

    t2 = v_f32_sel(iy_lseq_huge, t2, _t2_);
    t2 = v_f32_sel(iy_gt_huge, t2, t2_);
    /* s (sign of result -ve**odd) = -1 else = 1 */
    s = 1;
    /* (-ve)**(odd int) */
    // bool8_128 hx_shr_31_uint_sub_1_eq_0 = v_s32_cmp(EQ, v_u32_shr(hx, 31) & 0x7fffffff - 1, 0);
    int8_128 hx_shr_31_uint_sub_1_or_yisint_sub_1 = (v_u32_shr(hx, 31) & 0x7fffffff - 1) | (yisint - 1);
    bool8_128 mask_ = v_s32_cmp(EQ, hx_shr_31_uint_sub_1_or_yisint_sub_1, 0);
    s = v_f32_sel(mask_, s, -1);

    /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
    is = *(int8_128 *)(&y);
    _is_temp_ = v_u32_and(is, v_u32_move_i(0xfffff000));
    y1 = *(float8_128 *)(&_is_temp_);
    p_l = (y - y1) * t1 + y * t2;
    p_h = y1 * t1;
    z = p_l + p_h;
    j = *(int8_128 *)(&z);
    i = j & 0x7fffffff;

    /*compute 2**(p_h+p_l)*/
    k = v_u32_shr(i, 23) - 0x7f;
    n = 0;
    // float8_128 i_f = *(float8_128 *)(&i);
    /* if |z| > 0.5, set n = [z+0.5] */
    bool8_128 i_gt_one_half = v_s32_cmp(GT, i, 0x3f000000);
    n_ = j + v_u32_shr(0x00800000, (k + 1));
    /* new k for n */
    k = (v_u32_shr(n_ & 0x7fffffff, 23)) - 0x7f;
    int8_128 n_temp = n_ & (~v_u32_shr(0x007fffff, k));

    float8_128 t_ = *(float8_128 *)(&n_temp);
    n_ = v_u32_shr((n_ & 0x007fffff) | 0x00800000, 23 - k);
    bool8_128 j_ls_0 = v_s32_cmp(LS, j, 0);
    n_ = v_s32_sel(j_ls_0, n_, -n_);
    p_h_ = p_h - t_;
    /*这里在判断哪些寄存器会在上面用过的基础上给到之后的代码去使用*/
    p_h = v_f32_sel(i_gt_one_half, p_h, p_h_);
    n = v_s32_sel(i_gt_one_half, n, n_);
    t = v_f32_sel(i_gt_one_half, t, t_);
    t = p_l + p_h;

    is = *(int8_128 *)(&t);
    _is_temp_ = v_u32_and(is, v_u32_move_i(0xfffff000));
    t = *(float8_128 *)(&_is_temp_);
    u = t * 6.93145752e-01;
    v = (p_l - (t - p_h)) * 6.9314718246e-01 + t * 1.42860654e-06;
    z = u + v;
    w = v - (z - u);
    t = z * z;
    t1 = z - t * (1.6666667163e-01 +
                  t * (-2.7777778450e-03 +
                       t * (6.6137559770e-05 + t * (-1.6533901999e-06 + t * 4.1381369442e-08))));
    r = (z * t1) * __dlc_frcp_rd_without_unary(t1 - 2)- (w + z * w);
    z = 1 - (r - z);
    j = *(int8_128 *)(&z);
    j += (n << 23);

    bool8_128 j_shr_23_lseq_0 = v_s32_cmp(LSEQ, v_u32_shr(j, 23), 0);
    /*不做处理*/
    // if((j>>23)<=0) z = scalbnf(z,(int)n);	/* subnormal output */
    z = v_f32_sel(j_shr_23_lseq_0, *(float8_128 *)(&j), z);
    result = s * z;
    /* |y| is huge */
    bool8_128 hy_ls_0 = v_s32_cmp(LS, hy, 0);
    result = v_f32_sel((iy_gt_huge & ix_ls_3f7ffff4) & hy_ls_0, result, __math_oflowf(0));
    result = v_f32_sel((iy_gt_huge & ix_ls_3f7ffff4) & hy_gt_0, result, __math_uflowf(0));
    result = v_f32_sel((iy_gt_huge & ix_gt_3f800007) & hy_ls_0, result, __math_oflowf(0));
    result = v_f32_sel((iy_gt_huge & ix_gt_3f800007) & hy_gt_0, result, __math_uflowf(0));
    /* overflow */
    bool8_128 j_gt_0 = v_s32_cmp(GT, j, 0);
    bool8_128 j_lseq_0 = v_s32_cmp(LSEQ, j, 0);
    bool8_128 i_gt_43000000 = v_s32_cmp(GT, i, 0x43000000);
    bool8_128 i_eq_43000000 = v_s32_cmp(EQ, i, 0x43000000);
    bool8_128 i_gt_43160000 = v_s32_cmp(GT, i, 0x43160000);
    bool8_128 i_eq_43160000 = v_s32_cmp(EQ, i, 0x43160000);

    int8_128 s_temp = v_s32_sel(v_f32_cmp(LS, s, 0), 0, 1);
    result = v_f32_sel(j_gt_0 & i_gt_43000000, result, __math_oflowf(s_temp));

    bool8_128 mask12 = v_f32_cmp(GT, p_l + 4.2995665694e-08, z - p_h);
    result = v_f32_sel((j_gt_0 & i_eq_43000000) & mask12, result, __math_oflowf(s_temp));

    result = v_f32_sel(j_lseq_0 & i_gt_43160000, result, __math_uflowf(s_temp));

    bool8_128 mask13 = v_f32_cmp(GT, p_l, z - p_h);
    result = v_f32_sel((j_lseq_0 & i_eq_43160000) & mask13, result, __math_uflowf(s_temp));
    /* (x<0)**(non-int) is NaN */
    int8_128 hx_shr_31_uint_sub_1 = (v_u32_shr(hx, 31) & 0x7fffffff )- 1;
    int8_128 hx_shr_31_uint_sub_1_or_yisint = hx_shr_31_uint_sub_1 | yisint;
    bool8_128 mask14 = v_s32_cmp(EQ, hx_shr_31_uint_sub_1_or_yisint, 0);

    result = v_f32_sel(mask14, result, (x - x) * __dlc_frcp_rd_without_unary(x - x));

    /* special value of x */
    bool8_128 ix_is_inf = v_s32_cmp(EQ, ix, 0x7f800000);
    bool8_128 ix_eq_0 = v_s32_cmp(EQ, ix, 0);

    /* +-1**+-inf = 1 */
    bool8_128 ix_eq_1 = v_s32_cmp(EQ, ix, 0x3f800000);
    bool8_128 x_special = (ix_is_inf | ix_eq_0) | ix_eq_1;
    /*x is +-0,+-inf,+-1*/
    z = ax;
    /* z = (1/|x|) */
    z = v_f32_sel(hy_ls_0 & x_special, z, __dlc_frcp_rd_without_unary(z));
    /* (-1)**non-int is NaN */
    int8_128 ix_sub_1 = ix - 0x3f800000;
    int8_128 ix_sub_1_or_yisint = ix_sub_1 | yisint;
    bool8_128 ix_sub_1_or_yisint_eq_0 = v_s32_cmp(EQ, ix_sub_1_or_yisint, 0);
    // bool8_128 ix_sub_1_eq_0 = v_s32_cmp(EQ, ix - 0x3f800000, 0);
    // bool8_128 yisint_eq_0 = v_s32_cmp(EQ, yisint, 0);
    z = v_f32_sel((ix_sub_1_or_yisint_eq_0 & hx_ls_0) & x_special, z, (z - z) *__dlc_frcp_rd_without_unary(z - z));
    /* (x<0)**odd = -(|x|**odd) */
    bool8_128 yisint_eq_1 = v_s32_cmp(EQ, yisint, 1);
    z = v_f32_sel(( yisint_eq_1 & hx_ls_0) & x_special, z, -z);

    result = v_f32_sel(x_special, result, z);
    /* y is  0.5 */
    bool8_128 hy_eq_one_half = v_s32_cmp(EQ, hy, 0x3f000000);
    bool8_128 hx_gteq_0 = v_s32_cmp(GTEQ, hx, 0);

    float8_128 x_sqrt = v_f32_sqrt(x);
    result = v_f32_sel(hx_gteq_0 & hy_eq_one_half, result, x_sqrt);
    /* y is  2 */
    bool8_128 hy_eq_2 = v_s32_cmp(EQ, hy, 0x40000000);
    result = v_f32_sel(hy_eq_2, result, x * x);
    /* y is  +-1 */
    bool8_128 iy_eq_1 = v_s32_cmp(EQ, iy, 0x3f800000);
    bool8_128 hy_ls_0_iy_eq_1 = hy_ls_0 & iy_eq_1;
    result = v_f32_sel(hy_ls_0_iy_eq_1, result, __dlc_frcp_rd_without_unary(x));
    bool8_128 hy_gteq_0 = v_s32_cmp(GTEQ, hy, 0);
    result = v_f32_sel(hy_gteq_0 & iy_eq_1, result, x);

    /* special value of y */
    /* y is +-inf */
    bool8_128 iy_isinf = v_s32_cmp(EQ, iy, 0x7f800000);
    /* +-1**+-inf = 1 */
    /* (|x|>1)**+-inf = inf,0 */
    bool8_128 ix_gt_1 = v_s32_cmp(GT, ix, 0x3f800000);
    /* (|x|<1)**-,+inf = inf,0 */
    bool8_128 ix_ls_1 = v_s32_cmp(LS, ix, 0x3f800000);

    bool8_128 mask8_9 = iy_isinf & ix_eq_1;
    result = v_f32_sel(mask8_9, result, 1);

    bool8_128 mask8_10 = iy_isinf & ix_gt_1;
    bool8_128 mask8_11 = iy_isinf & ix_ls_1;

    result = v_f32_sel(hy_gteq_0 & mask8_10, result, y);
    result = v_f32_sel(hy_ls_0 & mask8_10, result, 0);
    result = v_f32_sel(hy_ls_0 & mask8_11, result, -y);
    result = v_f32_sel(hy_gteq_0 & mask8_11, result, 0);

    /* x|y==NaN return NaN unless x==1 then return 1 */
    bool8_128 iy_is_nan = v_s32_cmp(GT, iy, v_u32_move_i(0x7f800000));
    bool8_128 ix_is_nan = v_s32_cmp(GT, ix, v_u32_move_i(0x7f800000));

    result = v_f32_sel(iy_is_nan, result, x + y); // x为nan的情况在上面处理掉，暂时不区分snan和qnan
    result = v_f32_sel(ix_is_nan, result, x + y);
    bool8_128 issignalingf_inline_y = issignalingf_inline(iy);

    bool8_128 hx_eq_1 = v_s32_cmp(EQ, hx, 0x3f800000);
    result = v_f32_sel(hx_eq_1 & (!issignalingf_inline_y), result, 1);
    /* y==zero: x**0 = 1 */

    bool8_128 issignalingf_inline_x = issignalingf_inline(ix);
    bool8_128 iy_eq_0 = v_s32_cmp(EQ, iy, 0x0);
    result = v_f32_sel(iy_eq_0, result, 1);

    result = v_f32_sel(issignalingf_inline_x & iy_eq_0, result, x + y);
    return result;
  #else
    return __dlc_powf(x, y);
  #endif
}

#endif // _POWF_WITHOUT_UNARY_H_
