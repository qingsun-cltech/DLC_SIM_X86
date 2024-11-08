#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #include "typehint.h"
#include "libdevice.h"

inline int8_128 _S(int a) { return v_u32_move_i(a); }

inline float8_128 _F(float a) { return v_u32_move_f(a); }

inline float8_128 $F(int8_128 a) {
    float8_128 result0;
    asm volatile("{V0@(pr0) %[res] = mov.u32 %[input];}" : [res] "=x"(result0) : [input] "x"(a) :);
    return result0;
}

inline int8_128 $S(float8_128 a) {
    int8_128 result0;
    asm volatile("{V0@(pr0) %[res] = mov.u32 %[input];}" : [res] "=x"(result0) : [input] "x"(a) :);
    return result0;
}

inline float8_128 permutef(float8_128 d, int8_128 idx) {
    float8_128 res = _F(0.0);
    asm volatile(
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 %[index], r32; V1@(pr0) vr1 = vcoreid; }"
        "{ pseudo@0 @pseudo imm_0 = 65408; V0@(pr0) vr1 = and.u32 vr1, r40; V1@(pr0) vr4 = mov.u32 %[data]; }"
        "{ pseudo@0 @pseudo imm_0 = 128; V0@(pr0) vr2 = add.s32 vr1, r32; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        "{ V0@(pr0) vmsk7 = gteq.s32 vr0, vr1; V0@(pr0) vmsk6 = ls.s32 vr0, vr2; MTI@(pr0) pcr<0> = "
        "setperm.sub vr0; }"
        "{ pseudo@0 @pseudo imm_0 = 896; V0@(pr0) vr0 = add.s32 vr0, r32; MTI@(pr0) trf<0> = pmt vr4; "
        "MISC@(pr0) vmsk7 = vmsk.and vmsk6; }"
        "{ pseudo@0 @pseudo imm_0 = 1023; V0@(pr0) vr0 = and.u32 vr0, r32; V1@(pr0) vr4 = rotate vr4, 0; "
        "MTR@(pr0) vr3 = pop trf<0>; }"
        "{ V0@(pr0) %[result] = sel vmsk7 %[result], vr3; }"
        : [result] "=x"(res)
        : [data] "x"(d), [index] "x"(idx)
        : "vr0", "vr1", "vr2", "vr3", "vr4", "vmsk6", "vmsk7");
    return res;
}

inline float8_128 and_wheref(bool8_128 c1, bool8_128 c2, float8_128 t, float8_128 f) {
    return v_f32_sel(c1, f, v_f32_sel(c2, f, t));
}

inline float8_128 vmem_permutef(SIM_X86::tensor d, int len, int8_128 idx) {
    float8_128 res = _F(0.0);

    for (int i = 0; i < len; i += 1024) {
        float8_128 rd = v_f32_ld_tnsr_b(i / 32, d);
        bool8_128 resm1 = v_s32_cmp(LS, idx, _S(1024));
        bool8_128 resm2 = v_s32_cmp(GTEQ, idx, _S(0));
        res = and_wheref(resm1, resm2, permutef(rd, idx), res);
        idx = v_s32_add(idx, _S(-1024));
    }

    return res;
}

inline float8_128 fabs(float8_128 x) { return v_f32_abs(x); }

inline float8_128 f32_div(float8_128 a, float8_128 b) { return v_f32_mul_b(a, v_f32_rcp_b(b)); }

inline float8_128 noopt(float8_128 x) {
    float8_128 r;
    asm volatile("{ V0@(pr0) %[r] = mov.u32 %[x]; }" : [r] "=x"(r) : [x] "x"(x) :);
    return r;
}

inline float8_128 ratevl(float8_128 x, SIM_X86::tensor num, int M, SIM_X86::tensor denom, int N) {
    // evaluating rational function, i.e., the ratio of two polynomials
    // the coefficients for numerator are given by `num` while coeffs for
    // denumerator are given by `denom`

    float8_128 absx = fabs(x);

    bool8_128 c1 = v_f32_cmp(GT, absx, v_u32_move_f(1.0f));
    int8_128 dir = v_s32_sel(c1, v_u32_move_i(1), v_u32_move_i(-1));
    int8_128 p = v_s32_sel(c1, v_u32_move_i(0), v_u32_move_i(M));
    float8_128 y = v_f32_sel(c1, x, v_f32_rcp_b(x));

    /* Evaluate the numerator */
    float8_128 num_ans = vmem_permutef(num, M, p);
    p = v_s32_add(p, dir);

    for (int i = 0; i < M; i++) {
        float8_128 nump = vmem_permutef(num, M, p);
        nump = noopt(nump);
        num_ans = v_f32_add_b(v_f32_mul_b(num_ans, y), nump);
        p = v_s32_add(p, dir);
    }

    /* Evaluate the denominator */
    p = v_s32_sel(c1, _S(0), _S(N));
    float8_128 denom_ans = vmem_permutef(denom, N, p);

    p = v_s32_add(p, dir);
    for (int i = 0; i < N; i++) {
        float8_128 denomp = vmem_permutef(denom, N, p);
        denomp = noopt(denomp);
        denom_ans = v_f32_add_b(v_f32_mul_b(denom_ans, y), denomp);
        p = v_s32_add(p, dir);
    }

    float8_128 ret = v_f32_mul_b(num_ans, v_f32_rcp_b(denom_ans));
    // N - M always 0 in this program
    // ret = v_f32_sel(c1, ret, v_f32_mul_b(__dlc_powf(x, v_u32_move_f(N - M)), ret));

    return ret;
}

inline float8_128 __dlc_expm1f(float8_128 a) {
    float8_128 result0;
    asm volatile("{ V0@(pr0)       vr10 = mov.u32 %[input]; }" : : [input] "x"(a) : "vr10");
    asm volatile("{"
                 "V0@(pr0)	(urf) = exp.f32 vr10;"
                 "V1@(pr0)	vmsk0 = ls.f32 vr10, r46;"
                 "}"
                 "{"
                 "MTR@(pr0)	vr0 = pop urf;"
                 "}"
                 "{"
                 "V0@(pr0)	(urf) = rcp.f32 vr0;"
                 "}"
                 "{"
                 "MTR@(pr0)	vr1 = pop urf;"
                 "}"
                 "{"
                 "V0@(pr0)	vr10 = sel vmsk0 vr0, vr1;"
                 "V1@(pr0)	vr10 = sub.f32 vr10, r49;"
                 "}"
                 :
                 :
                 : "vr0", "vr10", "vr1", "vmsk0");

    asm volatile("{V0@(pr0)        %[res] = mov.u32 vr10;}" : [res] "=x"(result0) : : "vr10");

    return result0;
}

inline float8_128 lanczos_sum_expg_scaled(float8_128 x, SIM_X86::tensor lanczos_sum_expg_scaled_num,
                                          SIM_X86::tensor lanczos_sum_expg_scaled_denom) {
    return ratevl(x, lanczos_sum_expg_scaled_num, 12, lanczos_sum_expg_scaled_denom, 12);
}

inline bool8_128 b_and(bool8_128 a, bool8_128 b) {
    return v_s32_cmp(EQ, v_s32_sel(a, v_u32_move_i(0), v_s32_sel(b, v_u32_move_i(0), v_u32_move_i(1))),
                     v_u32_move_i(1));
}

inline float8_128 _igam_helper_fac(float8_128 a, float8_128 x, SIM_X86::tensor lanczos_sum_expg_scaled_num,
                                   SIM_X86::tensor lanczos_sum_expg_scaled_denom) {
    float8_128 ax = v_f32_sub_b(v_f32_sub_b(v_f32_mul_b(a, v_f32_log(x)), x), __dlc_lgammaf(a));
    float8_128 ret1 =
        v_f32_sel(v_f32_cmp(LS, ax, v_u32_move_f(-88.72283905206835)), v_f32_exp(ax), v_u32_move_f(0.0));

    float8_128 fac = v_f32_add_b(a, v_u32_move_f(6.024680040776729583740234375 - 0.5));
    float8_128 res =
        f32_div(v_f32_sqrt(v_f32_mul_b(fac, v_u32_move_f(1. / 2.718281828459045))),
                lanczos_sum_expg_scaled(a, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom));

    float8_128 resT =
        v_f32_mul_b(res, v_f32_mul_b(v_f32_exp(v_f32_sub_b(a, x)), __dlc_powf(f32_div(x, fac), a)));

    float8_128 num = v_f32_sub_b(v_f32_sub_b(x, a), v_u32_move_f(6.024680040776729583740234375 - 0.5));
    float8_128 numfac = f32_div(num, fac);
    float8_128 resF = v_f32_mul_b(
        res, v_f32_exp(v_f32_add_b(
                 v_f32_mul_b(a, v_f32_sub_b(__dlc_log1pf(numfac), numfac)),
                 f32_div(v_f32_mul_b(x, v_u32_move_f(0.5 - 6.024680040776729583740234375)), fac))));
    float8_128 ret2 =
        v_f32_sel(v_f32_cmp(LS, a, v_u32_move_f(200.0)) & v_f32_cmp(LS, x, v_u32_move_f(200.0)), resF, resT);

    return v_f32_sel(v_f32_cmp(GT, fabs(v_f32_sub_b(a, x)), v_f32_mul_b(v_u32_move_f(0.4), fabs(a))), ret2,
                     ret1);
}

inline float8_128 _igam_helper_series(float8_128 a, float8_128 x, SIM_X86::tensor lanczos_sum_expg_scaled_num,
                                      SIM_X86::tensor lanczos_sum_expg_scaled_denom) {
    float8_128 ax = _igam_helper_fac(a, x, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom);

    float8_128 r = a;
    float8_128 c = v_u32_move_f(1.0);
    float8_128 ans = v_u32_move_f(1.0);
    bool8_128 stop = v_s32_eq(v_u32_move_i(0), v_u32_move_i(1));

    for (int i = 0; i < 2000; i++) {
        r = v_f32_sel(stop, v_f32_add_b(r, v_u32_move_f(1.0)), r);
        c = v_f32_sel(stop, v_f32_mul_b(c, v_f32_mul_b(x, v_f32_rcp_b(r))), c);
        ans = v_f32_sel(stop, v_f32_add_b(ans, c), ans);
        stop = v_f32_cmp(LSEQ, c, v_f32_mul_b(ans, v_u32_move_f(5.9604644775390625E-8)));
    }
    float8_128 ret1 = v_f32_mul_b(ans, v_f32_mul_b(ax, v_f32_rcp_b(a)));
    return v_f32_sel(v_f32_cmp(EQ, ax, v_u32_move_f(0.0)), ret1, v_u32_move_f(0.0));
}

inline float8_128 _igamc_helper_series(float8_128 a, float8_128 x) {
    bool8_128 stop = v_s32_eq(v_u32_move_i(0), v_u32_move_i(1));
    float8_128 fac = v_u32_move_f(1.0);
    float8_128 sum = v_u32_move_f(0.0);
    for (int n = 1; n < 2000; n++) {
        fac = v_f32_sel(
            stop,
            v_f32_mul_b(fac, v_f32_mul_b(v_f32_sub_b(v_u32_move_f(0.0), x), v_f32_rcp_b(v_u32_move_f(n)))),
            fac);
        float8_128 term = v_f32_mul_b(fac, v_f32_rcp_b(v_f32_add_b(a, v_u32_move_f(n))));
        sum = v_f32_sel(stop, v_f32_add_b(sum, term), sum);
        stop = v_f32_cmp(LSEQ, fabs(term), v_f32_mul_b(v_u32_move_f(5.9604644775390625E-8), fabs(sum)));
    }
    float8_128 logx = v_f32_log(x);
    float8_128 term = v_f32_sub_b(
        v_u32_move_f(0.0),
        __dlc_expm1f(v_f32_sub_b(v_f32_mul_b(a, logx), __dlc_lgammaf(v_f32_add_b(v_u32_move_f(1.0), a)))));
    return v_f32_sub_b(term,
                       v_f32_mul_b(sum, v_f32_exp(v_f32_sub_b(v_f32_mul_b(a, logx), __dlc_lgammaf(a)))));
}

inline float8_128 _igam_helper_asymptotic_series(float8_128 a, float8_128 x, bool igam,
                                                 float *_igam_helper_asymptotic_series_d) {
    float sgn = igam == 1 ? -1.0 : 1.0;
    float8_128 lambda = v_f32_mul_b(x, v_f32_rcp_b(a));
    float8_128 sigma = v_f32_mul_b(v_f32_sub_b(x, a), v_f32_rcp_b(a));
    float8_128 tmp = v_f32_sqrt(v_f32_mul_b(_F(-2.0), v_f32_sub_b(__dlc_log1pf(sigma), sigma)));
    float8_128 eta =
        v_f32_sel(v_f32_cmp(GT, lambda, _F(1.0)),
                  v_f32_sel(v_f32_cmp(LS, lambda, _F(1.0)), _F(0.0), v_f32_sub_b(_F(0.0), tmp)), tmp);
    float8_128 res = v_f32_mul_b(
        _F(0.5), __dlc_erfcf(v_f32_mul_b(_F(sgn), v_f32_mul_b(eta, v_f32_sqrt(v_f32_mul_b(a, _F(0.5)))))));

    int8_128 maxpow = _S(0);
    float8_128 __attribute__((address_space(VMEM))) etapow[25];
    etapow[0] = _F(1.0);
    for (int i = 1; i < 25; i++) {
        etapow[i] = _F(0.0);
    }
    float8_128 afac = _F(1.0);
    float8_128 absoldterm = v_u32_move_b(0x7f800000);
    float8_128 sum = _F(0.0);
    for (int k = 0; k < 25; k++) {
        float8_128 ck = _F(_igam_helper_asymptotic_series_d[k * 25]);
        bool8_128 stop2 = v_s32_eq(_S(1), _S(0));
        bool8_128 stop0 = v_s32_eq(_S(1), _S(0));
        bool8_128 stop1 = v_s32_eq(_S(1), _S(0));
        for (int n = 1; n < 25; n++) {
            bool8_128 update = v_s32_cmp(GT, _S(n), maxpow);
            update = update & !stop2;
            etapow[n] = v_f32_sel(update, etapow[n], v_f32_mul_b(eta, etapow[n - 1]));
            maxpow = v_s32_sel(update, maxpow, v_s32_add(maxpow, _S(1)));
            float8_128 ckterm = v_f32_mul_b(etapow[n], _F(_igam_helper_asymptotic_series_d[k * 25 + n]));
            ck = v_f32_sel(stop2, v_f32_add_b(ck, ckterm), ck);
            stop2 = v_f32_cmp(LS, fabs(ckterm), v_f32_mul_b(fabs(ck), _F(5.9604644775390625E-8)));
        }
        float8_128 term = v_f32_mul_b(ck, afac);
        float8_128 absterm = fabs(term);
        stop0 = v_f32_cmp(GT, absterm, absoldterm);
        sum = v_f32_sel(stop0, v_f32_add_b(sum, term), sum);
        stop1 = v_f32_cmp(LS, absterm, v_f32_mul_b(fabs(sum), _F(5.9604644775390625E-8)));
        stop1 = stop1 | stop0;
        absoldterm = v_f32_sel(stop1, absterm, absoldterm);
        afac = v_f32_sel(stop1, v_f32_mul_b(afac, v_f32_rcp_b(a)), afac);
    }
    float8_128 tmp1 = v_f32_exp(v_f32_mul_b(_F(-0.5), v_f32_mul_b(a, v_f32_mul_b(eta, eta))));
    float8_128 tmp3 = v_f32_rcp_b(v_f32_sqrt(v_f32_mul_b(_F(6.28318530717958647692), a)));
    float8_128 tmp2 = v_f32_mul_b(tmp3, sum);
    float8_128 tmp0 = v_f32_mul_b(tmp1, tmp2);
    res = v_f32_add_b(res, v_f32_mul_b(_F(sgn), tmp0));
    return res;
}

inline float8_128 _igamc_helper_continued_fraction(float8_128 a, float8_128 x,
                                                   SIM_X86::tensor lanczos_sum_expg_scaled_num,
                                                   SIM_X86::tensor lanczos_sum_expg_scaled_denom) {
    float8_128 ax = _igam_helper_fac(a, x, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom);

    float8_128 y = v_f32_sub_b(_F(1.0), a);
    float8_128 z = v_f32_add_b(x, v_f32_add_b(y, _F(1.0)));
    float8_128 c = _F(0.0);
    float8_128 pkm2 = _F(1.0);
    float8_128 qkm2 = x;
    float8_128 pkm1 = v_f32_add_b(x, _F(1.0));
    float8_128 qkm1 = v_f32_mul_b(z, x);
    float8_128 ans = v_f32_mul_b(pkm1, v_f32_rcp_b(qkm1));

    bool8_128 stop = v_s32_eq(_S(0), _S(1));
    for (int i = 0; i < 2000; i++) {
        c = v_f32_sel(stop, v_f32_add_b(c, _F(1.0)), c);
        y = v_f32_sel(stop, v_f32_add_b(y, _F(1.0)), y);
        z = v_f32_sel(stop, v_f32_add_b(z, _F(2.0)), z);
        float8_128 yc = v_f32_mul_b(y, c);
        float8_128 pk = v_f32_sub_b(v_f32_mul_b(pkm1, z), v_f32_mul_b(pkm2, yc));
        float8_128 qk = v_f32_sub_b(v_f32_mul_b(qkm1, z), v_f32_mul_b(qkm2, yc));

        bool8_128 br1 = v_f32_cmp(NEQ, qk, _F(0.0));
        float8_128 r = f32_div(pk, qk);
        float8_128 t = v_f32_sel(br1, _F(1.0), fabs(f32_div(v_f32_sub_b(ans, r), r)));
        ans = v_f32_sel(!stop & br1, ans, r);

        pkm2 = v_f32_sel(stop, pkm1, pkm2);
        pkm1 = v_f32_sel(stop, pk, pkm1);
        qkm2 = v_f32_sel(stop, qkm1, qkm2);
        qkm1 = v_f32_sel(stop, qk, qkm1);

        bool8_128 br2 = v_f32_cmp(GT, fabs(pk), _F(16777216.));
        pkm2 = v_f32_sel(br2, pkm2, v_f32_mul_b(pkm2, _F(5.9604644775390625E-8)));
        pkm1 = v_f32_sel(br2, pkm1, v_f32_mul_b(pkm1, _F(5.9604644775390625E-8)));
        qkm2 = v_f32_sel(br2, qkm2, v_f32_mul_b(qkm2, _F(5.9604644775390625E-8)));
        qkm1 = v_f32_sel(br2, qkm1, v_f32_mul_b(qkm1, _F(5.9604644775390625E-8)));

        stop = v_f32_cmp(LSEQ, t, _F(5.9604644775390625E-8));
    }

    return v_f32_mul_b(ans, ax);
}

inline bool8_128 isinf(float8_128 a) {
    return v_s32_cmp(EQ, v_u32_and($S(a), _S(0x7fffffff)), _S(0x7f800000));
}

inline float8_128 calc_igammac(float8_128 a, float8_128 x, SIM_X86::tensor lanczos_sum_expg_scaled_num,
                               SIM_X86::tensor lanczos_sum_expg_scaled_denom,
                               float *_igam_helper_asymptotic_series_d) {
    float8_128 ret1 = v_u32_move_b(0x7fffffff);

    float8_128 ret2_1 = _F(0.0);
    float8_128 ret2_2 = v_u32_move_b(0x7fffffff);
    float8_128 ret2 = v_f32_sel(v_f32_cmp(GT, x, _F(0.0)), ret2_2, ret2_1);

    float8_128 ret3 = _F(1.0);

    float8_128 ret4_1 = v_u32_move_b(0x7fffffff);
    float8_128 ret4_2 = _F(1.0);
    float8_128 ret4 = v_f32_sel(isinf(x), ret4_2, ret4_1);

    float8_128 ret5 = _F(0.0);

    float8_128 absxma_a = f32_div(fabs(v_f32_sub_b(x, a)), a);

    float8_128 ret6_12 = _igam_helper_asymptotic_series(a, x, 0, _igam_helper_asymptotic_series_d);

    float8_128 ihsax = _igamc_helper_series(a, x);
    float8_128 ret6_3_123_1 = v_f32_sub_b(_F(1.0), _igam_helper_series(a, x, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom));
    float8_128 ret6_3_1_2 =
        _igamc_helper_continued_fraction(a, x, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom);

    float8_128 ret6_3_1 = v_f32_sel(v_f32_cmp(LS, x, a), ret6_3_1_2, ret6_3_123_1);

    float8_128 ret6_3_2 = v_f32_sel(v_f32_cmp(LS, f32_div(_F(-0.4), v_f32_log(x)), a), ihsax, ret6_3_123_1);

    float8_128 ret6_3_3 = v_f32_sel(v_f32_cmp(LS, v_f32_mul_b(x, _F(1.1)), a), ihsax, ret6_3_123_1);

    float8_128 ret6_3 = 
        v_f32_sel(v_f32_cmp(GT, x, _F(1.1)),
                  v_f32_sel(v_f32_cmp(LSEQ, x, _F(0.5)), 
                            ret6_3_3, 
                            ret6_3_2), 
                  ret6_3_1);
    
    const float SMALL = 20.0;
    const float LARGE = 200.0;
    const float SMALLRATIO = 0.3;
    const float LARGERATIO = 4.5;
    float8_128 ret6 = 
        v_f32_sel((v_f32_cmp(GT, a, _F(SMALL)) & v_f32_cmp(LS, a, _F(LARGE)) & v_f32_cmp(LS, absxma_a, _F(SMALLRATIO))),
                  v_f32_sel((v_f32_cmp(GT, a, _F(LARGE)) & v_f32_cmp(LS, absxma_a, f32_div(_F(LARGERATIO), v_f32_sqrt(a)))),
                            ret6_3, 
                            ret6_12),
                  ret6_12);

    float8_128 ret =
        v_f32_sel(v_f32_cmp(LS, x, _F(0.0)) | v_f32_cmp(LS, a, _F(0.0)),
                  v_f32_sel(v_f32_cmp(EQ, a, _F(0.0)),
                            v_f32_sel(v_f32_cmp(EQ, x, _F(0.0)),
                                      v_f32_sel(isinf(a), 
                                                v_f32_sel(isinf(x), 
                                                          ret6, 
                                                          ret5), 
                                                ret4), 
                                      ret3),
                            ret2),
                  ret1);

    return ret;
}

inline float8_128 calc_igamma(float8_128 a, float8_128 x, SIM_X86::tensor lanczos_sum_expg_scaled_num,
                              SIM_X86::tensor lanczos_sum_expg_scaled_denom, float *_igam_helper_asymptotic_series_d) {
    float8_128 ret1 = v_u32_move_b(0x7fffffff);

    float8_128 ret2_1 = _F(1.0);
    float8_128 ret2_2 = v_u32_move_b(0x7fffffff);

    float8_128 ret3 = _F(0.0);

    float8_128 ret4_1 = v_u32_move_b(0x7fffffff);
    float8_128 ret4_2 = _F(0.0);

    float8_128 ret5 = _F(1.0);

    float8_128 igas = _igam_helper_asymptotic_series(a, x, 1, _igam_helper_asymptotic_series_d);

    float8_128 ret6_3_1 =
        v_f32_sub_b(_F(1.0), calc_igammac(a, x, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom,
                                          _igam_helper_asymptotic_series_d));

    float8_128 ret6_3_2 =
        _igam_helper_series(a, x, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom);

    float8_128 absxma_a = f32_div(fabs(v_f32_sub_b(x, a)), a);
    const float SMALL = 20.0;
    const float LARGE = 200.0;
    const float SMALLRATIO = 0.3;
    const float LARGERATIO = 4.5;

    return v_f32_sel(
        (v_f32_cmp(LS, x, _F(0.0)) | v_f32_cmp(LS, a, _F(0.0))),
        v_f32_sel(
            v_f32_cmp(EQ, a, _F(0.0)),
            v_f32_sel(
                v_f32_cmp(EQ, x, _F(0.0)),
                v_f32_sel(
                    isinf(a),
                    v_f32_sel(
                        isinf(x),
                        v_f32_sel((v_f32_cmp(GT, a, _F(SMALL)) & v_f32_cmp(LS, a, _F(LARGE)) &
                                   v_f32_cmp(LS, absxma_a, _F(SMALLRATIO))),
                                  v_f32_sel((v_f32_cmp(GT, a, _F(LARGE)) &
                                             v_f32_cmp(LS, absxma_a, f32_div(_F(LARGERATIO), v_f32_sqrt(a)))),
                                            v_f32_sel((v_f32_cmp(GT, x, _F(1.0)) & v_f32_cmp(GT, x, a)),
                                                      ret6_3_2, ret6_3_1),
                                            igas),
                                  igas),
                        ret5),
                    v_f32_sel(isinf(x), ret4_2, ret4_1)),
                ret3),
            v_f32_sel(v_f32_cmp(GT, x, _F(0.0)), ret2_2, ret2_1)),
        ret1);
}
