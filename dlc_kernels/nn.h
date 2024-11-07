#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#ifndef __NN_H_X86__
#define __NN_H_X86__

#include "ldst.h"
#include "matmul_t.h"

#include "align.h"

inline float8_128 load8_k(SIM_X86::tensor t, int st, int ldmk, int w, float fill) {
    float8_128 v = load8_128_stride_ldmk(0, st, t, ldmk);
    int8_128 c = get_core_id();
    bool8_128 m = v_s32_cmp(LS, v_u32_and(c, v_u32_move_i(127)), v_u32_move_i(w));
    return v_f32_sel(m, v_u32_move_f(fill), v);
}

inline float8_128 loadmin8_k(SIM_X86::tensor t, int st, int h, int ldmk, int w, float fill) {
    float8_128 v = load8_128_stride_ldmk(0, st, t, ldmk);
    int8_128 c = get_core_id();
    bool8_128 m = v_s32_cmp(LS, v_u32_and(c, v_u32_move_i(127)), v_u32_move_i(w));
    bool8_128 m2 = v_s32_cmp(LS, v_u32_shr(c, v_u32_move_i(7)), v_u32_move_i(h));
    return v_f32_sel(m2, v_u32_move_f(fill), v_f32_sel(m, v_u32_move_f(fill), v));
}

#define SWITCH_CASES_REV(x)                                                                                  \
    if ((x) > 15) {                                                                                          \
        CASE_ITEM(15)                                                                                        \
    }                                                                                                        \
    if ((x) > 14) {                                                                                          \
        CASE_ITEM(14)                                                                                        \
    }                                                                                                        \
    if ((x) > 13) {                                                                                          \
        CASE_ITEM(13)                                                                                        \
    }                                                                                                        \
    if ((x) > 12) {                                                                                          \
        CASE_ITEM(12)                                                                                        \
    }                                                                                                        \
    if ((x) > 11) {                                                                                          \
        CASE_ITEM(11)                                                                                        \
    }                                                                                                        \
    if ((x) > 10) {                                                                                          \
        CASE_ITEM(10)                                                                                        \
    }                                                                                                        \
    if ((x) > 9) {                                                                                           \
        CASE_ITEM(9)                                                                                         \
    }                                                                                                        \
    if ((x) > 8) {                                                                                           \
        CASE_ITEM(8)                                                                                         \
    }                                                                                                        \
    if ((x) > 7) {                                                                                           \
        CASE_ITEM(7)                                                                                         \
    }                                                                                                        \
    if ((x) > 6) {                                                                                           \
        CASE_ITEM(6)                                                                                         \
    }                                                                                                        \
    if ((x) > 5) {                                                                                           \
        CASE_ITEM(5)                                                                                         \
    }                                                                                                        \
    if ((x) > 4) {                                                                                           \
        CASE_ITEM(4)                                                                                         \
    }                                                                                                        \
    if ((x) > 3) {                                                                                           \
        CASE_ITEM(3)                                                                                         \
    }                                                                                                        \
    if ((x) > 2) {                                                                                           \
        CASE_ITEM(2)                                                                                         \
    }                                                                                                        \
    if ((x) > 1) {                                                                                           \
        CASE_ITEM(1)                                                                                         \
    }                                                                                                        \
    if ((x) > 0) {                                                                                           \
        CASE_ITEM(0)                                                                                         \
    }

inline float128_128 loadh_k(SIM_X86::tensor t, int st, int h, int w, float fill) {
    float8_128 data0 = v_u32_move_f(fill);
    float8_128 data1 = v_u32_move_f(fill);
    float8_128 data2 = v_u32_move_f(fill);
    float8_128 data3 = v_u32_move_f(fill);
    float8_128 data4 = v_u32_move_f(fill);
    float8_128 data5 = v_u32_move_f(fill);
    float8_128 data6 = v_u32_move_f(fill);
    float8_128 data7 = v_u32_move_f(fill);
    float8_128 data8 = v_u32_move_f(fill);
    float8_128 data9 = v_u32_move_f(fill);
    float8_128 data10 = v_u32_move_f(fill);
    float8_128 data11 = v_u32_move_f(fill);
    float8_128 data12 = v_u32_move_f(fill);
    float8_128 data13 = v_u32_move_f(fill);
    float8_128 data14 = v_u32_move_f(fill);
    float8_128 data15 = v_u32_move_f(fill);
    int nh = (h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        const int i = (x) * 8;                                                                               \
        const int cur_h = min(h - i, 8);                                                                     \
        const int ldmk = (1 << cur_h) - 1;                                                                   \
        data##x = loadmin8_k(tensor_slice(t, i * st * 128 / 32), st, cur_h, ldmk, w, fill);                  \
    }
    SWITCH_CASES_REV(nh)
#undef CASE_ITEM
    return v_concat_16(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11,
                       data12, data13, data14, data15);
}

inline float128_128 expand(float8_128 v) {
    float8_128 data0 = v;
    float8_128 data1 = v;
    float8_128 data2 = v;
    float8_128 data3 = v;
    float8_128 data4 = v;
    float8_128 data5 = v;
    float8_128 data6 = v;
    float8_128 data7 = v;
    float8_128 data8 = v;
    float8_128 data9 = v;
    float8_128 data10 = v;
    float8_128 data11 = v;
    float8_128 data12 = v;
    float8_128 data13 = v;
    float8_128 data14 = v;
    float8_128 data15 = v;
    return v_concat_16(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11,
                       data12, data13, data14, data15);
}

// [h, padding_w]
inline float128_128 loadh_k_T(SIM_X86::tensor t, int padding_h, int h, int padding_w, int w, float fill, int i,
                              int j) {
    int addr = 128 * j + i * 128 * padding_w;
    int st = padding_w / 128;
    int cur_w = min(w - 128 * j, 128);
    int cur_h = min(h - 128 * i, 128);
    float128_128 v = loadh_k(tensor_slice(t, addr / 32), st, cur_h, cur_w, fill);
    return m_transpose_128_128_nws(v, 0);
}

// [h, padding_w]
inline float128_128 loadh_k_T2(SIM_X86::tensor t, int padding_h, int h, int padding_w, int w, float fill, int i,
                              int j) {
    int addr = i * 128 * 128 + j * 128 * h;
    int cur_w = min(w - 128 * j, 128);
    int cur_h = min(h - 128 * i, 128);
    float128_128 v = loadh_k(tensor_slice(t, addr / 32), 1, cur_h, cur_w, fill);
    return m_transpose_128_128_nws(v, 0);
}

inline float128_128 load128_128_ex(SIM_X86::tensor t, int h, int w, int ih, int iw) {
    int pw = (w + 127) & 0xffffff80;
    return loadh_k(t + (ih * pw + iw) / 32, (w + 127) / 128, min(h - ih, 128), min(w - iw, 128), 0.0f);
}

inline float128_128 load128_128_ex_T(SIM_X86::tensor t, int h, int w, int ih, int iw) {
    int t1 = w;
    w = h;
    h = t1;
    int t2 = iw;
    iw = ih;
    ih = t2;
    return loadh_k(t + (ih * w + iw) / 32, (w + 127) / 128, min(h - ih, 128), min(w - iw, 128), 0.0f);
}

inline float128_128 loadhk_ex(SIM_X86::tensor t, int h, int w, int nh, int nw, int ih, int iw) {
    int pw = (w + 127) & 0xffffff80;
    return loadh_k(t + (ih * pw + iw) / 32, (w + 127) / 128, nh, nw, 0.0f);
}

inline float128_128 loadhk_ex_T(SIM_X86::tensor t, int h, int w, int nh, int nw, int ih, int iw) {
    int t1 = w;
    w = h;
    h = t1;
    int t2 = iw;
    iw = ih;
    ih = t2;
    return loadh_k(t + (ih * w + iw) / 32, (w + 127) / 128, nw, nh, 0.0f);
}

inline void store128_128_ex(SIM_X86::tensor t, int h, int w, int ih, int iw, float128_128 v) {
    int pw = (w + 127) & 0xffffff80;
    // int cur_w = min(w - iw, 128);
    int cur_h = min(h - ih, 128);
    int kS = (cur_h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        int i = (x);                                                                                         \
        int cur_sth = min(cur_h - i * 8, 8);                                                                 \
        store8_128_stride_stmk(((ih + i * 8) * pw + iw) / 32, pw / 128, t, sub_vector(v, x),                 \
                               (1 << cur_sth) - 1);                                                          \
    }
    SWITCH_CASES_REV(kS)
#undef CASE_ITEM
}

inline void store128_128_ex2(SIM_X86::tensor t, int h, int w, int ih, int iw, float128_128 v) {
    // int pw = (w + 127) & 0xffffff80;
    // int cur_w = min(w - iw, 128);
    int cur_h = min(h - ih, 128);
    int kS = (cur_h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        int i = (x);                                                                                         \
        int cur_sth = min(cur_h - i * 8, 8);                                                                 \
        store8_128_stride_stmk((iw * h + (ih + i * 8) * 128) / 32, 1, t, sub_vector(v, x),                 \
                               (1 << cur_sth) - 1);                                                          \
    }
    SWITCH_CASES_REV(kS)
#undef CASE_ITEM
}

#undef SWITCH_CASES_REV

inline void matmul(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw) {
    // int ah128 = ALIGN128(ah);
    // int aw128 = ALIGN128(aw);
    // int bw128 = ALIGN128(bw);
    for (int iah = 0; iah < ah; iah += 128) {
        for (int ibw = 0; ibw < bw; ibw += 128) {
            float128_128 res = expand(v_u32_move_f(0));
            for (int iaw = 0; iaw < aw; iaw += 128) {
                // int cur_ah = min(ah - iah, 128);
                // int cur_aw = min(aw - iaw, 128);
                // int cur_bw = min(bw - ibw, 128);
                float128_128 left = load128_128_ex(A, ah, aw, iah, iaw);
                float128_128 right = load128_128_ex(B, aw, bw, iaw, ibw);
                float128_128 ret = m_matmul_dest_128_128_128(left, right, 0);
                res = add128_128(res, ret);
            }
            store128_128_ex(C, ah, bw, iah, ibw, res);
        }
    }
}

inline void matmul_temp(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw) {
    // int ah128 = ALIGN128(ah);
    // int aw128 = ALIGN128(aw);
    // int bw128 = ALIGN128(bw);
    for (int iah = 0; iah < ah; iah += 128) {
        for (int ibw = 0; ibw < bw; ibw += 128) {
            float128_128 res = expand(v_u32_move_f(0));
            for (int iaw = 0; iaw < aw; iaw += 128) {
                // int cur_ah = min(ah - iah, 128);
                // int cur_aw = min(aw - iaw, 128);
                // int cur_bw = min(bw - ibw, 128);
                float128_128 left = load128_128_ex(A, ah, aw, iah, iaw);
                float128_128 right = load128_128_ex(B, aw, bw, iaw, ibw);
                float128_128 ret = m_matmul_dest_128_128_128(left, right, 0);
                res = add128_128(res, ret);
            }
            float128_128 temp = load128_128_ex(C, ah, bw, iah, ibw);
            res = add128_128(res, temp);
            store128_128_ex(C, ah, bw, iah, ibw, res);
        }
    }
}

// B^T size is [aw, bw]
inline void matmul_t(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw) {
    // int ah128 = ALIGN128(ah);
    // int aw128 = ALIGN128(aw);
    // int bw128 = ALIGN128(bw);
    for (int iah = 0; iah < ah; iah += 128) {
        for (int ibw = 0; ibw < bw; ibw += 128) {
            float128_128 res = expand(v_u32_move_f(0));
            for (int iaw = 0; iaw < aw; iaw += 128) {
                // int cur_ah = min(ah - iah, 128);
                // int cur_aw = min(aw - iaw, 128);
                // int cur_bw = min(bw - ibw, 128);
                float128_128 left = load128_128_ex(A, ah, aw, iah, iaw);
                float128_128 right = load128_128_ex(B, bw, aw, ibw, iaw);
                float128_128 ret = m_matmul_dest_128_128_128_T(left, right, 0);
                res = add128_128(res, ret);
            }
            store128_128_ex(C, ah, bw, iah, ibw, res);
        }
    }
}



// A^T size is [ah, aw]
inline void t_matmul(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw) {
    // int ah128 = ALIGN128(ah);
    // int aw128 = ALIGN128(aw);
    // int bw128 = ALIGN128(bw);
    for (int iah = 0; iah < ah; iah += 128) {
        for (int ibw = 0; ibw < bw; ibw += 128) {
            float128_128 res = expand(v_u32_move_f(0));
            for (int iaw = 0; iaw < aw; iaw += 128) {
                // int cur_ah = min(ah - iah, 128);
                // int cur_aw = min(aw - iaw, 128);
                // int cur_bw = min(bw - ibw, 128);
                float128_128 left = load128_128_ex(A, aw, ah, iaw, iah);
                left = m_transpose_128_128_nws(left, 0);
                float128_128 right = load128_128_ex(B, aw, bw, iaw, ibw);
                float128_128 ret = m_matmul_dest_128_128_128(left, right, 0);
                res = add128_128(res, ret);
            }
            store128_128_ex(C, ah, bw, iah, ibw, res);
        }
    }
}


inline void pgsnf(float128_128 r, bool pgx) {
    push_gsnf(sub_vector(r, 15), pgx);
    push_gsnf(sub_vector(r, 14), pgx);
    push_gsnf(sub_vector(r, 13), pgx);
    push_gsnf(sub_vector(r, 12), pgx);
    push_gsnf(sub_vector(r, 11), pgx);
    push_gsnf(sub_vector(r, 10), pgx);
    push_gsnf(sub_vector(r, 9), pgx);
    push_gsnf(sub_vector(r, 8), pgx);
    push_gsnf(sub_vector(r, 7), pgx);
    push_gsnf(sub_vector(r, 6), pgx);
    push_gsnf(sub_vector(r, 5), pgx);
    push_gsnf(sub_vector(r, 4), pgx);
    push_gsnf(sub_vector(r, 3), pgx);
    push_gsnf(sub_vector(r, 2), pgx);
    push_gsnf(sub_vector(r, 1), pgx);
    push_gsnf(sub_vector(r, 0), pgx);

    float8_128 d = v_u32_move_f(0.0);

    if (pgx) {
        asm volatile("{ MTI@(pr0) mrf<1> = mulgsnf %[d]; MTR@(pr0) %[d] = pop mrf<1>; }"
                     : [d] "=x"(d)
                     :
                     :);
    } else {
        asm volatile("{ MTI@(pr0) mrf<0> = mulgsnf %[d]; MTR@(pr0) %[d] = pop mrf<0>; }"
                     : [d] "=x"(d)
                     :
                     :);
    }
    // volatile float8_128 _ = m_matmul_gsnf(v_u32_move_f(0.0), pgx);
}



inline void matmul_gain(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int ibw, int add_src_flag){
    int n = (ah + 7) / 8;
    int m = min(12 , n);
    int stride = (bw + 127) / 128;
    int bw128 = ALIGN128(bw);
    #pragma clang loop unroll_count(4)
    for(int i = 0; i < m; i ++){
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        m_matmul_single(left0, 0, 0);
    }
    #pragma clang loop unroll_count(4)
    for(int i = 12; i < n; i ++){
        int AoffsetPgx0 = ah * iaw  / 32 + i * 32;
//         int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        int Coffset = (((i - 12) * 8) * bw128 + ibw) / 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0) 
                            ? v_u32_move_f(0.0)
                            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);

        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);

        m_matmul_single(left0, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);

        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    #pragma clang loop unroll_count(2)
    for(int i = n - m; i < n - 1; i ++){
        int Coffset = ((i * 8) * bw128 + ibw) / 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
                            ? v_u32_move_f(0.0)
                            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    if(m != 0){
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = ((i * 8) * bw128 + ibw) / 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
                            ? v_u32_move_f(0.0)
                            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);            
    }          
}

inline void matmul_gain_2pgx(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int bw, int iaw, int ibw, int add_src_flag){
    int n = (ah + 7) / 8;
    int m = min(12 , n);
    int stride = (bw + 127) / 128;
    int bw128 = ALIGN128(bw);
    #pragma clang loop unroll_count(4)
    for(int i = 0; i < m; i ++){
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);

        m_matmul_single(left0, 0, 0);
        m_matmul_single(left1, 0, 1);
    }
    #pragma clang loop unroll_count(4)
    for(int i = 12; i < n; i ++){
        int AoffsetPgx0 = ah * iaw  / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        int Coffset = (((i - 12) * 8) * bw128 + ibw) / 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0) 
                            ? v_u32_move_f(0.0)
                            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);

        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);

        m_matmul_single(left0, 0, 0);
        m_matmul_single(left1, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);  

        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    #pragma clang loop unroll_count(2)
    for(int i = n - m; i < n - 1; i ++){
        int Coffset = ((i * 8) * bw128 + ibw) / 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
                            ? v_u32_move_f(0.0)
                            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    if(m != 0){
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = ((i * 8) * bw128 + ibw) / 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
                            ? v_u32_move_f(0.0)
                            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);            
    }        
}

inline void push_gain_2pgx(SIM_X86::tensor B0, SIM_X86::tensor B1, float scale){
    float8_128 pgx0_gain_0 = v_f32_ld_tnsr_st_msk(480, B0, 1, 255);
    float8_128 pgx1_gain_0 = v_f32_ld_tnsr_st_msk(480, B1, 1, 255);
    float8_128 pgx0_gain_1 = v_f32_ld_tnsr_st_msk(448, B0, 1, 255);
    float8_128 pgx1_gain_1 = v_f32_ld_tnsr_st_msk(448, B1, 1, 255);
    float8_128 pgx0_gain_2 = v_f32_ld_tnsr_st_msk(416, B0, 1, 255);
    float8_128 pgx1_gain_2 = v_f32_ld_tnsr_st_msk(416, B1, 1, 255);
    float8_128 pgx0_gain_3 = v_f32_ld_tnsr_st_msk(384, B0, 1, 255);
    float8_128 pgx1_gain_3 = v_f32_ld_tnsr_st_msk(384, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);

    pgx0_gain_0 = v_f32_ld_tnsr_st_msk(352, B0, 1, 255);
    pgx1_gain_0 = v_f32_ld_tnsr_st_msk(352, B1, 1, 255);
    pgx0_gain_1 = v_f32_ld_tnsr_st_msk(320, B0, 1, 255);
    pgx1_gain_1 = v_f32_ld_tnsr_st_msk(320, B1, 1, 255);
    pgx0_gain_2 = v_f32_ld_tnsr_st_msk(288, B0, 1, 255);
    pgx1_gain_2 = v_f32_ld_tnsr_st_msk(288, B1, 1, 255);
    pgx0_gain_3 = v_f32_ld_tnsr_st_msk(256, B0, 1, 255);
    pgx1_gain_3 = v_f32_ld_tnsr_st_msk(256, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);

    pgx0_gain_0 = v_f32_ld_tnsr_st_msk(224, B0, 1, 255);
    pgx1_gain_0 = v_f32_ld_tnsr_st_msk(224, B1, 1, 255);
    pgx0_gain_1 = v_f32_ld_tnsr_st_msk(192, B0, 1, 255);
    pgx1_gain_1 = v_f32_ld_tnsr_st_msk(192, B1, 1, 255);
    pgx0_gain_2 = v_f32_ld_tnsr_st_msk(160, B0, 1, 255);
    pgx1_gain_2 = v_f32_ld_tnsr_st_msk(160, B1, 1, 255);
    pgx0_gain_3 = v_f32_ld_tnsr_st_msk(128, B0, 1, 255);
    pgx1_gain_3 = v_f32_ld_tnsr_st_msk(128, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);

    pgx0_gain_0 = v_f32_ld_tnsr_st_msk(96, B0, 1, 255);
    pgx1_gain_0 = v_f32_ld_tnsr_st_msk(96, B1, 1, 255);
    pgx0_gain_1 = v_f32_ld_tnsr_st_msk(64, B0, 1, 255);
    pgx1_gain_1 = v_f32_ld_tnsr_st_msk(64, B1, 1, 255);
    pgx0_gain_2 = v_f32_ld_tnsr_st_msk(32, B0, 1, 255);
    pgx1_gain_2 = v_f32_ld_tnsr_st_msk(32, B1, 1, 255);
    pgx0_gain_3 = v_f32_ld_tnsr_st_msk(0, B0, 1, 255);
    pgx1_gain_3 = v_f32_ld_tnsr_st_msk(0, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;
    
    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);
}

// [ah, aw] * [aw, bw] = [ah, bw] 
inline void matmul_aw256(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag) {
    int aw256 = aw & 0xffffff00;
    int last_aw = aw - aw256;
    int last_awn = (last_aw + 7) / 8;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            int BoffsetPgx0 = (aw * ibw + iaw * 128) / 32;
            int BoffsetPgx1 = BoffsetPgx0 + 4 * 128;
            push_gain_2pgx(B + BoffsetPgx0, B + BoffsetPgx1, 1.0);

            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            matmul_gain_2pgx(A, C, ah, bw, iaw, ibw, add_src_flag);
            
        }
        for(;iaw < aw; iaw += 128){
            int BoffsetPgx0 = (aw * ibw + iaw * 128) / 32;
            float8_128 zero = 0;
            for(int i = 15; i >= last_awn; i --){
                push_gsnf(zero, 0);
            }
            if(1){
                int i = last_awn - 1;
                int h = min(last_aw - i * 8, 8);
                int mask = pre_exp2(h);
                float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + i * 32, 1, mask, min(bw - ibw, 128), 0);
                push_gsnf(gain_pgx0, 0);                
            }
            for(int i = last_awn - 2; i >= 0; i --){
                float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + i * 32, 1, 255, min(bw - ibw, 128), 0);
                push_gsnf(gain_pgx0, 0);
            }
            m_fakemul(zero, 0, 0);

            matmul_gain(A, C, ah, aw, bw, iaw, ibw, add_src_flag);
        }
    }
}

inline float8_128 as_float(int8_128 a) {
    return *(float8_128*)&a;
}

inline int8_128 as_int(float8_128 a) {
    return *(int8_128*)&a;
}

inline void push_hi_bf16_2pgx(SIM_X86::tensor B0, SIM_X86::tensor B1, int bw) {
    int stride = (bw + 127) / 128;
    int once_offset = 8 * bw / 32;
    float8_128 pgx0_gain_0 = load8_128_stride(once_offset * 15, stride, B0);
    float8_128 pgx1_gain_0 = load8_128_stride(once_offset * 15, stride, B1);
    float8_128 pgx0_gain_1 = load8_128_stride(once_offset * 14, stride, B0);
    float8_128 pgx1_gain_1 = load8_128_stride(once_offset * 14, stride, B1);
    float8_128 pgx0_gain_2 = load8_128_stride(once_offset * 13, stride, B0);
    float8_128 pgx1_gain_2 = load8_128_stride(once_offset * 13, stride, B1);
    float8_128 pgx0_gain_3 = load8_128_stride(once_offset * 12, stride, B0);
    float8_128 pgx1_gain_3 = load8_128_stride(once_offset * 12, stride, B1);

    pushgain_hi(pgx0_gain_0, 0, 0);
    pushgain_hi(pgx1_gain_0, 0, 1);
    pushgain_hi(pgx0_gain_1, 0, 0);
    pushgain_hi(pgx1_gain_1, 0, 1);
    pushgain_hi(pgx0_gain_2, 0, 0);
    pushgain_hi(pgx1_gain_2, 0, 1);
    pushgain_hi(pgx0_gain_3, 0, 0);
    pushgain_hi(pgx1_gain_3, 0, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 11, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 11, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 10, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 10, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 9, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 9, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 8, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 8, stride, B1);

    pushgain_hi(pgx0_gain_0, 0, 0);
    pushgain_hi(pgx1_gain_0, 0, 1);
    pushgain_hi(pgx0_gain_1, 0, 0);
    pushgain_hi(pgx1_gain_1, 0, 1);
    pushgain_hi(pgx0_gain_2, 0, 0);
    pushgain_hi(pgx1_gain_2, 0, 1);
    pushgain_hi(pgx0_gain_3, 0, 0);
    pushgain_hi(pgx1_gain_3, 0, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 7, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 7, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 6, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 6, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 5, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 5, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 4, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 4, stride, B1);

    pushgain_hi(pgx0_gain_0, 0, 0);
    pushgain_hi(pgx1_gain_0, 0, 1);
    pushgain_hi(pgx0_gain_1, 0, 0);
    pushgain_hi(pgx1_gain_1, 0, 1);
    pushgain_hi(pgx0_gain_2, 0, 0);
    pushgain_hi(pgx1_gain_2, 0, 1);
    pushgain_hi(pgx0_gain_3, 0, 0);
    pushgain_hi(pgx1_gain_3, 0, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 3, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 3, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 2, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 2, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 1, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 1, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 0, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 0, stride, B1);


    pushgain_hi(pgx0_gain_0, 0, 0);
    pushgain_hi(pgx1_gain_0, 0, 1);
    pushgain_hi(pgx0_gain_1, 0, 0);
    pushgain_hi(pgx1_gain_1, 0, 1);
    pushgain_hi(pgx0_gain_2, 0, 0);
    pushgain_hi(pgx1_gain_2, 0, 1);
    pushgain_hi(pgx0_gain_3, 0, 0);
    pushgain_hi(pgx1_gain_3, 0, 1);
}

inline void push_lo_bf16_2pgx(SIM_X86::tensor B0, SIM_X86::tensor B1, int bw) {
    int stride = bw / 128;
    int once_offset = 8 * bw / 32;
    float8_128 pgx0_gain_0 = load8_128_stride(once_offset * 15, stride, B0);
    float8_128 pgx1_gain_0 = load8_128_stride(once_offset * 15, stride, B1);
    float8_128 pgx0_gain_1 = load8_128_stride(once_offset * 14, stride, B0);
    float8_128 pgx1_gain_1 = load8_128_stride(once_offset * 14, stride, B1);
    float8_128 pgx0_gain_2 = load8_128_stride(once_offset * 13, stride, B0);
    float8_128 pgx1_gain_2 = load8_128_stride(once_offset * 13, stride, B1);
    float8_128 pgx0_gain_3 = load8_128_stride(once_offset * 12, stride, B0);
    float8_128 pgx1_gain_3 = load8_128_stride(once_offset * 12, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 11, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 11, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 10, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 10, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 9, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 9, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 8, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 8, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 7, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 7, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 6, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 6, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 5, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 5, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 4, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 4, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 3, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 3, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 2, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 2, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 1, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 1, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 0, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 0, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);
}


inline void packed_matmul_gain(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int ibw, int bf_ibw, int add_src_flag) {
    int n = (ah + 15) / 16;
    int m = min(6, n);
    int stride = (bw + 127) / 128;
    int bw128 = ALIGN128(bw);
    int bf_bw = ((bw128 + 255) / 256) * 128;
    int bf_stride = bf_bw / 128;
#pragma clang loop unroll_count(2)
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        m_matmul_packed_single(left0, 0, 0);
    }
#pragma clang loop unroll_count(2)
    for (int i = 6; i < n; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int Coffset = (((i - 6) * 16) * bw128 + ibw) / 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 res1 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);

        m_matmul_packed_single(left0, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(0);

        res0 = res0 + ret0;
        res1 = res1 + ret1;

        if(ibw % 256){
            int OutOffset = (((i - 6) * 16) * bf_bw + bf_ibw) / 32;
            Coffset = (((i - 6) * 16) * bw128 + ibw - 128) / 32;
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
            float8_128 res1_lo = load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(float_to_bfloat16(res1, res1_lo)));
        }else if(ibw == bw128 - 128){
            int OutOffset = (((i - 6) * 16) * bf_bw + bf_ibw) / 32;
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(v_u32_shr(as_int(res0), 16)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(v_u32_shr(as_int(res1), 16)));
        }
        else{
            store8_128_stride_stmk(Coffset, stride, C, res0, 255);
            store8_128_stride_stmk(Coffset + 8 * bw128 / 32, stride, C, res1, 255);
        }
    }
#pragma clang loop unroll_count(2)
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = ((i * 16) * bw128 + ibw) / 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 res1 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(0);
        res0 = res0 + ret0;
        res1 = res1 + ret1;

        if(ibw % 256){
            int OutOffset = ((i * 16) * bf_bw + bf_ibw) / 32;
            Coffset = ((i * 16) * bw128 + ibw - 128) / 32;
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
            float8_128 res1_lo = load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(float_to_bfloat16(res1, res1_lo)));
        }else if(ibw == bw128 - 128){
            int OutOffset = ((i * 16) * bf_bw + bf_ibw) / 32;
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(v_u32_shr(as_int(res0), 16)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(v_u32_shr(as_int(res1), 16)));
        }
        else{
            store8_128_stride_stmk(Coffset, stride, C, res0, 255);
            store8_128_stride_stmk(Coffset + 8 * bw128 / 32, stride, C, res1, 255);
        }
    }
    if (m != 0) {
        int i = n - 1;
//  int ori_n = (ah + 7) / 8;
//  int ori_i = ori_n - 1;
//  int h = min(ah - ori_i * 8, 8);
        int Coffset = ((i * 16) * bw128 + ibw) / 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 res1 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(0);
        res0 = res0 + ret0;
        res1 = res1 + ret1;


        if(ibw % 256){
            int OutOffset = ((i * 16) * bf_bw + bf_ibw) / 32;
            Coffset = ((i * 16) * bw128 + ibw - 128) / 32;
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
            float8_128 res1_lo = load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(float_to_bfloat16(res1, res1_lo)));
        }else if(ibw == bw128 - 128){
            int OutOffset = ((i * 16) * bf_bw + bf_ibw) / 32;
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(v_u32_shr(as_int(res0), 16)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(v_u32_shr(as_int(res1), 16)));
        }
        else{
            store8_128_stride_stmk(Coffset, stride, C, res0, 255);
            store8_128_stride_stmk(Coffset + 8 * bw128 / 32, stride, C, res1, 255);
        }
    }
}

inline void packed_matmul_gain_2pgx(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int bw, int iaw, int ibw, int bf_ibw, int isLast, int add_src_flag) {
    int n = (ah + 15) / 16;
    int m = min(6, n);
    int stride = (bw + 127) / 128;
    int bw128 = ALIGN128(bw);
    int bf_bw = ((bw128 + 255) / 256) * 128;
    int bf_stride = bf_bw / 128;
#pragma clang loop unroll_count(2)
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);
        m_matmul_packed_single(left0, 0, 0);
        m_matmul_packed_single(left1, 0, 1);
    }

#pragma clang loop unroll_count(2)
    for (int i = 6; i < n; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        int Coffset = (((i - 6) * 16) * bw128 + ibw) / 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 res1 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);

        m_matmul_packed_single(left0, 0, 0);
        m_matmul_packed_single(left1, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        float8_128 ret2 = m_pop_mrf(0);
        float8_128 ret3 = m_pop_mrf(1);
        res0 = res0 + ret0;
        res0 = res0 + ret1;
        res1 = res1 + ret2;
        res1 = res1 + ret3;
        if(isLast && (ibw % 256)){
            int OutOffset = (((i - 6) * 16) * bf_bw + bf_ibw) / 32;
            Coffset = (((i - 6) * 16) * bw128 + ibw - 128) / 32;
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
            float8_128 res1_lo = load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(float_to_bfloat16(res1, res1_lo)));
        }else{
            store8_128_stride_stmk(Coffset, stride, C, res0, 255);
            store8_128_stride_stmk(Coffset + 8 * bw128 / 32, stride, C, res1, 255);
        }
    }
#pragma clang loop unroll_count(2)
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = ((i * 16) * bw128 + ibw) / 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 res1 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        float8_128 ret2 = m_pop_mrf(0);
        float8_128 ret3 = m_pop_mrf(1);
        res0 = res0 + ret0;
        res0 = res0 + ret1;
        res1 = res1 + ret2;
        res1 = res1 + ret3;
        if(isLast && (ibw % 256)){
            int OutOffset = ((i * 16) * bf_bw + bf_ibw) / 32;
            Coffset = ((i * 16) * bw128 + ibw - 128) / 32;
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
            float8_128 res1_lo = load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(float_to_bfloat16(res1, res1_lo)));
        }else{
            store8_128_stride_stmk(Coffset, stride, C, res0, 255);
            store8_128_stride_stmk(Coffset + 8 * bw128 / 32, stride, C, res1, 255);
        }
    }
    if (m != 0) {
        int i = n - 1;
//      int ori_n = (ah + 7) / 8;
//      int ori_i = ori_n - 1;
//      int h = min(ah - ori_i * 8, 8);
        int Coffset = ((i * 16) * bw128 + ibw) / 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 res1 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        float8_128 ret2 = m_pop_mrf(0);
        float8_128 ret3 = m_pop_mrf(1);
        res0 = res0 + ret0;
        res0 = res0 + ret1;
        res1 = res1 + ret2;
        res1 = res1 + ret3;
        if(isLast && (ibw % 256)){
            int OutOffset = ((i * 16) * bf_bw + bf_ibw) / 32;
            Coffset = ((i * 16) * bw128 + ibw - 128) / 32;
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
            float8_128 res1_lo = load8_128_stride_with_ldmask(Coffset + 8 * bw128 / 32, stride, 255, C);
            store8_128_stride_with_stmask(OutOffset, bf_stride, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
            store8_128_stride_with_stmask(OutOffset + 8 * bf_bw / 32, bf_stride, 255, D, as_float(float_to_bfloat16(res1, res1_lo)));
        }else{
            store8_128_stride_stmk(Coffset, stride, C, res0, 255);
            store8_128_stride_stmk(Coffset + 8 * bw128 / 32, stride, C, res1, 255);
        }
    }
}

inline void matmul_bf16_aw256(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, SIM_X86::tensor D,int ah, int aw, int bw, int add_src_flag) {
    int aw256 = aw & 0xffffff00;
    int bf_bw = (ALIGN128(bw) / ALIGN128(bw)) * ((bw + 255) / 256) * 256 / 2;
    int last_aw = aw - aw256;
    int last_awn = (last_aw + 7) / 8;
    int bf_ibw = 0;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            int BoffsetPgx0 = (bf_bw * iaw + bf_ibw) / 32;
            int BoffsetPgx1 = BoffsetPgx0 + bf_bw * 128 / 32;
            if (ibw % 256) {
                push_hi_bf16_2pgx(B + BoffsetPgx0, B + BoffsetPgx1, bf_bw);
            }
            else {
                push_lo_bf16_2pgx(B + BoffsetPgx0, B + BoffsetPgx1, bf_bw);
            }

            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            packed_matmul_gain_2pgx(A, C, D, ah, bw, iaw, ibw, bf_ibw, (iaw == aw - 256), add_src_flag);
        }
        for (;iaw < aw; iaw += 128) {
            int BoffsetPgx0 = (bf_bw * iaw + bf_ibw) / 32;
            int stride = (bf_bw + 127) / 128;
            int once_offset = 8 * bf_bw / 32;
            float8_128 zero = 0;
            for (int i = 15; i >= last_awn; i--) {
                push_gsnf(zero, 0);
            }
            if (ibw % 256) {
                if (1) {
                    int i = last_awn - 1;
                    int h = min(last_aw - i * 8, 8);
                    int mask = pre_exp2(h);
                    float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, mask, min(bw - ibw, 128), 0);
                    pushgain_hi(gain_pgx0, 0, 0);
                }
                for (int i = last_awn - 2; i >= 0; i--) {
                    float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, 255, min(bw - ibw, 128), 0);
                    pushgain_hi(gain_pgx0, 0, 0);
                }
            }
            else {
                if (1) {
                    int i = last_awn - 1;
                    int h = min(last_aw - i * 8, 8);
                    int mask = pre_exp2(h);
                    float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, mask, min(bw - ibw, 128), 0);
                    gain_pgx0 = as_float(v_u32_shl(as_int(gain_pgx0), v_u32_move_i(16)));
                    push_gsnf(gain_pgx0, 0);
                }
                for (int i = last_awn - 2; i >= 0; i--) {
                    float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, 255, min(bw - ibw, 128), 0);
                    gain_pgx0 = as_float(v_u32_shl(as_int(gain_pgx0), v_u32_move_i(16)));
                    push_gsnf(gain_pgx0, 0);
                }
            }
            m_fakemul(zero, 0, 0);

            packed_matmul_gain(A, C, D, ah, aw, bw, iaw, ibw, bf_ibw, add_src_flag);
        }
        if (ibw % 256) bf_ibw += 128;
    }
}

inline void push_gstf_2pgx(SIM_X86::tensor B0, SIM_X86::tensor B1, float scale){
    float8_128 pgx0_gain_0 = v_f32_ld_tnsr_st_msk(480, B0, 1, 255);
    float8_128 pgx1_gain_0 = v_f32_ld_tnsr_st_msk(480, B1, 1, 255);
    float8_128 pgx0_gain_1 = v_f32_ld_tnsr_st_msk(448, B0, 1, 255);
    float8_128 pgx1_gain_1 = v_f32_ld_tnsr_st_msk(448, B1, 1, 255);
    float8_128 pgx0_gain_2 = v_f32_ld_tnsr_st_msk(416, B0, 1, 255);
    float8_128 pgx1_gain_2 = v_f32_ld_tnsr_st_msk(416, B1, 1, 255);
    float8_128 pgx0_gain_3 = v_f32_ld_tnsr_st_msk(384, B0, 1, 255);
    float8_128 pgx1_gain_3 = v_f32_ld_tnsr_st_msk(384, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);

    pgx0_gain_0 = v_f32_ld_tnsr_st_msk(352, B0, 1, 255);
    pgx1_gain_0 = v_f32_ld_tnsr_st_msk(352, B1, 1, 255);
    pgx0_gain_1 = v_f32_ld_tnsr_st_msk(320, B0, 1, 255);
    pgx1_gain_1 = v_f32_ld_tnsr_st_msk(320, B1, 1, 255);
    pgx0_gain_2 = v_f32_ld_tnsr_st_msk(288, B0, 1, 255);
    pgx1_gain_2 = v_f32_ld_tnsr_st_msk(288, B1, 1, 255);
    pgx0_gain_3 = v_f32_ld_tnsr_st_msk(256, B0, 1, 255);
    pgx1_gain_3 = v_f32_ld_tnsr_st_msk(256, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);

    pgx0_gain_0 = v_f32_ld_tnsr_st_msk(224, B0, 1, 255);
    pgx1_gain_0 = v_f32_ld_tnsr_st_msk(224, B1, 1, 255);
    pgx0_gain_1 = v_f32_ld_tnsr_st_msk(192, B0, 1, 255);
    pgx1_gain_1 = v_f32_ld_tnsr_st_msk(192, B1, 1, 255);
    pgx0_gain_2 = v_f32_ld_tnsr_st_msk(160, B0, 1, 255);
    pgx1_gain_2 = v_f32_ld_tnsr_st_msk(160, B1, 1, 255);
    pgx0_gain_3 = v_f32_ld_tnsr_st_msk(128, B0, 1, 255);
    pgx1_gain_3 = v_f32_ld_tnsr_st_msk(128, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);

    pgx0_gain_0 = v_f32_ld_tnsr_st_msk(96, B0, 1, 255);
    pgx1_gain_0 = v_f32_ld_tnsr_st_msk(96, B1, 1, 255);
    pgx0_gain_1 = v_f32_ld_tnsr_st_msk(64, B0, 1, 255);
    pgx1_gain_1 = v_f32_ld_tnsr_st_msk(64, B1, 1, 255);
    pgx0_gain_2 = v_f32_ld_tnsr_st_msk(32, B0, 1, 255);
    pgx1_gain_2 = v_f32_ld_tnsr_st_msk(32, B1, 1, 255);
    pgx0_gain_3 = v_f32_ld_tnsr_st_msk(0, B0, 1, 255);
    pgx1_gain_3 = v_f32_ld_tnsr_st_msk(0, B1, 1, 255);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);
}

inline void push_gstf_2pgx_stride(SIM_X86::tensor B0, SIM_X86::tensor B1, int src_stride, float scale){
    float8_128 pgx0_gain_0 = load8_128_stride(480 * src_stride, src_stride, B0);
    float8_128 pgx1_gain_0 = load8_128_stride(480 * src_stride, src_stride, B1);
    float8_128 pgx0_gain_1 = load8_128_stride(448 * src_stride, src_stride, B0);
    float8_128 pgx1_gain_1 = load8_128_stride(448 * src_stride, src_stride, B1);
    float8_128 pgx0_gain_2 = load8_128_stride(416 * src_stride, src_stride, B0);
    float8_128 pgx1_gain_2 = load8_128_stride(416 * src_stride, src_stride, B1);
    float8_128 pgx0_gain_3 = load8_128_stride(384 * src_stride, src_stride, B0);
    float8_128 pgx1_gain_3 = load8_128_stride(384 * src_stride, src_stride, B1);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);

    pgx0_gain_0 = load8_128_stride(352 * src_stride, src_stride, B0);
    pgx1_gain_0 = load8_128_stride(352 * src_stride, src_stride, B1);
    pgx0_gain_1 = load8_128_stride(320 * src_stride, src_stride, B0);
    pgx1_gain_1 = load8_128_stride(320 * src_stride, src_stride, B1);
    pgx0_gain_2 = load8_128_stride(288 * src_stride, src_stride, B0);
    pgx1_gain_2 = load8_128_stride(288 * src_stride, src_stride, B1);
    pgx0_gain_3 = load8_128_stride(256 * src_stride, src_stride, B0);
    pgx1_gain_3 = load8_128_stride(256 * src_stride, src_stride, B1);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);

    pgx0_gain_0 = load8_128_stride(224 * src_stride, src_stride, B0);
    pgx1_gain_0 = load8_128_stride(224 * src_stride, src_stride, B1);
    pgx0_gain_1 = load8_128_stride(192 * src_stride, src_stride, B0);
    pgx1_gain_1 = load8_128_stride(192 * src_stride, src_stride, B1);
    pgx0_gain_2 = load8_128_stride(160 * src_stride, src_stride, B0);
    pgx1_gain_2 = load8_128_stride(160 * src_stride, src_stride, B1);
    pgx0_gain_3 = load8_128_stride(128 * src_stride, src_stride, B0);
    pgx1_gain_3 = load8_128_stride(128 * src_stride, src_stride, B1);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);

    pgx0_gain_0 = load8_128_stride(96 * src_stride, src_stride, B0);
    pgx1_gain_0 = load8_128_stride(96 * src_stride, src_stride, B1);
    pgx0_gain_1 = load8_128_stride(64 * src_stride, src_stride, B0);
    pgx1_gain_1 = load8_128_stride(64 * src_stride, src_stride, B1);
    pgx0_gain_2 = load8_128_stride(32 * src_stride, src_stride, B0);
    pgx1_gain_2 = load8_128_stride(32 * src_stride, src_stride, B1);
    pgx0_gain_3 = load8_128_stride(0 * src_stride, src_stride, B0);
    pgx1_gain_3 = load8_128_stride(0 * src_stride, src_stride, B1);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx1_gain_0, 1);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx1_gain_1, 1);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx1_gain_2, 1);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx1_gain_3, 1);
}

inline void push_gstf_1pgxw(SIM_X86::tensor B, int w, float scale){
    int8_128 c = get_core_id();
    bool8_128 m = v_s32_cmp(LS, v_u32_and(c, v_u32_move_i(127)), v_u32_move_i(w));
    // int a = 10;
    float8_128 pgx0_gain_0 = v_f32_ld_tnsr_st_msk(480, B, 1, 255);
    float8_128 pgx0_gain_1 = v_f32_ld_tnsr_st_msk(448, B, 1, 255);
    float8_128 pgx0_gain_2 = v_f32_ld_tnsr_st_msk(416, B, 1, 255);
    float8_128 pgx0_gain_3 = v_f32_ld_tnsr_st_msk(384, B, 1, 255);
    float8_128 pgx0_gain_4 = v_f32_ld_tnsr_st_msk(352, B, 1, 255);
    float8_128 pgx0_gain_5 = v_f32_ld_tnsr_st_msk(320, B, 1, 255);
    float8_128 pgx0_gain_6 = v_f32_ld_tnsr_st_msk(288, B, 1, 255);
    float8_128 pgx0_gain_7 = v_f32_ld_tnsr_st_msk(256, B, 1, 255);
    // Print("pgx0_gain_0", pgx0_gain_0[10]);
    pgx0_gain_0 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_0);
    pgx0_gain_1 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_1);
    pgx0_gain_2 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_2);
    pgx0_gain_3 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_3);
    pgx0_gain_4 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_4);
    pgx0_gain_5 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_5);
    pgx0_gain_6 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_6);
    pgx0_gain_7 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_7);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx0_gain_4 = pgx0_gain_4 * scale;
    pgx0_gain_5 = pgx0_gain_5 * scale;
    pgx0_gain_6 = pgx0_gain_6 * scale;
    pgx0_gain_7 = pgx0_gain_7 * scale;

    // Print("pgx0_gain_0:%f\n", pgx0_gain_0);
    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx0_gain_4, 0);
    push_gstf(pgx0_gain_5, 0);
    push_gstf(pgx0_gain_6, 0);
    push_gstf(pgx0_gain_7, 0);

    pgx0_gain_0 = v_f32_ld_tnsr_st_msk(224, B, 1, 255);
    pgx0_gain_1 = v_f32_ld_tnsr_st_msk(192, B, 1, 255);
    pgx0_gain_2 = v_f32_ld_tnsr_st_msk(160, B, 1, 255);
    pgx0_gain_3 = v_f32_ld_tnsr_st_msk(128, B, 1, 255);
    pgx0_gain_4 = v_f32_ld_tnsr_st_msk(96, B, 1, 255);
    pgx0_gain_5 = v_f32_ld_tnsr_st_msk(64, B, 1, 255);
    pgx0_gain_6 = v_f32_ld_tnsr_st_msk(32, B, 1, 255);
    pgx0_gain_7 = v_f32_ld_tnsr_st_msk(0, B, 1, 255);

    pgx0_gain_0 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_0);
    pgx0_gain_1 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_1);
    pgx0_gain_2 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_2);
    pgx0_gain_3 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_3);
    pgx0_gain_4 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_4);
    pgx0_gain_5 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_5);
    pgx0_gain_6 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_6);
    pgx0_gain_7 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_7);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx0_gain_4 = pgx0_gain_4 * scale;
    pgx0_gain_5 = pgx0_gain_5 * scale;
    pgx0_gain_6 = pgx0_gain_6 * scale;
    pgx0_gain_7 = pgx0_gain_7 * scale;

    push_gstf(pgx0_gain_0, 0);
    push_gstf(pgx0_gain_1, 0);
    push_gstf(pgx0_gain_2, 0);
    push_gstf(pgx0_gain_3, 0);
    push_gstf(pgx0_gain_4, 0);
    push_gstf(pgx0_gain_5, 0);
    push_gstf(pgx0_gain_6, 0);
    push_gstf(pgx0_gain_7, 0);
}

inline void push_gstf_1pgxw_stride(SIM_X86::tensor B, int w, float scale, int stride, int pgx){
    int8_128 c = get_core_id();
    bool8_128 m = v_s32_cmp(LS, v_u32_and(c, v_u32_move_i(127)), v_u32_move_i(w));
    // int a = 10;
    float8_128 pgx0_gain_0 = load8_128_stride(480 * stride, stride, B);
    float8_128 pgx0_gain_1 = load8_128_stride(448 * stride, stride, B);
    float8_128 pgx0_gain_2 = load8_128_stride(416 * stride, stride, B);
    float8_128 pgx0_gain_3 = load8_128_stride(384 * stride, stride, B);
    float8_128 pgx0_gain_4 = load8_128_stride(352 * stride, stride, B);
    float8_128 pgx0_gain_5 = load8_128_stride(320 * stride, stride, B);
    float8_128 pgx0_gain_6 = load8_128_stride(288 * stride, stride, B);
    float8_128 pgx0_gain_7 = load8_128_stride(256 * stride, stride, B);

    pgx0_gain_0 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_0);
    pgx0_gain_1 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_1);
    pgx0_gain_2 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_2);
    pgx0_gain_3 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_3);
    pgx0_gain_4 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_4);
    pgx0_gain_5 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_5);
    pgx0_gain_6 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_6);
    pgx0_gain_7 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_7);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx0_gain_4 = pgx0_gain_4 * scale;
    pgx0_gain_5 = pgx0_gain_5 * scale;
    pgx0_gain_6 = pgx0_gain_6 * scale;
    pgx0_gain_7 = pgx0_gain_7 * scale;

    push_gstf(pgx0_gain_0, pgx);
    push_gstf(pgx0_gain_1, pgx);
    push_gstf(pgx0_gain_2, pgx);
    push_gstf(pgx0_gain_3, pgx);
    push_gstf(pgx0_gain_4, pgx);
    push_gstf(pgx0_gain_5, pgx);
    push_gstf(pgx0_gain_6, pgx);
    push_gstf(pgx0_gain_7, pgx);

    pgx0_gain_0 = load8_128_stride(224 * stride, stride, B);
    pgx0_gain_1 = load8_128_stride(192 * stride, stride, B);
    pgx0_gain_2 = load8_128_stride(160 * stride, stride, B);
    pgx0_gain_3 = load8_128_stride(128 * stride, stride, B);
    pgx0_gain_4 = load8_128_stride(96 * stride, stride, B);
    pgx0_gain_5 = load8_128_stride(64 * stride, stride, B);
    pgx0_gain_6 = load8_128_stride(32 * stride, stride, B);
    pgx0_gain_7 = load8_128_stride(0 * stride, stride, B);

    pgx0_gain_0 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_0);
    pgx0_gain_1 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_1);
    pgx0_gain_2 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_2);
    pgx0_gain_3 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_3);
    pgx0_gain_4 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_4);
    pgx0_gain_5 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_5);
    pgx0_gain_6 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_6);
    pgx0_gain_7 = v_f32_sel(m, v_u32_move_f(0), pgx0_gain_7);

    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx0_gain_4 = pgx0_gain_4 * scale;
    pgx0_gain_5 = pgx0_gain_5 * scale;
    pgx0_gain_6 = pgx0_gain_6 * scale;
    pgx0_gain_7 = pgx0_gain_7 * scale;

    push_gstf(pgx0_gain_0, pgx);
    push_gstf(pgx0_gain_1, pgx);
    push_gstf(pgx0_gain_2, pgx);
    push_gstf(pgx0_gain_3, pgx);
    push_gstf(pgx0_gain_4, pgx);
    push_gstf(pgx0_gain_5, pgx);
    push_gstf(pgx0_gain_6, pgx);
    push_gstf(pgx0_gain_7, pgx);
}

// [ah, aw] * [bw, aw] = [ah, bw] 
inline void matmul_aw256_T(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag) {
    uint aw256 = aw & 0xffffff00;
    uint bw128 = bw & 0xffffff80;
    uint last_bw = bw - bw128;
    uint last_bwn = (last_bw + 7) / 8;
    for (uint ibw = 0; ibw < bw128; ibw += 128) {
        uint iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            uint BoffsetPgx0 = (bw * iaw + ibw * 128) / 32;
            uint BoffsetPgx1 = BoffsetPgx0 + bw * 4;
            push_gstf_2pgx(B + BoffsetPgx0, B + BoffsetPgx1, 1.0);

            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            matmul_gain_2pgx(A, C, ah, bw, iaw, ibw, add_src_flag);
            
        }
        for(;iaw < aw; iaw += 128){
            uint BoffsetPgx = (bw * iaw + ibw * 128) / 32;
            push_gstf_1pgxw(B + BoffsetPgx, min(aw - iaw, 128), 1.0);
            m_fakemul(v_u32_move_b(0), 1, 0);
            matmul_gain(A, C, ah, aw, bw, iaw, ibw, add_src_flag);
        }
    }
    if(last_bwn){
        int iaw = 0;
        float8_128 zero = 0;
        for (; iaw < aw256; iaw += 256) {
            int BoffsetPgx0 = (bw * iaw + bw128 * 128) / 32;
            int BoffsetPgx1 = BoffsetPgx0 + bw * 4;
            for(int i = 15; i >= last_bwn; i --){
                push_gstf(zero, 0);
                push_gstf(zero, 1);
            }

            for(int i = last_bwn - 1; i >= 0; i --){
                float8_128 gain_pgx0 = load8_128_stride_ldmk(BoffsetPgx0+ i * 32, 1, B, 255);
                float8_128 gain_pgx1 = load8_128_stride_ldmk(BoffsetPgx1+ i * 32, 1, B, 255);
                push_gstf(gain_pgx0, 0);
                push_gstf(gain_pgx1, 1);  
            }
            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            matmul_gain_2pgx(A, C, ah, bw, iaw, bw128, add_src_flag);
        }
        for(;iaw < aw; iaw += 128){
            int BoffsetPgx = (bw * iaw + bw128 * 128) / 32;
            float8_128 zero = 0;
            for(int i = 15; i >= last_bwn; i --){
                push_gstf(zero, 0);
            }
            int8_128 c = get_core_id();
            bool8_128 m = v_s32_cmp(LS, v_u32_and(c, v_u32_move_i(127)), v_u32_move_i(min(aw - iaw, 128)));
            for(int i = last_bwn - 1; i >= 0; i --){
                float8_128 gain_pgx0 =load8_128_stride_ldmk(BoffsetPgx+ i * 32, 1, B, 255);
                push_gstf(v_f32_sel(m, v_u32_move_f(0), gain_pgx0), 0);
            }
            m_fakemul(zero, 1, 0);
            matmul_gain(A, C, ah, aw, bw, iaw, bw128, add_src_flag);
        }
    }

}

inline void matmul_2pgx_256_pre16_bf16(SIM_X86::tensor A0) {
    float8_128 x0 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);
    float8_128 x1 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    short8_128 pgx10_h = unpack_16b(as_int(x0), 1);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);
    float8_128 pgx10 = bfloat16_to_float(pgx10_h);

    float8_128 x2 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    short8_128 pgx11_h = unpack_16b(as_int(x1), 1);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);
    float8_128 pgx11 = bfloat16_to_float(pgx11_h);

    float8_128 x3 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    short8_128 pgx12_h = unpack_16b(as_int(x2), 1);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);
    float8_128 pgx12 = bfloat16_to_float(pgx12_h);


    //第一部分的[64, 128] * [128, 128]
    m_matmul_single(pgx00, 0, 0);
    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    short8_128 pgx13_h = unpack_16b(as_int(x3), 1);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    float8_128 pgx13 = bfloat16_to_float(pgx13_h);

    x0 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx01, 0, 0);

    x1 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    m_matmul_single(pgx02, 0, 0);

    x2 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx03, 0, 0);

    x3 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);

    //第二部分的[64, 128] * [128, 128]
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx00, 0, 0);

    x0 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx01, 0, 0);

    x1 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    m_matmul_single(pgx02, 0, 0);

    x2 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx03, 0, 0);

    x3 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);

    //第三部分的[64, 128] * [128, 128]
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx00, 0, 0);

    x0 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx01, 0, 0);

    x1 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    m_matmul_single(pgx02, 0, 0);

    x2 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx03, 0, 0);

    x3 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);

    //第三部分的[64, 128] * [128, 128]
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx00, 0, 0);

    m_matmul_single(pgx10, 0, 1);

    m_matmul_single(pgx01, 0, 0);

    m_matmul_single(pgx11, 0, 1);

    m_matmul_single(pgx02, 0, 0);

    m_matmul_single(pgx12, 0, 1);

    m_matmul_single(pgx03, 0, 0);

    m_matmul_single(pgx13, 0, 1);
}

//matmul when ah = 256, full loop unroll, 2pgx, iaw == 0.
//it performs the last 16 matmuls, and pop the results, converting results to bf16 and store it on SIM_X86::tensor D at the same time.
inline void matmul_2pgx_256_pop_first_store_bf16(SIM_X86::tensor A0, SIM_X86::tensor C, SIM_X86::tensor D, int ibw, int ah, int bw128) {
    float8_128 x0 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);

    float8_128 x1 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    short8_128 pgx10_h = unpack_16b(as_int(x0), 1);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);
    float8_128 pgx10 = bfloat16_to_float(pgx10_h);

    float8_128 x2 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    short8_128 pgx11_h = unpack_16b(as_int(x1), 1);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);
    float8_128 pgx11 = bfloat16_to_float(pgx11_h);
    float8_128 x3 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);

    float8_128 res0 = m_pop_mrf(0);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    short8_128 pgx12_h = unpack_16b(as_int(x2), 1);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);
    float8_128 pgx12 = bfloat16_to_float(pgx12_h);

    float8_128 res1 = m_pop_mrf(1);
    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    short8_128 pgx13_h = unpack_16b(as_int(x3), 1);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    float8_128 pgx13 = bfloat16_to_float(pgx13_h);
    //第一个[32, 128] * [128, 128]
    //1
    m_matmul_single(pgx00, 0, 0);
    float8_128 res2 = m_pop_mrf(0);

    //2
    x0 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    float8_128 res3 = m_pop_mrf(1);

    //3
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx01, 0, 0);
    float8_128 res4 = m_pop_mrf(0);

    //4
    x1 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    float8_128 res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    m_matmul_single(pgx02, 0, 0);
    float8_128 res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    x2 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    float8_128 res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    x3 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    x0 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //11
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx01, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    //12
    x1 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    res5 = m_pop_mrf(1);

    //13
    res0 = res0 + res1;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    m_matmul_single(pgx02, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;
    x2 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx03, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res0 = m_pop_mrf(0);

    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    x3 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx00, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //19
    m_matmul_single(pgx01, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    //20
    m_matmul_single(pgx11, 0, 1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    res5 = m_pop_mrf(1);

    //21
    res0 = res0 + res1;
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    m_matmul_single(pgx03, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res0 = m_pop_mrf(0);

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);

    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    res5 = m_pop_mrf(1);

    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
    x0 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }

    x1 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    x2 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);

    x3 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);

    //第四个[32, 128] * [128, 128]
    //25
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);

    m_matmul_single(pgx00, 0, 0);
    //26
    m_matmul_single(pgx10, 0, 1);
    //27
    m_matmul_single(pgx01, 0, 0);
    //28
    m_matmul_single(pgx11, 0, 1);
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    m_matmul_single(pgx12, 0, 1);
    //31
    m_matmul_single(pgx03, 0, 0);
    //32
    m_matmul_single(pgx13, 0, 1);

    C += 4 * 128;
    D += 4 * 128;
    res0 = m_pop_mrf(0);

    res1 = m_pop_mrf(1);

    res2 = m_pop_mrf(0);

    res3 = m_pop_mrf(1);


    res4 = m_pop_mrf(0);


    res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    res1 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    //12
    res5 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    //13
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    res1 = m_pop_mrf(1);


    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    res3 = m_pop_mrf(1);


    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    //19
    res4 = m_pop_mrf(0);

    //20
    res5 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    //21
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    res1 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //26
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    //27
    res4 = m_pop_mrf(0);

    //28
    res5 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    //29
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //30
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //31
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //32
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
}
//matmul when ah = 256, full loop unroll, 2pgx.
//it performs the last 16 matmuls, and pop the results, converting results to bf16 and store it on SIM_X86::tensor D at the same time.
inline void matmul_2pgx_256_pop_store_bf16(SIM_X86::tensor A0, SIM_X86::tensor C, SIM_X86::tensor D, int ibw, int ah, int bw128) {
    float8_128 res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    float8_128 res4 = m_pop_mrf(0);

    float8_128 res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    float8_128 res5 = m_pop_mrf(1);

    float8_128 res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    float8_128 res6 = m_pop_mrf(0);

    float8_128 res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    float8_128 res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    float8_128 x0 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);

    res1 = res1 + res6;
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    short8_128 pgx10_h = unpack_16b(as_int(x0), 1);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);
    float8_128 pgx10 = bfloat16_to_float(pgx10_h);

    res0 = res0 + res5;
    float8_128 x1 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    short8_128 pgx11_h = unpack_16b(as_int(x1), 1);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);
    float8_128 pgx11 = bfloat16_to_float(pgx11_h);
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    float8_128 x2 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    res5 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    short8_128 pgx12_h = unpack_16b(as_int(x2), 1);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);
    float8_128 pgx12 = bfloat16_to_float(pgx12_h);
    res7 = m_pop_mrf(1);

    res3 = res3 + res5;
    float8_128 x3 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);

    res3 = res3 + res7;
    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    short8_128 pgx13_h = unpack_16b(as_int(x3), 1);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    float8_128 pgx13 = bfloat16_to_float(pgx13_h);

    //第一个[32, 128] * [128, 128]
    //1

    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);
    m_matmul_single(pgx00, 0, 0);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res4 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    m_matmul_single(pgx01, 0, 0);
    res5 = m_pop_mrf(1);

    //4
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    m_matmul_single(pgx02, 0, 0);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;
    x0 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    //7
    res1 = res1 + res6;
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx03, 0, 0);

    //8
    res0 = res0 + res5;
    x1 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);

    m_matmul_single(pgx00, 0, 0);
    res6 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    x2 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res5 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx01, 0, 0);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;
    x3 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    //13
    res3 = res3 + res7;
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx02, 0, 0);


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res5 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res6 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    x0 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);

    res1 = res1 + res6;
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);

    res0 = res0 + res5;
    x1 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    x2 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx00, 0, 0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx10, 0, 1);
    res7 = m_pop_mrf(1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //19
    res3 = res3 + res5;
    x3 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    //20
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    res3 = res3 + res7;
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx11, 0, 1);

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    m_matmul_single(pgx02, 0, 0);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    m_matmul_single(pgx12, 0, 1);
    res5 = m_pop_mrf(1);

    //23
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res6 = m_pop_mrf(0);

    //24
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res7 = m_pop_mrf(1);


    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;

    res3 = res3 + res7;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    x0 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    x1 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);

    //第四个[32, 128] * [128, 128]
    //25
    x2 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    m_matmul_single(pgx00, 0, 0);

    //26
    m_matmul_single(pgx10, 0, 1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);

    //27
    m_matmul_single(pgx01, 0, 0);
    x3 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);

    //28
    m_matmul_single(pgx11, 0, 1);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    m_matmul_single(pgx12, 0, 1);
    //31
    m_matmul_single(pgx03, 0, 0);
    //32
    m_matmul_single(pgx13, 0, 1);

    C += 4 * 128;
    D += 4 * 128;

    res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    res5 = m_pop_mrf(1);

    res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res3 = res3 + res5;

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    res3 = res3 + res7;

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    res5 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;

    //7
    res1 = res1 + res6;

    //8
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    //13
    res3 = res3 + res7;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }

    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    res5 = m_pop_mrf(1);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //19
    res3 = res3 + res5;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    //20
    res3 = res3 + res7;
    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    res5 = m_pop_mrf(1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    res6 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    res7 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    res0 = res0 + res4;
    //26
    res1 = res1 + res6;

    //27
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //28
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //29
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //30
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //31
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    //32
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }

    res3 = res3 + res7;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);
}
//matmul when ah = 256, full loop unroll, 2pgx, iaw == 0.
//it performs the last 16 matmuls, and pop the results.
inline void matmul_2pgx_256_pop_first_bf16(SIM_X86::tensor A0, SIM_X86::tensor C) {
    float8_128 x0 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);

    float8_128 x1 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    short8_128 pgx10_h = unpack_16b(as_int(x0), 1);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);
    float8_128 pgx10 = bfloat16_to_float(pgx10_h);

    float8_128 x2 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    short8_128 pgx11_h = unpack_16b(as_int(x1), 1);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);
    float8_128 pgx11 = bfloat16_to_float(pgx11_h);
    float8_128 x3 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);

    float8_128 res0 = m_pop_mrf(0);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    short8_128 pgx12_h = unpack_16b(as_int(x2), 1);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);
    float8_128 pgx12 = bfloat16_to_float(pgx12_h);

    float8_128 res1 = m_pop_mrf(1);
    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    short8_128 pgx13_h = unpack_16b(as_int(x3), 1);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    float8_128 pgx13 = bfloat16_to_float(pgx13_h);
    //第一个[32, 128] * [128, 128]
    //1
    m_matmul_single(pgx00, 0, 0);
    float8_128 res2 = m_pop_mrf(0);

    //2
    x0 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    float8_128 res3 = m_pop_mrf(1);

    //3
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx01, 0, 0);
    float8_128 res4 = m_pop_mrf(0);

    //4
    x1 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    float8_128 res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    m_matmul_single(pgx02, 0, 0);
    float8_128 res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    x2 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    float8_128 res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);

    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    x3 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    x0 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //11
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx01, 0, 0);
    res4 = m_pop_mrf(0);

    //12
    x1 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    res5 = m_pop_mrf(1);

    //13
    res0 = res0 + res1;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;
    x2 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    x3 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //19
    m_matmul_single(pgx01, 0, 0);
    res4 = m_pop_mrf(0);

    //20
    m_matmul_single(pgx11, 0, 1);
    res5 = m_pop_mrf(1);

    //21
    res0 = res0 + res1;
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    res4 = m_pop_mrf(0);

    res5 = m_pop_mrf(1);

    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
    x0 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);
    push_gsnf(pgx10, 1);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    push_gsnf(pgx01, 0);

    pgx02 = as_float(v_u32_shl(as_int(pgx02), v_u32_move_i(16)));
    x1 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);
    push_gsnf(pgx11, 1);

    pgx12 = as_float(v_u32_shl(as_int(pgx12), v_u32_move_i(16)));
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    push_gsnf(pgx02, 0);

    pgx03 = as_float(v_u32_shl(as_int(pgx03), v_u32_move_i(16)));
    x2 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    push_gsnf(pgx12, 1);

    pgx13 = as_float(v_u32_shl(as_int(pgx13), v_u32_move_i(16)));
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    push_gsnf(pgx03, 0);

    x3 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);
    push_gsnf(pgx13, 1);

    //第四个[32, 128] * [128, 128]
    //25
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);

    m_matmul_single(pgx00, 0, 0);
    //26
    m_matmul_single(pgx10, 0, 1);
    //27
    m_matmul_single(pgx01, 0, 0);
    //28
    m_matmul_single(pgx11, 0, 1);
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    m_matmul_single(pgx12, 0, 1);
    //31
    m_matmul_single(pgx03, 0, 0);
    //32
    m_matmul_single(pgx13, 0, 1);

    C += 4 * 128;

    res0 = m_pop_mrf(0);

    res1 = m_pop_mrf(1);

    res2 = m_pop_mrf(0);

    res3 = m_pop_mrf(1);


    res4 = m_pop_mrf(0);


    res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    res1 = m_pop_mrf(1);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    res4 = m_pop_mrf(0);

    //12
    res5 = m_pop_mrf(1);

    //13
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    res1 = m_pop_mrf(1);


    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //19
    res4 = m_pop_mrf(0);

    //20
    res5 = m_pop_mrf(1);

    //21
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    res1 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //26
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //27
    res4 = m_pop_mrf(0);

    //28
    res5 = m_pop_mrf(1);

    //29
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //30
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //31
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    //32
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
}
//matmul when ah = 256, full loop unroll, 2pgx.
//it performs the last 16 matmuls, and pop the results.
inline void matmul_2pgx_256_pop_bf16(SIM_X86::tensor A0, SIM_X86::tensor C) {
    float8_128 res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    float8_128 res4 = m_pop_mrf(0);

    float8_128 res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    float8_128 res5 = m_pop_mrf(1);

    float8_128 res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    float8_128 res6 = m_pop_mrf(0);

    float8_128 res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    float8_128 res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    float8_128 x0 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);

    res1 = res1 + res6;
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    short8_128 pgx10_h = unpack_16b(as_int(x0), 1);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);
    float8_128 pgx10 = bfloat16_to_float(pgx10_h);

    res0 = res0 + res5;
    float8_128 x1 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    short8_128 pgx11_h = unpack_16b(as_int(x1), 1);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);
    float8_128 pgx11 = bfloat16_to_float(pgx11_h);
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;

    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    float8_128 x2 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    res5 = m_pop_mrf(0);

    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    short8_128 pgx12_h = unpack_16b(as_int(x2), 1);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);
    float8_128 pgx12 = bfloat16_to_float(pgx12_h);
    res7 = m_pop_mrf(1);

    res3 = res3 + res5;
    float8_128 x3 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);

    res3 = res3 + res7;
    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    short8_128 pgx13_h = unpack_16b(as_int(x3), 1);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    float8_128 pgx13 = bfloat16_to_float(pgx13_h);

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);
    m_matmul_single(pgx00, 0, 0);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    m_matmul_single(pgx01, 0, 0);
    res5 = m_pop_mrf(1);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    m_matmul_single(pgx02, 0, 0);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;
    x0 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    //7
    res1 = res1 + res6;
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);
    m_matmul_single(pgx03, 0, 0);

    //8
    res0 = res0 + res5;
    x1 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);

    m_matmul_single(pgx00, 0, 0);
    res6 = m_pop_mrf(1);

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    x2 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res5 = m_pop_mrf(0);


    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx01, 0, 0);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;
    x3 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    //13
    res3 = res3 + res7;
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx02, 0, 0);


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res5 = m_pop_mrf(1);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    x0 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);

    res1 = res1 + res6;
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);

    res0 = res0 + res5;
    x1 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    x2 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx00, 0, 0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);
    m_matmul_single(pgx10, 0, 1);
    res7 = m_pop_mrf(1);

    //19
    res3 = res3 + res5;
    x3 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    //20
    res3 = res3 + res7;
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    m_matmul_single(pgx11, 0, 1);

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    m_matmul_single(pgx02, 0, 0);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    m_matmul_single(pgx12, 0, 1);
    res5 = m_pop_mrf(1);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res7 = m_pop_mrf(1);


    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;

    res3 = res3 + res7;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);


    x0 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx10_h = unpack_16b(as_int(x0), 1);
    pgx00 = bfloat16_to_float(pgx00_h);
    pgx10 = bfloat16_to_float(pgx10_h);

    x1 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx11_h = unpack_16b(as_int(x1), 1);
    pgx01 = bfloat16_to_float(pgx01_h);
    pgx11 = bfloat16_to_float(pgx11_h);

    //第四个[32, 128] * [128, 128]
    //25
    x2 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    m_matmul_single(pgx00, 0, 0);

    //26
    m_matmul_single(pgx10, 0, 1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx12_h = unpack_16b(as_int(x2), 1);
    pgx02 = bfloat16_to_float(pgx02_h);
    pgx12 = bfloat16_to_float(pgx12_h);

    //27
    m_matmul_single(pgx01, 0, 0);
    x3 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);

    //28
    m_matmul_single(pgx11, 0, 1);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx13_h = unpack_16b(as_int(x3), 1);
    pgx03 = bfloat16_to_float(pgx03_h);
    pgx13 = bfloat16_to_float(pgx13_h);
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    m_matmul_single(pgx12, 0, 1);
    //31
    m_matmul_single(pgx03, 0, 0);
    //32
    m_matmul_single(pgx13, 0, 1);

    C += 4 * 128;

    res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    res5 = m_pop_mrf(1);

    res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    res3 = res3 + res5;

    res3 = res3 + res7;

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    res5 = m_pop_mrf(1);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;

    //7
    res1 = res1 + res6;

    //8
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;

    //13
    res3 = res3 + res7;


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    res5 = m_pop_mrf(1);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //19
    res3 = res3 + res5;

    //20
    res3 = res3 + res7;
    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    res5 = m_pop_mrf(1);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    res7 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    res0 = res0 + res4;
    //26
    res1 = res1 + res6;

    //27
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //28
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //29
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //30
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //31
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    //32
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;

    res3 = res3 + res7;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);
}
//matmul when ah = 256, full loop unroll.
//it performs the upper 16 matmuls.
inline void matmul_256_pre16_bf16(SIM_X86::tensor A0, int cur_aw) {
    float8_128 x0 = load8_k(A0 + 0, 1, 255, cur_aw, 0);
    float8_128 x1 = load8_k(A0 + 32, 1, 255, cur_aw, 0);

    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);

    float8_128 x2 = load8_k(A0 + 64, 1, 255, cur_aw, 0);
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);

    float8_128 x3 = load8_k(A0 + 96, 1, 255, cur_aw, 0);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);

    //第一部分的[64, 128] * [128, 128]
    m_matmul_single(pgx00, 0, 0);
    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    x0 = load8_k(A0 + 128, 1, 255, cur_aw, 0);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx01, 0, 0);

    x1 = load8_k(A0 + 160, 1, 255, cur_aw, 0);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    m_matmul_single(pgx02, 0, 0);

    x2 = load8_k(A0 + 192, 1, 255, cur_aw, 0);

    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);
    m_matmul_single(pgx03, 0, 0);

    x3 = load8_k(A0 + 224, 1, 255, cur_aw, 0);

    //第二部分的[64, 128] * [128, 128]
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx00, 0, 0);

    x0 = load8_k(A0 + 256, 1, 255, cur_aw, 0);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx01, 0, 0);

    x1 = load8_k(A0 + 288, 1, 255, cur_aw, 0);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    m_matmul_single(pgx02, 0, 0);

    x2 = load8_k(A0 + 320, 1, 255, cur_aw, 0);

    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);
    m_matmul_single(pgx03, 0, 0);

    x3 = load8_k(A0 + 352, 1, 255, cur_aw, 0);

    //第三部分的[64, 128] * [128, 128]
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx00, 0, 0);

    x0 = load8_k(A0 + 384, 1, 255, cur_aw, 0);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx01, 0, 0);

    x1 = load8_k(A0 + 416, 1, 255, cur_aw, 0);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    m_matmul_single(pgx02, 0, 0);

    x2 = load8_k(A0 + 448, 1, 255, cur_aw, 0);

    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);
    m_matmul_single(pgx03, 0, 0);

    x3 = load8_k(A0 + 480, 1, 255, cur_aw, 0);

    //第三部分的[64, 128] * [128, 128]
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx00, 0, 0);

    m_matmul_single(pgx01, 0, 0);

    m_matmul_single(pgx02, 0, 0);

    m_matmul_single(pgx03, 0, 0);
}
//matmul when ah = 256, full loop unroll, iaw == 0.
//it performs the last 16 matmuls, and pop the results, converting results to bf16 and store it on SIM_X86::tensor D at the same time.
inline void matmul_256_pop_first_store_bf16(SIM_X86::tensor A0, SIM_X86::tensor C, SIM_X86::tensor D, int ibw, int ah, int bw128, int cur_aw) {
    float8_128 x0 = load8_k(A0 + 0, 1, 255, cur_aw, 0);

    float8_128 x1 = load8_k(A0 + 32, 1, 255, cur_aw, 0);
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);

    float8_128 x2 = load8_k(A0 + 64, 1, 255, cur_aw, 0);
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);
    float8_128 x3 = load8_k(A0 + 96, 1, 255, cur_aw, 0);

    float8_128 res0 = m_pop_mrf(0);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);

    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    //第一个[32, 128] * [128, 128]
    //1
    m_matmul_single(pgx00, 0, 0);
    float8_128 res2 = m_pop_mrf(0);

    //2
    x0 = load8_k(A0 + 128, 1, 255, cur_aw, 0);

    //3
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx01, 0, 0);
    float8_128 res4 = m_pop_mrf(0);

    //4
    x1 = load8_k(A0 + 160, 1, 255, cur_aw, 0);

    //5
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    m_matmul_single(pgx02, 0, 0);
    float8_128 res6 = m_pop_mrf(0);

    //6
    x2 = load8_k(A0 + 192, 1, 255, cur_aw, 0);

    //7
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //8
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    x3 = load8_k(A0 + 224, 1, 255, cur_aw, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    x0 = load8_k(A0 + 256, 1, 255, cur_aw, 0);

    //11
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx01, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    //12
    x1 = load8_k(A0 + 288, 1, 255, cur_aw, 0);

    //13
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    m_matmul_single(pgx02, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    res6 = m_pop_mrf(0);

    //14
    x2 = load8_k(A0 + 320, 1, 255, cur_aw, 0);

    //15
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);
    m_matmul_single(pgx03, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res0 = m_pop_mrf(0);

    //16
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    x3 = load8_k(A0 + 352, 1, 255, cur_aw, 0);

    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx00, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);

    //19
    m_matmul_single(pgx01, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    //20
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }

    //21
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //23
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    m_matmul_single(pgx03, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res0 = m_pop_mrf(0);

    //24
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);

    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
    x0 = load8_k(A0 + 384, 1, 255, cur_aw, 0);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }

    x1 = load8_k(A0 + 416, 1, 255, cur_aw, 0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    x2 = load8_k(A0 + 448, 1, 255, cur_aw, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);

    x3 = load8_k(A0 + 480, 1, 255, cur_aw, 0);

    //第四个[32, 128] * [128, 128]
    //25
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);

    m_matmul_single(pgx00, 0, 0);

    m_matmul_single(pgx01, 0, 0);

    m_matmul_single(pgx02, 0, 0);

    m_matmul_single(pgx03, 0, 0);

    C += 4 * 128;
    D += 4 * 128;
    res0 = m_pop_mrf(0);

    res2 = m_pop_mrf(0);

    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res0 = m_pop_mrf(0);

    //8
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    res4 = m_pop_mrf(0);

    //12
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    //13
    res6 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //15
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //16
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);


    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    //19
    res4 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    //21
    res6 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }

    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }

    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    //27
    res4 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    //29
    res6 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //31
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //32
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res4, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res4)));
    }
    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res6, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res6)));
    }
    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
}
//matmul when ah = 256, full loop unroll.
//it performs the last 16 matmuls, and pop the results, converting results to bf16 and store it on SIM_X86::tensor D at the same time.
inline void matmul_256_pop_store_bf16(SIM_X86::tensor A0, SIM_X86::tensor C, SIM_X86::tensor D, int ibw, int ah, int bw128, int cur_aw) {
    float8_128 res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    float8_128 res4 = m_pop_mrf(0);

    float8_128 res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);

    float8_128 res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    float8_128 res6 = m_pop_mrf(0);

    float8_128 res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);

    res0 = res0 + res4;
    float8_128 x0 = load8_k(A0 + 0, 1, 255, cur_aw, 0);

    res1 = res1 + res6;
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);

    float8_128 x1 = load8_k(A0 + 32, 1, 255, cur_aw, 0);
    res4 = m_pop_mrf(0);

    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);

    res2 = res2 + res4;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    float8_128 x2 = load8_k(A0 + 64, 1, 255, cur_aw, 0);
    float8_128 res5 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);

    res3 = res3 + res5;
    float8_128 x3 = load8_k(A0 + 96, 1, 255, cur_aw, 0);

    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);

    //第一个[32, 128] * [128, 128]
    //1

    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);
    m_matmul_single(pgx00, 0, 0);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    //4
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    m_matmul_single(pgx02, 0, 0);

    //6
    res0 = res0 + res4;
    x0 = load8_k(A0 + 128, 1, 255, cur_aw, 0);

    //7
    res1 = res1 + res6;
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx03, 0, 0);

    //8
    x1 = load8_k(A0 + 160, 1, 255, cur_aw, 0);
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);

    m_matmul_single(pgx00, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    x2 = load8_k(A0 + 192, 1, 255, cur_aw, 0);
    res5 = m_pop_mrf(0);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    //11
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);
    m_matmul_single(pgx01, 0, 0);

    //12
    res3 = res3 + res5;
    x3 = load8_k(A0 + 224, 1, 255, cur_aw, 0);

    //13
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx02, 0, 0);


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);

    res0 = res0 + res4;
    x0 = load8_k(A0 + 256, 1, 255, cur_aw, 0);

    res1 = res1 + res6;
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);

    x1 = load8_k(A0 + 288, 1, 255, cur_aw, 0);
    res4 = m_pop_mrf(0);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    x2 = load8_k(A0 + 320, 1, 255, cur_aw, 0);
    m_matmul_single(pgx00, 0, 0);
    res5 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //19
    res3 = res3 + res5;
    x3 = load8_k(A0 + 352, 1, 255, cur_aw, 0);
    m_matmul_single(pgx01, 0, 0);

    //20
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    m_matmul_single(pgx02, 0, 0);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);

    //23
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res6 = m_pop_mrf(0);

    //24
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res4 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res3 = res3 + res5;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    x0 = load8_k(A0 + 384, 1, 255, cur_aw, 0);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    x1 = load8_k(A0 + 416, 1, 255, cur_aw, 0);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);

    //第四个[32, 128] * [128, 128]
    //25
    x2 = load8_k(A0 + 448, 1, 255, cur_aw, 0);
    m_matmul_single(pgx00, 0, 0);

    //26
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);

    //27
    m_matmul_single(pgx01, 0, 0);
    x3 = load8_k(A0 + 480, 1, 255, cur_aw, 0);

    //28
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    m_matmul_single(pgx03, 0, 0);

    C += 4 * 128;
    D += 4 * 128;

    res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);

    res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res4 = m_pop_mrf(0);


    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(0, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(0, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res3 = res3 + res5;

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(32, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(32, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(64, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(64, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(96, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(96, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);

    //6
    res0 = res0 + res4;

    //7
    res1 = res1 + res6;

    //8
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(128, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(128, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    //11
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);

    //12
    res3 = res3 + res5;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(160, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(160, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    //13
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(192, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(192, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }

    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(224, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(224, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res4 = m_pop_mrf(0);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(256, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(256, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    //19
    res3 = res3 + res5;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(288, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(288, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(320, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(320, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }
    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    res6 = m_pop_mrf(0);
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(352, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(352, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }
    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);

    //第四个[32, 128] * [128, 128]
    //25
    res0 = res0 + res4;
    //26
    res1 = res1 + res6;

    //27
    res4 = m_pop_mrf(0);

    //29
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //30
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);

    //31
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(384, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(384, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
    }
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    //32
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(416, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(res1, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(416, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res1)));
    }
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res3 = res3 + res5;
    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(448, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(res2, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(448, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res2)));
    }

    if (ibw % 256) {
        float8_128 res0_lo = load8_128_stride_with_ldmask(480, 1, 255, C - ah * 4);
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(res3, res0_lo)));
    }
    else if (ibw == bw128 - 128) {
        store8_128_stride_with_stmask(480, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res3)));
    }

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);
}
//matmul when ah = 256, full loop unroll, iaw == 0.
//it performs the last 16 matmuls, and pop the results.
inline void matmul_256_pop_first_bf16(SIM_X86::tensor A0, SIM_X86::tensor C, int cur_aw) {
    float8_128 x0 = load8_k(A0 + 0, 1, 255, cur_aw, 0);

    float8_128 x1 = load8_k(A0 + 32, 1, 255, cur_aw, 0);
    short8_128 pgx00_h = unpack_16b(as_int(x0), 0);
    float8_128 pgx00 = bfloat16_to_float(pgx00_h);

    float8_128 x2 = load8_k(A0 + 64, 1, 255, cur_aw, 0);
    short8_128 pgx01_h = unpack_16b(as_int(x1), 0);
    float8_128 pgx01 = bfloat16_to_float(pgx01_h);
    float8_128 x3 = load8_k(A0 + 96, 1, 255, cur_aw, 0);

    float8_128 res0 = m_pop_mrf(0);
    short8_128 pgx02_h = unpack_16b(as_int(x2), 0);
    float8_128 pgx02 = bfloat16_to_float(pgx02_h);

    short8_128 pgx03_h = unpack_16b(as_int(x3), 0);
    float8_128 pgx03 = bfloat16_to_float(pgx03_h);
    //第一个[32, 128] * [128, 128]
    //1
    m_matmul_single(pgx00, 0, 0);
    float8_128 res2 = m_pop_mrf(0);

    //2
    x0 = load8_k(A0 + 128, 1, 255, cur_aw, 0);

    //3
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx01, 0, 0);
    float8_128 res4 = m_pop_mrf(0);

    //4
    x1 = load8_k(A0 + 160, 1, 255, cur_aw, 0);

    //5
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    m_matmul_single(pgx02, 0, 0);
    float8_128 res6 = m_pop_mrf(0);

    //6
    x2 = load8_k(A0 + 192, 1, 255, cur_aw, 0);

    //7
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);

    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //8
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    x3 = load8_k(A0 + 224, 1, 255, cur_aw, 0);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    x0 = load8_k(A0 + 256, 1, 255, cur_aw, 0);

    //11
    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);
    m_matmul_single(pgx01, 0, 0);

    res4 = m_pop_mrf(0);

    //12
    x1 = load8_k(A0 + 288, 1, 255, cur_aw, 0);
    //13
    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);
    m_matmul_single(pgx02, 0, 0);

    res6 = m_pop_mrf(0);

    //14
    x2 = load8_k(A0 + 320, 1, 255, cur_aw, 0);

    //15
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);
    m_matmul_single(pgx03, 0, 0);

    res0 = m_pop_mrf(0);

    //16
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    x3 = load8_k(A0 + 352, 1, 255, cur_aw, 0);

    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);
    m_matmul_single(pgx00, 0, 0);

    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);

    //19
    m_matmul_single(pgx01, 0, 0);

    res4 = m_pop_mrf(0);

    //21
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //23
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    m_matmul_single(pgx03, 0, 0);

    res0 = m_pop_mrf(0);

    //24
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);

    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);

    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
    x0 = load8_k(A0 + 384, 1, 255, cur_aw, 0);

    pgx00_h = unpack_16b(as_int(x0), 0);
    pgx00 = bfloat16_to_float(pgx00_h);

    x1 = load8_k(A0 + 416, 1, 255, cur_aw, 0);

    pgx01_h = unpack_16b(as_int(x1), 0);
    pgx01 = bfloat16_to_float(pgx01_h);

    x2 = load8_k(A0 + 448, 1, 255, cur_aw, 0);

    pgx02_h = unpack_16b(as_int(x2), 0);
    pgx02 = bfloat16_to_float(pgx02_h);

    x3 = load8_k(A0 + 480, 1, 255, cur_aw, 0);

    //第四个[32, 128] * [128, 128]
    //25
    pgx03_h = unpack_16b(as_int(x3), 0);
    pgx03 = bfloat16_to_float(pgx03_h);

    m_matmul_single(pgx00, 0, 0);

    m_matmul_single(pgx01, 0, 0);

    m_matmul_single(pgx02, 0, 0);

    m_matmul_single(pgx03, 0, 0);

    C += 4 * 128;

    res0 = m_pop_mrf(0);

    res2 = m_pop_mrf(0);

    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);

    res0 = m_pop_mrf(0);

    //8
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);

    res4 = m_pop_mrf(0);

    //13
    res6 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //16
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);


    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);

    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);

    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
}

inline void matmul_gain_2pgx_no_pack_opt_store_rest_bf16(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int cur_ah, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag, float8_128 scale) {

    int n = (cur_ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int bw128 = ALIGN128(bw);
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_128_stride(AoffsetPgx0, 1, A);
        short8_128 x1 = unpack_16b(as_int(left0), 0);
        short8_128 x2 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
        m_matmul_single(bfloat16_to_float(x2) * scale, 0, 1);
    }

#pragma unroll
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);

        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_128_stride(AoffsetPgx0, 1, A);
        short8_128 x1 = unpack_16b(as_int(left0), 0);
        short8_128 x2 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
        m_matmul_single(bfloat16_to_float(x2) * scale, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, 1, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res)));
        }
    }

#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, 1, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res)));
        }
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(cur_ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, 1, mask, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(res, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(v_u32_move_f(0), res)));
        }
    }
}

inline void matmul_gain_2pgx_no_pack_opt_rest_bf16(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int cur_ah, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag, float8_128 scale) {
    int n = (cur_ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_128_stride(AoffsetPgx0, 1, A);
        short8_128 x1 = unpack_16b(as_int(left0), 0);
        short8_128 x2 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
        m_matmul_single(bfloat16_to_float(x2) * scale, 0, 1);
    }

#pragma unroll
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);

        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_128_stride(AoffsetPgx0, 1, A);
        short8_128 x1 = unpack_16b(as_int(left0), 0);
        short8_128 x2 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
        m_matmul_single(bfloat16_to_float(x2) * scale, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }

#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(cur_ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);
    }
}

//默认gain里已有数据，只做高度为ah，aw为256的matmul。使用经过完全循环展开和指令排布的矩阵乘法，性能较好，指令数较多
//左矩阵和输出均为bf16,C存放f32格式的结果，D存放bf16格式的结果。
inline void matmul_LHS_aw256_pipeline_bf16(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag) {
    int bw128 = ALIGN128(bw);
    int ah256 = ah & 0xFFFFFF00;
    int iah = 0;
    for (; iah < ah256; iah += 256) {
        matmul_2pgx_256_pre16_bf16(A + ah * bf_iaw / 32 + iah * 4);
        if (iaw == aw - 256) {
            if (iaw == 0 && add_src_flag == 0) matmul_2pgx_256_pop_first_store_bf16(A + ah * bf_iaw / 32 + 4 * 128 + iah * 4, C + iah * 4, D + iah * 4, ibw, ah, bw128);
            else matmul_2pgx_256_pop_store_bf16(A + ah * bf_iaw / 32 + 4 * 128 + iah * 4, C + iah * 4, D + iah * 4, ibw, ah, bw128);
        }
        else {
            if (iaw == 0 && add_src_flag == 0) matmul_2pgx_256_pop_first_bf16(A + ah * bf_iaw / 32 + 4 * 128 + iah * 4, C + iah * 4);
            else matmul_2pgx_256_pop_bf16(A + ah * bf_iaw / 32 + 4 * 128 + iah * 4, C + iah * 4);
        }
    }

    if (iah < ah) {
        if (iaw == aw - 256)
            matmul_gain_2pgx_no_pack_opt_store_rest_bf16(A + iah * 4, C + iah * 4, D + iah * 4, ah - iah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, 1.0);
        else
            matmul_gain_2pgx_no_pack_opt_rest_bf16(A + iah * 4, C + iah * 4, D + iah * 4, ah - iah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, 1.0);
    }
}
//默认gain里已有数据，只做高度为ah，aw为256的matmul。
//左矩阵和输出均为bf16，C存放f32格式的结果，D存放bf16格式的结果。
inline void matmul_LHS_aw256_bf16(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag, int is_last_aw, float8_128 scale) {
    if (is_last_aw && iaw + 256 >= aw)
        matmul_gain_2pgx_no_pack_opt_store_rest_bf16(A, C, D, ah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, scale);
    else
        matmul_gain_2pgx_no_pack_opt_rest_bf16(A, C, D, ah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, scale);
}

//默认gain里已有数据，只做高度为ah，aw为256的matmul。使用经过完全循环展开和指令排布的矩阵乘法，性能较好，指令数较多
//左矩阵为bf16,输出为f32,C存放f32格式的结果
inline void matmul_LHS_aw256_pipeline_bf16_out_f32(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag) {
    int ah256 = ah & 0xFFFFFF00;
    int iah = 0;
    for (; iah < ah256; iah += 256) {
        matmul_2pgx_256_pre16_bf16(A + ah * bf_iaw / 32 + iah * 4);
        if (iaw == 0 && add_src_flag == 0) matmul_2pgx_256_pop_first_bf16(A + ah * bf_iaw / 32 + 4 * 128 + iah * 4, C + iah * 4);
        else matmul_2pgx_256_pop_bf16(A + ah * bf_iaw / 32 + 4 * 128 + iah * 4, C + iah * 4);
    }

    if (iah < ah) {
        matmul_gain_2pgx_no_pack_opt_rest_bf16(A + iah * 4, C + iah * 4, C + iah * 4, ah - iah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, 1.0);
    }
}

//默认gain里已有数据，只做高度为ah，aw为256的matmul。
//左矩阵为bf16,输出为f32,C存放f32格式的结果
inline void matmul_LHS_aw256_bf16_out_f32(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag, float8_128 scale) {
    matmul_gain_2pgx_no_pack_opt_rest_bf16(A, C, C, ah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, scale);
}

inline void matmul_gain_no_pack_opt_store_bf16(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int bf_ibw, int add_src_flag, float8_128 scale) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int bw128 = ALIGN128(bw);
    int is_hi = (iaw / 128) % 2;
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        short8_128 x1;
        if (is_hi == 0) x1 = unpack_16b(as_int(left0), 0);
        else x1 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
    }
#pragma unroll
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        short8_128 x1;
        if (is_hi == 0) x1 = unpack_16b(as_int(left0), 0);
        else x1 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);

        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
        }
    }
#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
        }
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, mask);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(res0, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
        }
    }
}

inline void matmul_gain_no_pack_opt_bf16(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int bf_ibw, int add_src_flag, float8_128 scale) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int is_hi = (iaw / 128) % 2;
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        short8_128 x1;
        if (is_hi == 0) x1 = unpack_16b(as_int(left0), 0);
        else x1 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
    }
#pragma unroll
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        short8_128 x1;
        if (is_hi == 0) x1 = unpack_16b(as_int(left0), 0);
        else x1 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);

        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
    }
#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, mask);
    }
}

inline void matmul_2pgx_256_pre16(SIM_X86::tensor A0, SIM_X86::tensor A1) {
    float8_128 pgx00 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);
    float8_128 pgx10 = v_f32_ld_tnsr_st_msk(0, A1, 1, 255);
    float8_128 pgx01 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    float8_128 pgx11 = v_f32_ld_tnsr_st_msk(32, A1, 1, 255);
    float8_128 pgx02 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    float8_128 pgx12 = v_f32_ld_tnsr_st_msk(64, A1, 1, 255);
    float8_128 pgx03 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);

    //第一部分的[64, 128] * [128, 128]
    float8_128 pgx13 = v_f32_ld_tnsr_st_msk(96, A1, 1, 255);
    m_matmul_single(pgx00, 0, 0);

    pgx00 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);

    pgx10 = v_f32_ld_tnsr_st_msk(128, A1, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    pgx01 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    pgx11 = v_f32_ld_tnsr_st_msk(160, A1, 1, 255);
    m_matmul_single(pgx02, 0, 0);

    pgx02 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    pgx12 = v_f32_ld_tnsr_st_msk(192, A1, 1, 255);
    m_matmul_single(pgx03, 0, 0);

    pgx03 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);

    //第二部分的[64, 128] * [128, 128]
    pgx13 = v_f32_ld_tnsr_st_msk(224, A1, 1, 255);
    m_matmul_single(pgx00, 0, 0);

    pgx00 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);

    pgx10 = v_f32_ld_tnsr_st_msk(256, A1, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    pgx01 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    pgx11 = v_f32_ld_tnsr_st_msk(288, A1, 1, 255);
    m_matmul_single(pgx02, 0, 0);

    pgx02 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    pgx12 = v_f32_ld_tnsr_st_msk(320, A1, 1, 255);
    m_matmul_single(pgx03, 0, 0);

    pgx03 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);

    //第三部分的[64, 128] * [128, 128]
    pgx13 = v_f32_ld_tnsr_st_msk(352, A1, 1, 255);
    m_matmul_single(pgx00, 0, 0);

    pgx00 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);

    pgx10 = v_f32_ld_tnsr_st_msk(384, A1, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    pgx01 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    pgx11 = v_f32_ld_tnsr_st_msk(416, A1, 1, 255);
    m_matmul_single(pgx02, 0, 0);

    pgx02 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    pgx12 = v_f32_ld_tnsr_st_msk(448, A1, 1, 255);
    m_matmul_single(pgx03, 0, 0);

    pgx03 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);

    //第三部分的[64, 128] * [128, 128]
    pgx13 = v_f32_ld_tnsr_st_msk(480, A1, 1, 255);
    m_matmul_single(pgx00, 0, 0);

    m_matmul_single(pgx10, 0, 1);

    m_matmul_single(pgx01, 0, 0);

    m_matmul_single(pgx11, 0, 1);

    m_matmul_single(pgx02, 0, 0);

    m_matmul_single(pgx12, 0, 1);

    m_matmul_single(pgx03, 0, 0);

    m_matmul_single(pgx13, 0, 1);

}

inline void matmul_2pgx_256_pre16_stride(SIM_X86::tensor A0, SIM_X86::tensor A1, int stride) {
    float8_128 pgx00 = load8_128_stride(0, stride, A0);
    float8_128 pgx10 = load8_128_stride(0, stride, A1);
    float8_128 pgx01 = load8_128_stride(stride * 32, stride, A0);
    float8_128 pgx11 = load8_128_stride(stride * 32, stride, A1);
    float8_128 pgx02 = load8_128_stride(stride * 64, stride, A0);
    float8_128 pgx12 = load8_128_stride(stride * 64, stride, A1);
    float8_128 pgx03 = load8_128_stride(stride * 96, stride, A0);

    //第一部分的[64, 128] * [128, 128]
    float8_128 pgx13 = load8_128_stride(stride * 96, stride, A1);
    m_matmul_single(pgx00, 0, 0);

    pgx00 = load8_128_stride(stride * 128, stride, A0);
    m_matmul_single(pgx10, 0, 1);

    pgx10 = load8_128_stride(stride * 128, stride, A1);
    m_matmul_single(pgx01, 0, 0);

    pgx01 = load8_128_stride(stride * 160, stride, A0);
    m_matmul_single(pgx11, 0, 1);

    pgx11 = load8_128_stride(stride * 160, stride, A1);
    m_matmul_single(pgx02, 0, 0);

    pgx02 = load8_128_stride(stride * 192, stride, A0);
    m_matmul_single(pgx12, 0, 1);

    pgx12 = load8_128_stride(stride * 192, stride, A1);
    m_matmul_single(pgx03, 0, 0);

    pgx03 = load8_128_stride(stride * 224, stride, A0);
    m_matmul_single(pgx13, 0, 1);

    //第二部分的[64, 128] * [128, 128]
    pgx13 = load8_128_stride(stride * 224, stride, A1);
    m_matmul_single(pgx00, 0, 0);

    pgx00 = load8_128_stride(stride * 256, stride, A0);
    m_matmul_single(pgx10, 0, 1);

    pgx10 = load8_128_stride(stride * 256, stride, A1);
    m_matmul_single(pgx01, 0, 0);

    pgx01 = load8_128_stride(stride * 288, stride, A0);
    m_matmul_single(pgx11, 0, 1);

    pgx11 = load8_128_stride(stride * 288, stride, A1);
    m_matmul_single(pgx02, 0, 0);

    pgx02 = load8_128_stride(stride * 320, stride, A0);
    m_matmul_single(pgx12, 0, 1);

    pgx12 = load8_128_stride(stride * 320, stride, A1);
    m_matmul_single(pgx03, 0, 0);

    pgx03 = load8_128_stride(stride * 352, stride, A0);
    m_matmul_single(pgx13, 0, 1);

    //第三部分的[64, 128] * [128, 128]
    pgx13 = load8_128_stride(stride * 352, stride, A1);
    m_matmul_single(pgx00, 0, 0);

    pgx00 = load8_128_stride(stride * 384, stride, A0);
    m_matmul_single(pgx10, 0, 1);

    pgx10 = load8_128_stride(stride * 384, stride, A1);
    m_matmul_single(pgx01, 0, 0);

    pgx01 = load8_128_stride(stride * 416, stride, A0);
    m_matmul_single(pgx11, 0, 1);

    pgx11 = load8_128_stride(stride * 416, stride, A1);
    m_matmul_single(pgx02, 0, 0);

    pgx02 = load8_128_stride(stride * 448, stride, A0);
    m_matmul_single(pgx12, 0, 1);

    pgx12 = load8_128_stride(stride * 448, stride, A1);
    m_matmul_single(pgx03, 0, 0);

    pgx03 = load8_128_stride(stride * 480, stride, A0);
    m_matmul_single(pgx13, 0, 1);

    //第三部分的[64, 128] * [128, 128]
    pgx13 = load8_128_stride(stride * 480, stride, A1);
    m_matmul_single(pgx00, 0, 0);

    m_matmul_single(pgx10, 0, 1);

    m_matmul_single(pgx01, 0, 0);

    m_matmul_single(pgx11, 0, 1);

    m_matmul_single(pgx02, 0, 0);

    m_matmul_single(pgx12, 0, 1);

    m_matmul_single(pgx03, 0, 0);

    m_matmul_single(pgx13, 0, 1);
}

inline void matmul_2pgx_256_pop_first(SIM_X86::tensor A0, SIM_X86::tensor A1, SIM_X86::tensor C) {
    float8_128 pgx00 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);
    float8_128 pgx10 = v_f32_ld_tnsr_st_msk(0, A1, 1, 255);
    float8_128 pgx01 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    float8_128 pgx11 = v_f32_ld_tnsr_st_msk(32, A1, 1, 255);
    float8_128 pgx02 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    float8_128 pgx12 = v_f32_ld_tnsr_st_msk(64, A1, 1, 255);
    float8_128 pgx03 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);
    float8_128 res0 = m_pop_mrf(0);
    float8_128 pgx13 = v_f32_ld_tnsr_st_msk(96, A1, 1, 255);
    float8_128 res1 = m_pop_mrf(1);

    //第一个[32, 128] * [128, 128]
    //1
    m_matmul_single(pgx00, 0, 0);
    float8_128 res2 = m_pop_mrf(0);

    //2
    pgx00 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    float8_128 res3 = m_pop_mrf(1);

    //3
    pgx10 = v_f32_ld_tnsr_st_msk(128, A1, 1, 255);
    m_matmul_single(pgx01, 0, 0);
    float8_128 res4 = m_pop_mrf(0);

    //4
    pgx01 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    float8_128 res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    pgx11 = v_f32_ld_tnsr_st_msk(160, A1, 1, 255);
    m_matmul_single(pgx02, 0, 0);
    float8_128 res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    pgx02 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    float8_128 res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    pgx12 = v_f32_ld_tnsr_st_msk(192, A1, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    pgx03 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    pgx13 = v_f32_ld_tnsr_st_msk(224, A1, 1, 255);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    pgx00 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //11
    pgx10 = v_f32_ld_tnsr_st_msk(256, A1, 1, 255);
    m_matmul_single(pgx01, 0, 0);
    res4 = m_pop_mrf(0);

    //12
    pgx01 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    res5 = m_pop_mrf(1);

    //13
    res0 = res0 + res1;
    pgx11 = v_f32_ld_tnsr_st_msk(288, A1, 1, 255);
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;
    pgx02 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx12 = v_f32_ld_tnsr_st_msk(320, A1, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    pgx03 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    pgx13 = v_f32_ld_tnsr_st_msk(352, A1, 1, 255);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //19
    m_matmul_single(pgx01, 0, 0);
    res4 = m_pop_mrf(0);

    //20
    m_matmul_single(pgx11, 0, 1);
    res5 = m_pop_mrf(1);

    //21
    res0 = res0 + res1;
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);


    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    pgx00 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);
    res4 = m_pop_mrf(0);

    pgx01 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);
    res5 = m_pop_mrf(1);

    pgx10 = v_f32_ld_tnsr_st_msk(384, A1, 1, 255);
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    pgx11 = v_f32_ld_tnsr_st_msk(416, A1, 1, 255);
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    pgx02 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    res6 = res6 + res7;
    pgx12 = v_f32_ld_tnsr_st_msk(448, A1, 1, 255);
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    pgx03 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    pgx13 = v_f32_ld_tnsr_st_msk(480, A1, 1, 255);
    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);

    //第四个[32, 128] * [128, 128]
    //25
    m_matmul_single(pgx00, 0, 0);
    //26
    m_matmul_single(pgx10, 0, 1);
    //27
    m_matmul_single(pgx01, 0, 0);
    //28
    m_matmul_single(pgx11, 0, 1);
    //29
    m_matmul_single(pgx02, 0, 0);
    res0 = m_pop_mrf(0);
    //30
    m_matmul_single(pgx12, 0, 1);
    res1 = m_pop_mrf(1);
    //31
    m_matmul_single(pgx03, 0, 0);
    res2 = m_pop_mrf(0);
    //32
    m_matmul_single(pgx13, 0, 1);
    C += 4 * 128;
    res3 = m_pop_mrf(1);

    res4 = m_pop_mrf(0);

    res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    res1 = m_pop_mrf(1);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //11
    res4 = m_pop_mrf(0);

    //12
    res5 = m_pop_mrf(1);

    //13
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;

    res7 = m_pop_mrf(1);

    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    res1 = m_pop_mrf(1);


    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //19
    res4 = m_pop_mrf(0);

    res5 = m_pop_mrf(1);

    //21
    res0 = res0 + res1;

    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;

    res7 = m_pop_mrf(1);

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    res1 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //26
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //27
    res4 = m_pop_mrf(0);

    //28
    res5 = m_pop_mrf(1);

    //29
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //30
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //31
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    //32
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
}

inline void matmul_2pgx_256_pop_first_stride(SIM_X86::tensor A0, SIM_X86::tensor A1, SIM_X86::tensor C, int lhs_stride) {
    float8_128 pgx00 = load8_128_stride(0, lhs_stride, A0);
    float8_128 pgx10 = load8_128_stride(0,  lhs_stride, A1);
    float8_128 pgx01 = load8_128_stride(lhs_stride * 32, lhs_stride, A0);
    float8_128 pgx11 = load8_128_stride(lhs_stride * 32,  lhs_stride, A1);
    float8_128 pgx02 = load8_128_stride(lhs_stride * 64, lhs_stride, A0);
    float8_128 pgx12 = load8_128_stride(lhs_stride * 64,  lhs_stride, A1);
    float8_128 pgx03 = load8_128_stride(lhs_stride * 96, lhs_stride, A0);
    float8_128 res0 = m_pop_mrf(0);
    float8_128 pgx13 = load8_128_stride(lhs_stride * 96,  lhs_stride, A1);
    float8_128 res1 = m_pop_mrf(1);

    //第一个[32, 128] * [128, 128]
    //1
    m_matmul_single(pgx00, 0, 0);
    float8_128 res2 = m_pop_mrf(0);

    //2
    pgx00 = load8_128_stride(lhs_stride * 128, lhs_stride, A0);
    m_matmul_single(pgx10, 0, 1);
    float8_128 res3 = m_pop_mrf(1);

    //3
    pgx10 = load8_128_stride(lhs_stride * 128,  lhs_stride, A1);
    m_matmul_single(pgx01, 0, 0);
    float8_128 res4 = m_pop_mrf(0);

    //4
    pgx01 = load8_128_stride(lhs_stride * 160, lhs_stride, A0);
    m_matmul_single(pgx11, 0, 1);
    float8_128 res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    pgx11 = load8_128_stride(lhs_stride * 160,  lhs_stride, A1);
    m_matmul_single(pgx02, 0, 0);
    float8_128 res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    pgx02 = load8_128_stride(lhs_stride * 192, lhs_stride, A0);
    m_matmul_single(pgx12, 0, 1);
    float8_128 res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    pgx12 = load8_128_stride(lhs_stride * 192,  lhs_stride, A1);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    pgx03 = load8_128_stride(lhs_stride * 224, lhs_stride, A0);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    pgx13 = load8_128_stride(lhs_stride * 224,  lhs_stride, A1);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    pgx00 = load8_128_stride(lhs_stride * 256, lhs_stride, A0);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //11
    pgx10 = load8_128_stride(lhs_stride * 256, lhs_stride, A1);
    m_matmul_single(pgx01, 0, 0);
    res4 = m_pop_mrf(0);

    //12
    pgx01 = load8_128_stride(lhs_stride * 288, lhs_stride, A0);
    m_matmul_single(pgx11, 0, 1);
    res5 = m_pop_mrf(1);

    //13
    res0 = res0 + res1;
    pgx11 = load8_128_stride(lhs_stride * 288, lhs_stride, A1);
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;
    pgx02 = load8_128_stride(lhs_stride * 320, lhs_stride, A0);
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx12 = load8_128_stride(lhs_stride * 320, lhs_stride, A1);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    pgx03 = load8_128_stride(lhs_stride * 352, lhs_stride, A0);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    pgx13 = load8_128_stride(lhs_stride * 352,  lhs_stride, A1);
    m_matmul_single(pgx00, 0, 0);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    m_matmul_single(pgx10, 0, 1);
    res3 = m_pop_mrf(1);

    //19
    m_matmul_single(pgx01, 0, 0);
    res4 = m_pop_mrf(0);

    //20
    m_matmul_single(pgx11, 0, 1);
    res5 = m_pop_mrf(1);

    //21
    res0 = res0 + res1;
    m_matmul_single(pgx02, 0, 0);
    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;
    m_matmul_single(pgx12, 0, 1);
    res7 = m_pop_mrf(1);

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    m_matmul_single(pgx03, 0, 0);
    res0 = m_pop_mrf(0);

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    m_matmul_single(pgx13, 0, 1);
    res1 = m_pop_mrf(1);


    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    pgx00 = load8_128_stride(lhs_stride * 384, lhs_stride, A0);
    res4 = m_pop_mrf(0);

    pgx01 = load8_128_stride(lhs_stride * 416, lhs_stride, A0);
    res5 = m_pop_mrf(1);

    pgx10 = load8_128_stride(lhs_stride * 384, lhs_stride, A1);
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    pgx11 = load8_128_stride(lhs_stride * 416, lhs_stride, A1);
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    pgx02 = load8_128_stride(lhs_stride * 448, lhs_stride, A0);
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    res6 = res6 + res7;
    pgx12 = load8_128_stride(lhs_stride * 448, lhs_stride, A1);
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    pgx03 = load8_128_stride(lhs_stride * 480, lhs_stride, A0);
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    pgx13 = load8_128_stride(lhs_stride * 480, lhs_stride, A1);
    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);

    //第四个[32, 128] * [128, 128]
    //25
    m_matmul_single(pgx00, 0, 0);
    //26
    m_matmul_single(pgx10, 0, 1);
    //27
    m_matmul_single(pgx01, 0, 0);
    //28
    m_matmul_single(pgx11, 0, 1);
    //29
    m_matmul_single(pgx02, 0, 0);
    res0 = m_pop_mrf(0);
    //30
    m_matmul_single(pgx12, 0, 1);
    res1 = m_pop_mrf(1);
    //31
    m_matmul_single(pgx03, 0, 0);
    res2 = m_pop_mrf(0);
    //32
    m_matmul_single(pgx13, 0, 1);
    C += 4 * 128;
    res3 = m_pop_mrf(1);

    res4 = m_pop_mrf(0);

    res5 = m_pop_mrf(1);

    //5
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //6
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //7
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //8
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    res1 = m_pop_mrf(1);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //11
    res4 = m_pop_mrf(0);

    //12
    res5 = m_pop_mrf(1);

    //13
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //14
    res2 = res2 + res3;

    res7 = m_pop_mrf(1);

    //15
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //16
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    res1 = m_pop_mrf(1);


    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //19
    res4 = m_pop_mrf(0);

    res5 = m_pop_mrf(1);

    //21
    res0 = res0 + res1;

    res6 = m_pop_mrf(0);

    //22
    res2 = res2 + res3;

    res7 = m_pop_mrf(1);

    //23
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //24
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);
    res1 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //26
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);
    res3 = m_pop_mrf(1);

    //27
    res4 = m_pop_mrf(0);

    //28
    res5 = m_pop_mrf(1);

    //29
    res0 = res0 + res1;
    res6 = m_pop_mrf(0);

    //30
    res2 = res2 + res3;
    res7 = m_pop_mrf(1);

    //31
    res4 = res4 + res5;
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    //32
    res6 = res6 + res7;
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);

}

inline void matmul_2pgx_256_pop(SIM_X86::tensor A0, SIM_X86::tensor A1, SIM_X86::tensor C) {
    float8_128 res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    float8_128 res4 = m_pop_mrf(0);

    float8_128 res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    float8_128 res5 = m_pop_mrf(1);

    float8_128 res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    float8_128 res6 = m_pop_mrf(0);

    float8_128 res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    float8_128 res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    float8_128 pgx00 = v_f32_ld_tnsr_st_msk(0, A0, 1, 255);

    res1 = res1 + res6;
    float8_128 pgx10 = v_f32_ld_tnsr_st_msk(0, A1, 1, 255);

    res0 = res0 + res5;
    float8_128 pgx01 = v_f32_ld_tnsr_st_msk(32, A0, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    float8_128 pgx11 = v_f32_ld_tnsr_st_msk(32, A1, 1, 255);
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    float8_128 pgx02 = v_f32_ld_tnsr_st_msk(64, A0, 1, 255);
    res5 = m_pop_mrf(0);

    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    float8_128 pgx12 = v_f32_ld_tnsr_st_msk(64, A1, 1, 255);
    res7 = m_pop_mrf(1);

    res3 = res3 + res5;
    float8_128 pgx03 = v_f32_ld_tnsr_st_msk(96, A0, 1, 255);

    res3 = res3 + res7;
    float8_128 pgx13 = v_f32_ld_tnsr_st_msk(96, A1, 1, 255);

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);
    m_matmul_single(pgx00, 0, 0);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    m_matmul_single(pgx01, 0, 0);
    res5 = m_pop_mrf(1);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    m_matmul_single(pgx02, 0, 0);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;
    pgx00 = v_f32_ld_tnsr_st_msk(128, A0, 1, 255);
    m_matmul_single(pgx12, 0, 1);

    //7
    res1 = res1 + res6;
    pgx10 = v_f32_ld_tnsr_st_msk(128, A1, 1, 255);
    m_matmul_single(pgx03, 0, 0);

    //8
    res0 = res0 + res5;
    pgx01 = v_f32_ld_tnsr_st_msk(160, A0, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    pgx11 = v_f32_ld_tnsr_st_msk(160, A1, 1, 255);
    m_matmul_single(pgx00, 0, 0);
    res6 = m_pop_mrf(1);

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx02 = v_f32_ld_tnsr_st_msk(192, A0, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res5 = m_pop_mrf(0);


    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    pgx12 = v_f32_ld_tnsr_st_msk(192, A1, 1, 255);
    m_matmul_single(pgx01, 0, 0);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;
    pgx03 = v_f32_ld_tnsr_st_msk(224, A0, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    //13
    res3 = res3 + res7;
    pgx13 = v_f32_ld_tnsr_st_msk(224, A1, 1, 255);
    m_matmul_single(pgx02, 0, 0);


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res5 = m_pop_mrf(1);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    pgx00 = v_f32_ld_tnsr_st_msk(256, A0, 1, 255);

    res1 = res1 + res6;
    pgx10 = v_f32_ld_tnsr_st_msk(256, A1, 1, 255);

    res0 = res0 + res5;
    pgx01 = v_f32_ld_tnsr_st_msk(288, A0, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    pgx11 = v_f32_ld_tnsr_st_msk(288, A1, 1, 255);
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    pgx02 = v_f32_ld_tnsr_st_msk(320, A0, 1, 255);
    m_matmul_single(pgx00, 0, 0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    pgx12 = v_f32_ld_tnsr_st_msk(320, A1, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res7 = m_pop_mrf(1);

    //19
    res3 = res3 + res5;
    pgx03 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    //20
    res3 = res3 + res7;
    pgx13 = v_f32_ld_tnsr_st_msk(352, A1, 1, 255);
    m_matmul_single(pgx11, 0, 1);

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    m_matmul_single(pgx02, 0, 0);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    m_matmul_single(pgx12, 0, 1);
    res5 = m_pop_mrf(1);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res7 = m_pop_mrf(1);


    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;

    res3 = res3 + res7;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);

    pgx00 = v_f32_ld_tnsr_st_msk(384, A0, 1, 255);

    pgx10 = v_f32_ld_tnsr_st_msk(384, A1, 1, 255);

    pgx01 = v_f32_ld_tnsr_st_msk(416, A0, 1, 255);

    pgx11 = v_f32_ld_tnsr_st_msk(416, A1, 1, 255);

    //第四个[32, 128] * [128, 128]
    //25
    pgx02 = v_f32_ld_tnsr_st_msk(448, A0, 1, 255);
    m_matmul_single(pgx00, 0, 0);

    //26
    m_matmul_single(pgx10, 0, 1);
    pgx12 = v_f32_ld_tnsr_st_msk(448, A1, 1, 255);

    //27
    m_matmul_single(pgx01, 0, 0);
    pgx03 = v_f32_ld_tnsr_st_msk(480, A0, 1, 255);

    //28
    m_matmul_single(pgx11, 0, 1);
    pgx13 = v_f32_ld_tnsr_st_msk(480, A1, 1, 255);
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    m_matmul_single(pgx12, 0, 1);
    //31
    m_matmul_single(pgx03, 0, 0);
    //32
    m_matmul_single(pgx13, 0, 1);

    C += 4 * 128;

    res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    res5 = m_pop_mrf(1);

    res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    res3 = res3 + res5;

    res3 = res3 + res7;

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    res5 = m_pop_mrf(1);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;

    //7
    res1 = res1 + res6;

    //8
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;

    //13
    res3 = res3 + res7;


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    res5 = m_pop_mrf(1);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //19
    res3 = res3 + res5;

    //20
    res3 = res3 + res7;

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    res5 = m_pop_mrf(1);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    res7 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    res0 = res0 + res4;
    //26
    res1 = res1 + res6;

    //27
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //28
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //29
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //30
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //31
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    //32
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;

    res3 = res3 + res7;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);
}

inline void matmul_2pgx_256_pop_stride(SIM_X86::tensor A0, SIM_X86::tensor A1, SIM_X86::tensor C, int lhs_stride) {
    float8_128 res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    float8_128 res4 = m_pop_mrf(0);

    float8_128 res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    float8_128 res5 = m_pop_mrf(1);

    float8_128 res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    float8_128 res6 = m_pop_mrf(0);

    float8_128 res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    float8_128 res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    float8_128 pgx00 = load8_128_stride(0, lhs_stride, A0);

    res1 = res1 + res6;
    float8_128 pgx10 = load8_128_stride(0, lhs_stride, A1);

    res0 = res0 + res5;
    float8_128 pgx01 = load8_128_stride(lhs_stride * 32, lhs_stride, A0);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    float8_128 pgx11 = load8_128_stride(lhs_stride * 32, lhs_stride, A1);
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    float8_128 pgx02 = load8_128_stride(lhs_stride * 64, lhs_stride, A0);
    res5 = m_pop_mrf(0);

    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    float8_128 pgx12 = load8_128_stride(lhs_stride * 64, lhs_stride, A1);
    res7 = m_pop_mrf(1);

    res3 = res3 + res5;
    float8_128 pgx03 = load8_128_stride(lhs_stride * 96, lhs_stride, A0);

    res3 = res3 + res7;
    float8_128 pgx13 = load8_128_stride(lhs_stride * 96, lhs_stride, A1);

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);
    m_matmul_single(pgx00, 0, 0);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    m_matmul_single(pgx10, 0, 1);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    m_matmul_single(pgx01, 0, 0);
    res5 = m_pop_mrf(1);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    m_matmul_single(pgx11, 0, 1);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    m_matmul_single(pgx02, 0, 0);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;
    pgx00 = load8_128_stride(lhs_stride * 128, lhs_stride, A0);
    m_matmul_single(pgx12, 0, 1);

    //7
    res1 = res1 + res6;
    pgx10 = load8_128_stride(lhs_stride * 128, lhs_stride, A1);
    m_matmul_single(pgx03, 0, 0);

    //8
    res0 = res0 + res5;
    pgx01 = load8_128_stride(lhs_stride * 160, lhs_stride, A0);
    m_matmul_single(pgx13, 0, 1);
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    pgx11 = load8_128_stride(lhs_stride * 160, lhs_stride, A1);
    m_matmul_single(pgx00, 0, 0);
    res6 = m_pop_mrf(1);

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx02 = load8_128_stride(lhs_stride * 192, lhs_stride, A0);
    m_matmul_single(pgx10, 0, 1);
    res5 = m_pop_mrf(0);


    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    pgx12 = load8_128_stride(lhs_stride * 192, lhs_stride, A1);
    m_matmul_single(pgx01, 0, 0);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;
    pgx03 = load8_128_stride(lhs_stride * 224, lhs_stride, A0);
    m_matmul_single(pgx11, 0, 1);

    //13
    res3 = res3 + res7;
    pgx13 = load8_128_stride(lhs_stride * 224, lhs_stride, A1);
    m_matmul_single(pgx02, 0, 0);


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    m_matmul_single(pgx12, 0, 1);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res5 = m_pop_mrf(1);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;
    pgx00 = load8_128_stride(lhs_stride * 256, lhs_stride, A0);

    res1 = res1 + res6;
    pgx10 = load8_128_stride(lhs_stride * 256, lhs_stride, A1);

    res0 = res0 + res5;
    pgx01 = load8_128_stride(lhs_stride * 288, lhs_stride, A0);
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    pgx11 = load8_128_stride(lhs_stride * 288, lhs_stride, A1);
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    pgx02 = load8_128_stride(lhs_stride * 320, lhs_stride, A0);
    m_matmul_single(pgx00, 0, 0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    pgx12 = load8_128_stride(lhs_stride * 320, lhs_stride, A1);
    m_matmul_single(pgx10, 0, 1);
    res7 = m_pop_mrf(1);

    //19
    res3 = res3 + res5;
    pgx03 = load8_128_stride(lhs_stride * 352, lhs_stride, A0);
    m_matmul_single(pgx01, 0, 0);

    //20
    res3 = res3 + res7;
    pgx13 = load8_128_stride(lhs_stride * 352, lhs_stride, A1);
    m_matmul_single(pgx11, 0, 1);

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    m_matmul_single(pgx02, 0, 0);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    m_matmul_single(pgx12, 0, 1);
    res5 = m_pop_mrf(1);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    m_matmul_single(pgx13, 0, 1);
    res7 = m_pop_mrf(1);


    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;

    res3 = res3 + res7;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);

    pgx00 = load8_128_stride(lhs_stride * 384, lhs_stride, A0);

    pgx10 = load8_128_stride(lhs_stride * 384, lhs_stride, A1);

    pgx01 = load8_128_stride(lhs_stride * 416, lhs_stride, A0);

    pgx11 = load8_128_stride(lhs_stride * 416, lhs_stride, A1);

    //第四个[32, 128] * [128, 128]
    //25
    pgx02 = load8_128_stride(lhs_stride * 448, lhs_stride, A0);
    m_matmul_single(pgx00, 0, 0);

    //26
    m_matmul_single(pgx10, 0, 1);
    pgx12 = load8_128_stride(lhs_stride * 448, lhs_stride, A1);

    //27
    m_matmul_single(pgx01, 0, 0);
    pgx03 = load8_128_stride(lhs_stride * 480, lhs_stride, A0);

    //28
    m_matmul_single(pgx11, 0, 1);
    pgx13 = load8_128_stride(lhs_stride * 480, lhs_stride, A1);
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    m_matmul_single(pgx12, 0, 1);
    //31
    m_matmul_single(pgx03, 0, 0);
    //32
    m_matmul_single(pgx13, 0, 1);

    C += 4 * 128;

    res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);
    res5 = m_pop_mrf(1);

    res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    res3 = res3 + res5;

    res3 = res3 + res7;

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    res5 = m_pop_mrf(1);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    res7 = m_pop_mrf(1);

    //6
    res0 = res0 + res4;

    //7
    res1 = res1 + res6;

    //8
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    //11
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //12
    res3 = res3 + res5;

    //13
    res3 = res3 + res7;


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    res5 = m_pop_mrf(1);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);
    res7 = m_pop_mrf(1);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //18
    res2 = res2 + res6;
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //19
    res3 = res3 + res5;

    //20
    res3 = res3 + res7;

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);
    res5 = m_pop_mrf(1);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);
    res7 = m_pop_mrf(1);

    //第四个[32, 128] * [128, 128]
    //25
    res0 = res0 + res4;
    //26
    res1 = res1 + res6;

    //27
    res0 = res0 + res5;
    res4 = m_pop_mrf(0);

    //28
    res1 = res1 + res7;
    res6 = m_pop_mrf(1);

    //29
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //30
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);
    res7 = m_pop_mrf(1);

    //31
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    //32
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res2 = res2 + res6;

    res3 = res3 + res5;

    res3 = res3 + res7;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);
}

inline void matmul_256_pre16(SIM_X86::tensor A0, int cur_aw) {
    float8_128 pgx00 = load8_k(A0 + 0, 1, 255, cur_aw, 0);
    float8_128 pgx01 = load8_k(A0 + 32, 1, 255, cur_aw, 0);
    float8_128 pgx02 = load8_k(A0 + 64, 1, 255, cur_aw, 0);
    float8_128 pgx03 = load8_k(A0 + 96, 1, 255, cur_aw, 0);

    //第一部分的[64, 128] * [128, 128]
    m_matmul_single(pgx00, 0, 0);

    pgx00 = load8_k(A0 + 128, 1, 255, cur_aw, 0);

    m_matmul_single(pgx01, 0, 0);

    pgx01 = load8_k(A0 + 160, 1, 255, cur_aw, 0);

    m_matmul_single(pgx02, 0, 0);

    pgx02 = load8_k(A0 + 192, 1, 255, cur_aw, 0);

    m_matmul_single(pgx03, 0, 0);

    pgx03 = load8_k(A0 + 224, 1, 255, cur_aw, 0);

    m_matmul_single(pgx00, 0, 0);

    pgx00 = load8_k(A0 + 256, 1, 255, cur_aw, 0);

    m_matmul_single(pgx01, 0, 0);

    pgx01 = load8_k(A0 + 288, 1, 255, cur_aw, 0);

    m_matmul_single(pgx02, 0, 0);

    pgx02 = load8_k(A0 + 320, 1, 255, cur_aw, 0);

    m_matmul_single(pgx03, 0, 0);

    pgx03 = load8_k(A0 + 352, 1, 255, cur_aw, 0);

    m_matmul_single(pgx00, 0, 0);

    pgx00 = load8_k(A0 + 384, 1, 255, cur_aw, 0);

    m_matmul_single(pgx01, 0, 0);

    pgx01 = load8_k(A0 + 416, 1, 255, cur_aw, 0);

    m_matmul_single(pgx02, 0, 0);

    pgx02 = load8_k(A0 + 448, 1, 255, cur_aw, 0);

    m_matmul_single(pgx03, 0, 0);

    pgx03 = load8_k(A0 + 480, 1, 255, cur_aw, 0);

    m_matmul_single(pgx00, 0, 0);

    m_matmul_single(pgx01, 0, 0);

    m_matmul_single(pgx02, 0, 0);

    m_matmul_single(pgx03, 0, 0);
}

inline void matmul_256_pop_first(SIM_X86::tensor A0, SIM_X86::tensor C, int cur_aw) {
    float8_128 res0 = m_pop_mrf(0);
    float8_128 pgx00 = load8_k(A0 + 0, 1, 255, cur_aw, 0);
    float8_128 pgx01 = load8_k(A0 + 32, 1, 255, cur_aw, 0);
    float8_128 pgx02 = load8_k(A0 + 64, 1, 255, cur_aw, 0);
    float8_128 pgx03 = load8_k(A0 + 96, 1, 255, cur_aw, 0);

    //第一个[32, 128] * [128, 128]
    //1
    float8_128 res2 = m_pop_mrf(0);
    m_matmul_single(pgx00, 0, 0);

    //2
    pgx00 = load8_k(A0 + 128, 1, 255, cur_aw, 0);

    //3
    float8_128 res4 = m_pop_mrf(0);
    m_matmul_single(pgx01, 0, 0);

    //4
    pgx01 = load8_k(A0 + 160, 1, 255, cur_aw, 0);

    //5
    float8_128 res6 = m_pop_mrf(0);
    m_matmul_single(pgx02, 0, 0);

    //6
    pgx02 = load8_k(A0 + 192, 1, 255, cur_aw, 0);

    //7
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res0 = m_pop_mrf(0);
    m_matmul_single(pgx03, 0, 0);

    //8
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);
    pgx03 = load8_k(A0 + 224, 1, 255, cur_aw, 0);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);
    m_matmul_single(pgx00, 0, 0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);
    pgx00 = load8_k(A0 + 256, 1, 255, cur_aw, 0);

    //11
    res4 = m_pop_mrf(0);
    m_matmul_single(pgx01, 0, 0);

    //12
    pgx01 = load8_k(A0 + 288, 1, 255, cur_aw, 0);

    //13
    res6 = m_pop_mrf(0);
    m_matmul_single(pgx02, 0, 0);

    //14
    pgx02 = load8_k(A0 + 320, 1, 255, cur_aw, 0);

    //15
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);
    m_matmul_single(pgx03, 0, 0);

    //16
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);
    pgx03 = load8_k(A0 + 352, 1, 255, cur_aw, 0);

    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);
    m_matmul_single(pgx00, 0, 0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);

    //19
    res4 = m_pop_mrf(0);
    m_matmul_single(pgx01, 0, 0);

    //20

    //21
    res6 = m_pop_mrf(0);
    m_matmul_single(pgx02, 0, 0);

    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);
    m_matmul_single(pgx03, 0, 0);

    //24
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);


    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);

    res4 = m_pop_mrf(0);


    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);

    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
    pgx00 = load8_k(A0 + 384, 1, 255, cur_aw, 0);

    pgx01 = load8_k(A0 + 416, 1, 255, cur_aw, 0);

    pgx02 = load8_k(A0 + 448, 1, 255, cur_aw, 0);

    pgx03 = load8_k(A0 + 480, 1, 255, cur_aw, 0);

    m_matmul_single(pgx00, 0, 0);
    //27
    m_matmul_single(pgx01, 0, 0);
    //28
    //29
    m_matmul_single(pgx02, 0, 0);
    //30
    //31
    m_matmul_single(pgx03, 0, 0);
    //32

    C += 4 * 128;

    res0 = m_pop_mrf(0);

    res2 = m_pop_mrf(0);

    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    //7
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //8
    v_f32_st_tnsr_st_msk(32, C, 1, 255, res2);

    //第二个[32, 128] * [128, 128]
    //9
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //10
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res6);

    //11
    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //16
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res2);


    //第三个[32, 128] * [128, 128]
    //17
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res6);

    //19
    res4 = m_pop_mrf(0);

    res6 = m_pop_mrf(0);

    //23
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res0 = m_pop_mrf(0);

    //24
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res2);

    //第四个[32, 128] * [128, 128]
    //25
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res4);
    res2 = m_pop_mrf(0);

    //26
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res6);

    //27
    res4 = m_pop_mrf(0);

    //29
    res6 = m_pop_mrf(0);

    //31
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    //32
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res2);

    //push gain下一次的前[64, 128]
    v_f32_st_tnsr_st_msk(448, C, 1, 255, res4);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res6);
}

inline void matmul_256_pop(SIM_X86::tensor A0, SIM_X86::tensor C, int cur_aw) {
    float8_128 res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    float8_128 res4 = m_pop_mrf(0);

    float8_128 res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);

    float8_128 res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    float8_128 res6 = m_pop_mrf(0);

    float8_128 res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);

    res0 = res0 + res4;
    float8_128 pgx00 = load8_k(A0 + 0, 1, 255, cur_aw, 0);

    res1 = res1 + res6;

    float8_128 pgx01 = load8_k(A0 + 32, 1, 255, cur_aw, 0);
    res4 = m_pop_mrf(0);

    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    float8_128 pgx02 = load8_k(A0 + 64, 1, 255, cur_aw, 0);
    float8_128 res5 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);

    res3 = res3 + res5;
    float8_128 pgx03 = load8_k(A0 + 96, 1, 255, cur_aw, 0);

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);
    m_matmul_single(pgx00, 0, 0);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);
    m_matmul_single(pgx02, 0, 0);

    //6
    res0 = res0 + res4;
    pgx00 = load8_k(A0 + 128, 1, 255, cur_aw, 0);

    //7
    res1 = res1 + res6;
    m_matmul_single(pgx03, 0, 0);

    //8
    pgx01 = load8_k(A0 + 160, 1, 255, cur_aw, 0);
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]
    //9
    m_matmul_single(pgx00, 0, 0);

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    pgx02 = load8_k(A0 + 192, 1, 255, cur_aw, 0);
    res5 = m_pop_mrf(0);


    //11
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);
    m_matmul_single(pgx01, 0, 0);

    //12
    res3 = res3 + res5;
    pgx03 = load8_k(A0 + 224, 1, 255, cur_aw, 0);

    //13
    m_matmul_single(pgx02, 0, 0);


    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);

    res0 = res0 + res4;
    pgx00 = load8_k(A0 + 256, 1, 255, cur_aw, 0);

    res1 = res1 + res6;

    pgx01 = load8_k(A0 + 288, 1, 255, cur_aw, 0);
    res4 = m_pop_mrf(0);


    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    pgx02 = load8_k(A0 + 320, 1, 255, cur_aw, 0);
    m_matmul_single(pgx00, 0, 0);
    res5 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);

    //19
    res3 = res3 + res5;
    pgx03 = v_f32_ld_tnsr_st_msk(352, A0, 1, 255);
    m_matmul_single(pgx01, 0, 0);

    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    m_matmul_single(pgx02, 0, 0);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    m_matmul_single(pgx03, 0, 0);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);


    res0 = res0 + res4;

    res1 = res1 + res6;

    res4 = m_pop_mrf(0);


    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);

    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res3 = res3 + res5;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);

    pgx00 = load8_k(A0 + 384, 1, 255, cur_aw, 0);

    pgx01 = load8_k(A0 + 416, 1, 255, cur_aw, 0);

    //第四个[32, 128] * [128, 128]
    //25
    pgx02 = load8_k(A0 + 448, 1, 255, cur_aw, 0);
    m_matmul_single(pgx00, 0, 0);

    //27
    m_matmul_single(pgx01, 0, 0);
    pgx03 = load8_k(A0 + 480, 1, 255, cur_aw, 0);

    //29
    m_matmul_single(pgx02, 0, 0);
    //30    //31
    m_matmul_single(pgx03, 0, 0);
    //32
    C += 4 * 128;

    res0 = v_f32_ld_tnsr_st_msk(0, C, 1, 255);
    res4 = m_pop_mrf(0);

    res1 = v_f32_ld_tnsr_st_msk(32, C, 1, 255);

    res2 = v_f32_ld_tnsr_st_msk(64, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(96, C, 1, 255);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res4 = m_pop_mrf(0);


    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(0, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    v_f32_st_tnsr_st_msk(32, C, 1, 255, res1);

    res3 = res3 + res5;

    //第一个[32, 128] * [128, 128]
    //1
    v_f32_st_tnsr_st_msk(64, C, 1, 255, res2);

    //2
    v_f32_st_tnsr_st_msk(96, C, 1, 255, res3);
    res0 = v_f32_ld_tnsr_st_msk(128, C, 1, 255);
    res4 = m_pop_mrf(0);

    //3
    res1 = v_f32_ld_tnsr_st_msk(160, C, 1, 255);

    //4
    res2 = v_f32_ld_tnsr_st_msk(192, C, 1, 255);
    res6 = m_pop_mrf(0);

    //5
    res3 = v_f32_ld_tnsr_st_msk(224, C, 1, 255);

    //6
    res0 = res0 + res4;

    //7
    res1 = res1 + res6;

    //8
    res4 = m_pop_mrf(0);

    //第二个[32, 128] * [128, 128]

    //10
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(128, C, 1, 255, res0);
    res5 = m_pop_mrf(0);


    //11
    v_f32_st_tnsr_st_msk(160, C, 1, 255, res1);

    //12
    res3 = res3 + res5;

    //13

    //14
    v_f32_st_tnsr_st_msk(192, C, 1, 255, res2);
    res0 = v_f32_ld_tnsr_st_msk(256, C, 1, 255);
    res4 = m_pop_mrf(0);

    //15
    v_f32_st_tnsr_st_msk(224, C, 1, 255, res3);
    res1 = v_f32_ld_tnsr_st_msk(288, C, 1, 255);

    //16
    res2 = v_f32_ld_tnsr_st_msk(320, C, 1, 255);
    res6 = m_pop_mrf(0);

    res3 = v_f32_ld_tnsr_st_msk(352, C, 1, 255);

    res0 = res0 + res4;

    res1 = res1 + res6;

    res4 = m_pop_mrf(0);
    //第三个[32, 128] * [128, 128]
    //17
    res2 = res2 + res4;
    v_f32_st_tnsr_st_msk(256, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //18
    v_f32_st_tnsr_st_msk(288, C, 1, 255, res1);

    //19
    res3 = res3 + res5;

    //20
    //21
    v_f32_st_tnsr_st_msk(320, C, 1, 255, res2);
    res4 = m_pop_mrf(0);

    //22
    v_f32_st_tnsr_st_msk(352, C, 1, 255, res3);

    //23
    res0 = v_f32_ld_tnsr_st_msk(384, C, 1, 255);
    res6 = m_pop_mrf(0);

    //24
    res1 = v_f32_ld_tnsr_st_msk(416, C, 1, 255);

    //第四个[32, 128] * [128, 128]
    //25
    res0 = res0 + res4;
    //26
    res1 = res1 + res6;

    //27
    res4 = m_pop_mrf(0);

    //28

    //29
    v_f32_st_tnsr_st_msk(384, C, 1, 255, res0);
    res5 = m_pop_mrf(0);

    //30
    v_f32_st_tnsr_st_msk(416, C, 1, 255, res1);

    //31
    res2 = v_f32_ld_tnsr_st_msk(448, C, 1, 255);

    //32
    res3 = v_f32_ld_tnsr_st_msk(480, C, 1, 255);

    res2 = res2 + res4;

    res3 = res3 + res5;

    v_f32_st_tnsr_st_msk(448, C, 1, 255, res2);

    v_f32_st_tnsr_st_msk(480, C, 1, 255, res3);
}

inline void matmul_gain_opt_rest(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int ibw, int cur_ah, int add_src_flag, float scale) {
    int n = (cur_ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        m_matmul_single(left0 * scale, 0, 0);
    }
#pragma unroll
    for (int i = 12; i < n; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        //         int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        int Coffset = (i - 12) * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);

        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);

        m_matmul_single(left0 * scale, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);

        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(cur_ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);
    }
}

inline void matmul_gain_opt_rest_stride(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int ibw, int cur_ah, int add_src_flag, float scale) {
    int n = (cur_ah + 7) / 8;
    int m = min(12, n);
    int aw128 = ALIGN128(aw);
    int stride = ALIGN128(aw) / 128;

// #pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx = (iaw + i * 8 * aw128) / 32;
        float8_128 left = load8_k(A + AoffsetPgx, stride, 255, min(aw - iaw, 128), 0);

        m_matmul_single(left * scale, 0, 0);
    }
// #pragma unroll
    for (int i = 12; i < n; i++) {
        int AoffsetPgx = (iaw + i * 8 * aw128) / 32;
        int Coffset = (i - 12) * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, 1, 255, C);
            
        float8_128 left = load8_k(A + AoffsetPgx, stride, 255, min(aw - iaw, 128), 0);
        // float8_128 left = load8_128_stride(AoffsetPgx, stride, A);
        m_matmul_single(left * scale, 0, 0);
        float8_128 ret = m_pop_mrf(0);
        res = res + ret;
        v_f32_st_tnsr_b(Coffset, C, res);
    }
// #pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, 1, 255, C);
        float8_128 ret = m_pop_mrf(0);
        res = res + ret;
        v_f32_st_tnsr_b(Coffset, C, res);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(cur_ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, 1, 255, C);
        float8_128 ret = m_pop_mrf(0);
        res = res + ret;
        v_f32_st_tnsr_st_msk(Coffset, C, 1, mask, res);
    }
}


inline void matmul_gain_2pgx_opt_rest(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int bw, int iaw, int ibw, int cur_ah, int add_src_flag, float scale) {
    int n = (cur_ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);

        m_matmul_single(left0 * scale, 0, 0);
        m_matmul_single(left1 * scale, 0, 1);
    }
#pragma unroll
    for (int i = 12; i < n; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        int Coffset = (i - 12) * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);

        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);

        m_matmul_single(left0 * scale, 0, 0);
        m_matmul_single(left1 * scale, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(cur_ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);
    }
}

inline void matmul_gain_2pgx_opt_rest_stride(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int ibw, int cur_ah, int add_src_flag, float scale) {
    int n = (cur_ah + 7) / 8;
    int m = min(12, n);
    int aw128 = ALIGN128(aw);
    int stride = ALIGN128(aw) / 128;
// #pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = (iaw + i * 8 * aw128) / 32;
        int AoffsetPgx1 = AoffsetPgx0 + 4;
        float8_128 left0 = load8_128_stride(AoffsetPgx0, stride, A);
        float8_128 left1 = load8_128_stride(AoffsetPgx1, stride, A);

        m_matmul_single(left0 * scale, 0, 0);
        m_matmul_single(left1 * scale, 0, 1);
    }
// #pragma unroll
    for (int i = 12; i < n; i++) {
        int AoffsetPgx0 = (iaw + i * 8 * aw128) / 32;
        int AoffsetPgx1 = AoffsetPgx0 + 4;
        int Coffset = (i - 12) * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, 1, 255, C);

        float8_128 left0 = load8_128_stride(AoffsetPgx0, stride, A);
        float8_128 left1 = load8_128_stride(AoffsetPgx1, stride, A);

        m_matmul_single(left0 * scale, 0, 0);
        m_matmul_single(left1 * scale, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        res = res + ret0;
        res = res + ret1;
        v_f32_st_tnsr_b(Coffset, C, res);
    }
// #pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, 1, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        v_f32_st_tnsr_b(Coffset, C, res);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(cur_ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, 1, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        v_f32_st_tnsr_st_msk(Coffset, C, 1, mask, res);
    }
}

//默认gain里已有数据，只做高度为ah，aw为256的matmul。使用经过完全循环展开和指令排布的矩阵乘法，性能较好，指令数较多
//左矩阵和输出均为f32
inline void matmul_LHS_aw256_pipeline_f32(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int bw, int iaw, int ibw, int add_src_flag) {
    int ah256 = ah & 0xFFFFFF00;
    int iah = 0;
    for (; iah < ah256; iah += 256) {
        matmul_2pgx_256_pre16(A + ah * iaw / 32 + iah * 4, A + ah * (iaw + 128) / 32 + iah * 4);
        if (iaw == 0 && add_src_flag == 0) matmul_2pgx_256_pop_first(A + ah * iaw / 32 + 4 * 128 + iah * 4, A + ah * (iaw + 128) / 32 + 4 * 128 + iah * 4, C + iah * 4);
        else matmul_2pgx_256_pop(A + ah * iaw / 32 + 4 * 128 + iah * 4, A + ah * (iaw + 128) / 32 + 4 * 128 + iah * 4, C + iah * 4);
    }

    if (iah < ah) {
        matmul_gain_2pgx_opt_rest(A + iah * 4, C + iah * 4, ah, bw, iaw, ibw, ah - iah, add_src_flag, 1.0);
    }
}

//默认gain里已有数据，只做高度为ah，aw为256的matmul。使用经过完全循环展开和指令排布的矩阵乘法，性能较好，指令数较多
//左矩阵和输出均为f32
//输入正常排布，需要有stride参数，输出特殊排布
inline void matmul_LHS_aw256_pipeline_f32_stride(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int ibw, int add_src_flag) {
    int ah256 = ah & 0xFFFFFF00;
    int aw128 = ALIGN128(aw);
    // int bw128 = ALIGN128(bw);
    int iah = 0;
    int lhs_stride = ALIGN128(aw) / 128;
    for (; iah < ah256; iah += 256) {
        int Aoffset0 = (iah * aw128 + iaw) / 32;
        int Aoffset1 = Aoffset0 + 4;
        matmul_2pgx_256_pre16_stride(A + Aoffset0, A + Aoffset1, lhs_stride);
        if (iaw == 0 && add_src_flag == 0) 
            matmul_2pgx_256_pop_first_stride(A + Aoffset0 + 4 * aw128, A + Aoffset1 + 4 * aw128, C + iah* 4,
                lhs_stride);
        else 
            matmul_2pgx_256_pop_stride(A + Aoffset0 + 4 * aw128, A + Aoffset1 + 4 * aw128, C + iah * 4,
                lhs_stride);
    }

    if (iah < ah) {
        matmul_gain_2pgx_opt_rest_stride(A + iah * aw128 / 32, C + iah * 4, ah, aw, bw, iaw, ibw, ah - iah, add_src_flag, 1.0);
    }
}

//默认gain里已有数据，只做高度为ah，aw为256的matmul。
//左矩阵和输出均为f32
inline void matmul_LHS_aw256_f32(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int bw, int iaw, int ibw, int add_src_flag, float Lscale) {
    matmul_gain_2pgx_opt_rest(A, C, ah, bw, iaw, ibw, ah, add_src_flag, Lscale);
}

//将输入的高16位当作bf16 push进gain里，输入需要是经过HBMtoVmem2函数特殊排列的
//push右矩阵的256*128
inline void push_hi_bf16_2pgx_h(SIM_X86::tensor B0, SIM_X86::tensor B1, float8_128 scale) {
    int stride = 1;
    int once_offset = 32;
    float8_128 pgx0_gain_0 = load8_128_stride(once_offset * 15, stride, B0);
    float8_128 pgx1_gain_0 = load8_128_stride(once_offset * 15, stride, B1);
    float8_128 pgx0_gain_1 = load8_128_stride(once_offset * 14, stride, B0);
    float8_128 pgx1_gain_1 = load8_128_stride(once_offset * 14, stride, B1);
    float8_128 pgx0_gain_2 = load8_128_stride(once_offset * 13, stride, B0);
    float8_128 pgx1_gain_2 = load8_128_stride(once_offset * 13, stride, B1);
    float8_128 pgx0_gain_3 = load8_128_stride(once_offset * 12, stride, B0);
    float8_128 pgx1_gain_3 = load8_128_stride(once_offset * 12, stride, B1);

    pushgain_hi(pgx0_gain_0 * scale, 0, 0);
    pushgain_hi(pgx1_gain_0 * scale, 0, 1);
    pushgain_hi(pgx0_gain_1 * scale, 0, 0);
    pushgain_hi(pgx1_gain_1 * scale, 0, 1);
    pushgain_hi(pgx0_gain_2 * scale, 0, 0);
    pushgain_hi(pgx1_gain_2 * scale, 0, 1);
    pushgain_hi(pgx0_gain_3 * scale, 0, 0);
    pushgain_hi(pgx1_gain_3 * scale, 0, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 11, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 11, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 10, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 10, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 9, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 9, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 8, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 8, stride, B1);

    pushgain_hi(pgx0_gain_0 * scale, 0, 0);
    pushgain_hi(pgx1_gain_0 * scale, 0, 1);
    pushgain_hi(pgx0_gain_1 * scale, 0, 0);
    pushgain_hi(pgx1_gain_1 * scale, 0, 1);
    pushgain_hi(pgx0_gain_2 * scale, 0, 0);
    pushgain_hi(pgx1_gain_2 * scale, 0, 1);
    pushgain_hi(pgx0_gain_3 * scale, 0, 0);
    pushgain_hi(pgx1_gain_3 * scale, 0, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 7, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 7, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 6, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 6, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 5, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 5, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 4, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 4, stride, B1);

    pushgain_hi(pgx0_gain_0 * scale, 0, 0);
    pushgain_hi(pgx1_gain_0 * scale, 0, 1);
    pushgain_hi(pgx0_gain_1 * scale, 0, 0);
    pushgain_hi(pgx1_gain_1 * scale, 0, 1);
    pushgain_hi(pgx0_gain_2 * scale, 0, 0);
    pushgain_hi(pgx1_gain_2 * scale, 0, 1);
    pushgain_hi(pgx0_gain_3 * scale, 0, 0);
    pushgain_hi(pgx1_gain_3 * scale, 0, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 3, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 3, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 2, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 2, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 1, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 1, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 0, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 0, stride, B1);


    pushgain_hi(pgx0_gain_0 * scale, 0, 0);
    pushgain_hi(pgx1_gain_0 * scale, 0, 1);
    pushgain_hi(pgx0_gain_1 * scale, 0, 0);
    pushgain_hi(pgx1_gain_1 * scale, 0, 1);
    pushgain_hi(pgx0_gain_2 * scale, 0, 0);
    pushgain_hi(pgx1_gain_2 * scale, 0, 1);
    pushgain_hi(pgx0_gain_3 * scale, 0, 0);
    pushgain_hi(pgx1_gain_3 * scale, 0, 1);
}

//将输入的低16位当作bf16 push进gain里，输入需要是经过HBMtoVmem2函数特殊排列的
//push右矩阵的256*128
inline void push_lo_bf16_2pgx_h(SIM_X86::tensor B0, SIM_X86::tensor B1, float8_128 scale) {
    int stride = 1;
    int once_offset = 32;
    float8_128 pgx0_gain_0 = load8_128_stride(once_offset * 15, stride, B0);
    float8_128 pgx1_gain_0 = load8_128_stride(once_offset * 15, stride, B1);
    float8_128 pgx0_gain_1 = load8_128_stride(once_offset * 14, stride, B0);
    float8_128 pgx1_gain_1 = load8_128_stride(once_offset * 14, stride, B1);
    float8_128 pgx0_gain_2 = load8_128_stride(once_offset * 13, stride, B0);
    float8_128 pgx1_gain_2 = load8_128_stride(once_offset * 13, stride, B1);
    float8_128 pgx0_gain_3 = load8_128_stride(once_offset * 12, stride, B0);
    float8_128 pgx1_gain_3 = load8_128_stride(once_offset * 12, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0 * scale, 0);
    push_gsnf(pgx1_gain_0 * scale, 1);
    push_gsnf(pgx0_gain_1 * scale, 0);
    push_gsnf(pgx1_gain_1 * scale, 1);
    push_gsnf(pgx0_gain_2 * scale, 0);
    push_gsnf(pgx1_gain_2 * scale, 1);
    push_gsnf(pgx0_gain_3 * scale, 0);
    push_gsnf(pgx1_gain_3 * scale, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 11, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 11, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 10, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 10, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 9, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 9, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 8, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 8, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0 * scale, 0);
    push_gsnf(pgx1_gain_0 * scale, 1);
    push_gsnf(pgx0_gain_1 * scale, 0);
    push_gsnf(pgx1_gain_1 * scale, 1);
    push_gsnf(pgx0_gain_2 * scale, 0);
    push_gsnf(pgx1_gain_2 * scale, 1);
    push_gsnf(pgx0_gain_3 * scale, 0);
    push_gsnf(pgx1_gain_3 * scale, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 7, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 7, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 6, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 6, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 5, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 5, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 4, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 4, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0 * scale, 0);
    push_gsnf(pgx1_gain_0 * scale, 1);
    push_gsnf(pgx0_gain_1 * scale, 0);
    push_gsnf(pgx1_gain_1 * scale, 1);
    push_gsnf(pgx0_gain_2 * scale, 0);
    push_gsnf(pgx1_gain_2 * scale, 1);
    push_gsnf(pgx0_gain_3 * scale, 0);
    push_gsnf(pgx1_gain_3 * scale, 1);

    pgx0_gain_0 = load8_128_stride(once_offset * 3, stride, B0);
    pgx1_gain_0 = load8_128_stride(once_offset * 3, stride, B1);
    pgx0_gain_1 = load8_128_stride(once_offset * 2, stride, B0);
    pgx1_gain_1 = load8_128_stride(once_offset * 2, stride, B1);
    pgx0_gain_2 = load8_128_stride(once_offset * 1, stride, B0);
    pgx1_gain_2 = load8_128_stride(once_offset * 1, stride, B1);
    pgx0_gain_3 = load8_128_stride(once_offset * 0, stride, B0);
    pgx1_gain_3 = load8_128_stride(once_offset * 0, stride, B1);

    pgx0_gain_0 = as_float(v_u32_shl(as_int(pgx0_gain_0), v_u32_move_i(16)));
    pgx1_gain_0 = as_float(v_u32_shl(as_int(pgx1_gain_0), v_u32_move_i(16)));
    pgx0_gain_1 = as_float(v_u32_shl(as_int(pgx0_gain_1), v_u32_move_i(16)));
    pgx1_gain_1 = as_float(v_u32_shl(as_int(pgx1_gain_1), v_u32_move_i(16)));
    pgx0_gain_2 = as_float(v_u32_shl(as_int(pgx0_gain_2), v_u32_move_i(16)));
    pgx1_gain_2 = as_float(v_u32_shl(as_int(pgx1_gain_2), v_u32_move_i(16)));
    pgx0_gain_3 = as_float(v_u32_shl(as_int(pgx0_gain_3), v_u32_move_i(16)));
    pgx1_gain_3 = as_float(v_u32_shl(as_int(pgx1_gain_3), v_u32_move_i(16)));

    push_gsnf(pgx0_gain_0 * scale, 0);
    push_gsnf(pgx1_gain_0 * scale, 1);
    push_gsnf(pgx0_gain_1 * scale, 0);
    push_gsnf(pgx1_gain_1 * scale, 1);
    push_gsnf(pgx0_gain_2 * scale, 0);
    push_gsnf(pgx1_gain_2 * scale, 1);
    push_gsnf(pgx0_gain_3 * scale, 0);
    push_gsnf(pgx1_gain_3 * scale, 1);
}

//将右矩阵的256*128push进gain里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_aw256_bf16(SIM_X86::tensor B, int aw, int iaw, int ibw, int bf_ibw, float8_128 scale){
    int BoffsetPgx0 = (aw * bf_ibw + iaw * 128) / 32;
    int BoffsetPgx1 = BoffsetPgx0 + 4 * 128;
    if (ibw % 256) {
        push_hi_bf16_2pgx_h(B + BoffsetPgx0, B + BoffsetPgx1, scale);
    }
    else {
        push_lo_bf16_2pgx_h(B + BoffsetPgx0, B + BoffsetPgx1, scale);
    }
}

//将右矩阵的128*128push进gain里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_aw128_bf16(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, int bf_ibw, float8_128 scale) {
    int aw256 = aw & 0xffffff00;
    int last_aw = aw - aw256;
    int cur_awn = min((aw - iaw + 7), 128) / 8;
    int BoffsetPgx0 = (aw * bf_ibw + iaw * 128) / 32;
    float8_128 zero = 0;
    for (int i = 15; i >= cur_awn; i--) {
        push_gsnf(zero, 0);
    }
    if (ibw % 256) {
        if (1) {
            int i = cur_awn - 1;
            int h = min(last_aw - i * 8, 8);
            int mask = pre_exp2(h);
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + 32 * i, 1, mask, min(bw - ibw, 128), 0);
            pushgain_hi(gain_pgx0 * scale, 0, 0);
        }
        for (int i = cur_awn - 2; i >= 0; i--) {
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + 32 * i, 1, 255, min(bw - ibw, 128), 0);
            pushgain_hi(gain_pgx0 * scale, 0, 0);
        }
    }
    else {
        if (1) {
            int i = cur_awn - 1;
            int h = min(last_aw - i * 8, 8);
            int mask = pre_exp2(h);
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + 32 * i, 1, mask, min(bw - ibw, 128), 0);
            gain_pgx0 = as_float(v_u32_shl(as_int(gain_pgx0), v_u32_move_i(16)));
            push_gsnf(gain_pgx0 * scale, 0);
        }
        for (int i = cur_awn - 2; i >= 0; i--) {
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + 32 * i, 1, 255, min(bw - ibw, 128), 0);
            gain_pgx0 = as_float(v_u32_shl(as_int(gain_pgx0), v_u32_move_i(16)));
            push_gsnf(gain_pgx0 * scale, 0);
        }
    }
}

//将右矩阵的256*128push进gstf里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_2pgx_T_bf16(SIM_X86::tensor B0, SIM_X86::tensor B1, int aw, float8_128 scaling_factor, int w) {
    int stride = 1;
    int once_offset = 32;
    if (1) {
        float8_128 pgx0_gain_0 = load8_k(B0 + once_offset * 15, stride, 255, w, 0);
        float8_128 pgx0_gain_1 = load8_k(B0 + once_offset * 14, stride, 255, w, 0);
        float8_128 pgx0_gain_2 = load8_k(B0 + once_offset * 13, stride, 255, w, 0);
        float8_128 pgx0_gain_3 = load8_k(B0 + once_offset * 12, stride, 255, w, 0);

        float8_128 pgx1_gain_0_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 1));
        float8_128 pgx1_gain_1_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 1));
        float8_128 pgx1_gain_2_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 1));
        float8_128 pgx1_gain_3_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 1));

        float8_128 pgx0_gain_0_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 0));
        float8_128 pgx0_gain_1_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 0));
        float8_128 pgx0_gain_2_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 0));
        float8_128 pgx0_gain_3_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 0));

        pgx0_gain_0_lo *= scaling_factor;
        pgx1_gain_0_hi *= scaling_factor;
        pgx0_gain_1_lo *= scaling_factor;
        pgx1_gain_1_hi *= scaling_factor;
        pgx0_gain_2_lo *= scaling_factor;
        pgx1_gain_2_hi *= scaling_factor;
        pgx0_gain_3_lo *= scaling_factor;
        pgx1_gain_3_hi *= scaling_factor;

        push_gstf(pgx0_gain_0_lo, 0);
        push_gstf(pgx1_gain_0_hi, 1);
        push_gstf(pgx0_gain_1_lo, 0);
        push_gstf(pgx1_gain_1_hi, 1);
        push_gstf(pgx0_gain_2_lo, 0);
        push_gstf(pgx1_gain_2_hi, 1);
        push_gstf(pgx0_gain_3_lo, 0);
        push_gstf(pgx1_gain_3_hi, 1);

        pgx0_gain_0 = load8_k(B0 + once_offset * 11, stride, 255, w, 0);
        pgx0_gain_1 = load8_k(B0 + once_offset * 10, stride, 255, w, 0);
        pgx0_gain_2 = load8_k(B0 + once_offset * 9, stride, 255, w, 0);
        pgx0_gain_3 = load8_k(B0 + once_offset * 8, stride, 255, w, 0);

        pgx1_gain_0_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 1));
        pgx1_gain_1_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 1));
        pgx1_gain_2_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 1));
        pgx1_gain_3_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 1));

        pgx0_gain_0_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 0));
        pgx0_gain_1_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 0));
        pgx0_gain_2_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 0));
        pgx0_gain_3_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 0));

        pgx0_gain_0_lo *= scaling_factor;
        pgx1_gain_0_hi *= scaling_factor;
        pgx0_gain_1_lo *= scaling_factor;
        pgx1_gain_1_hi *= scaling_factor;
        pgx0_gain_2_lo *= scaling_factor;
        pgx1_gain_2_hi *= scaling_factor;
        pgx0_gain_3_lo *= scaling_factor;
        pgx1_gain_3_hi *= scaling_factor;

        push_gstf(pgx0_gain_0_lo, 0);
        push_gstf(pgx1_gain_0_hi, 1);
        push_gstf(pgx0_gain_1_lo, 0);
        push_gstf(pgx1_gain_1_hi, 1);
        push_gstf(pgx0_gain_2_lo, 0);
        push_gstf(pgx1_gain_2_hi, 1);
        push_gstf(pgx0_gain_3_lo, 0);
        push_gstf(pgx1_gain_3_hi, 1);

        pgx0_gain_0 = load8_k(B0 + once_offset * 7, stride, 255, w, 0);
        pgx0_gain_1 = load8_k(B0 + once_offset * 6, stride, 255, w, 0);
        pgx0_gain_2 = load8_k(B0 + once_offset * 5, stride, 255, w, 0);
        pgx0_gain_3 = load8_k(B0 + once_offset * 4, stride, 255, w, 0);

        pgx1_gain_0_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 1));
        pgx1_gain_1_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 1));
        pgx1_gain_2_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 1));
        pgx1_gain_3_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 1));

        pgx0_gain_0_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 0));
        pgx0_gain_1_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 0));
        pgx0_gain_2_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 0));
        pgx0_gain_3_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 0));

        pgx0_gain_0_lo *= scaling_factor;
        pgx1_gain_0_hi *= scaling_factor;
        pgx0_gain_1_lo *= scaling_factor;
        pgx1_gain_1_hi *= scaling_factor;
        pgx0_gain_2_lo *= scaling_factor;
        pgx1_gain_2_hi *= scaling_factor;
        pgx0_gain_3_lo *= scaling_factor;
        pgx1_gain_3_hi *= scaling_factor;

        push_gstf(pgx0_gain_0_lo, 0);
        push_gstf(pgx1_gain_0_hi, 1);
        push_gstf(pgx0_gain_1_lo, 0);
        push_gstf(pgx1_gain_1_hi, 1);
        push_gstf(pgx0_gain_2_lo, 0);
        push_gstf(pgx1_gain_2_hi, 1);
        push_gstf(pgx0_gain_3_lo, 0);
        push_gstf(pgx1_gain_3_hi, 1);

        pgx0_gain_0 = load8_k(B0 + once_offset * 3, stride, 255, w, 0);
        pgx0_gain_1 = load8_k(B0 + once_offset * 2, stride, 255, w, 0);
        pgx0_gain_2 = load8_k(B0 + once_offset * 1, stride, 255, w, 0);
        pgx0_gain_3 = load8_k(B0 + once_offset * 0, stride, 255, w, 0);

        pgx1_gain_0_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 1));
        pgx1_gain_1_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 1));
        pgx1_gain_2_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 1));
        pgx1_gain_3_hi = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 1));

        pgx0_gain_0_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_0), 0));
        pgx0_gain_1_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_1), 0));
        pgx0_gain_2_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_2), 0));
        pgx0_gain_3_lo = bfloat16_to_float(unpack_16b(as_int(pgx0_gain_3), 0));

        pgx0_gain_0_lo *= scaling_factor;
        pgx1_gain_0_hi *= scaling_factor;
        pgx0_gain_1_lo *= scaling_factor;
        pgx1_gain_1_hi *= scaling_factor;
        pgx0_gain_2_lo *= scaling_factor;
        pgx1_gain_2_hi *= scaling_factor;
        pgx0_gain_3_lo *= scaling_factor;
        pgx1_gain_3_hi *= scaling_factor;

        push_gstf(pgx0_gain_0_lo, 0);
        push_gstf(pgx1_gain_0_hi, 1);
        push_gstf(pgx0_gain_1_lo, 0);
        push_gstf(pgx1_gain_1_hi, 1);
        push_gstf(pgx0_gain_2_lo, 0);
        push_gstf(pgx1_gain_2_hi, 1);
        push_gstf(pgx0_gain_3_lo, 0);
        push_gstf(pgx1_gain_3_hi, 1);
    }
}

//将右矩阵的256*128push进gstf里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_aw256_T_bf16(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, int bf_iaw, float scale) {
    int bf_aw = ALIGN256(aw) / 2;
    int BoffsetPgx0 = (bf_iaw * bw + ibw * 128) / 32;
    int BoffsetPgx1 = BoffsetPgx0;
    push_2pgx_T_bf16(B + BoffsetPgx0, B + BoffsetPgx1, bf_aw, scale, min(aw - iaw, 128));
}

//将右矩阵的128*128push进gstf里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_aw128_T_bf16(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, int bf_iaw, float scale) {
    int cur_bwn = min((bw - ibw + 7), 128) / 8;
    int BoffsetPgx0 = (bf_iaw * bw + ibw * 128) / 32;
    int stride = 1;
    int once_offset = 32;
    float8_128 zero = 0;
    for (int i = 15; i >= cur_bwn; i--) {
        push_gstf(zero, 0);
    }
    if (iaw % 256) {
        if (1) {
            int i = cur_bwn - 1;
            int h = min((bw - ibw) - i * 8, 8);
            int mask = pre_exp2(h);
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, mask, min(aw - iaw, 128), 0);
            pushgain_hi(gain_pgx0 * scale, 1, 0);
        }
        for (int i = cur_bwn - 2; i >= 0; i--) {
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, 255, min(aw - iaw, 128), 0);
            pushgain_hi(gain_pgx0 * scale, 1, 0);
        }
    }else{
        if (1) {
            int i = cur_bwn - 1;
            int h = min((bw - ibw) - i * 8, 8);
            int mask = pre_exp2(h);
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, mask, min(aw - iaw, 128), 0);
            gain_pgx0 = as_float(v_u32_shl(as_int(gain_pgx0), v_u32_move_i(16)));
            push_gstf(gain_pgx0 * scale, 0);
        }
        for (int i = cur_bwn - 2; i >= 0; i--) {
            float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + once_offset * i, stride, 255, min(aw - iaw, 128), 0);
            gain_pgx0 = as_float(v_u32_shl(as_int(gain_pgx0), v_u32_move_i(16)));
            push_gstf(gain_pgx0 * scale, 0);
        }
    }
}

//将右矩阵的128*128push进gsnf里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_aw256_f32(SIM_X86::tensor B, int aw, int iaw, int ibw, float scale){
    int BoffsetPgx0 = (aw * ibw + iaw * 128) / 32;
    int BoffsetPgx1 = BoffsetPgx0 + 4 * 128;
    push_gain_2pgx(B + BoffsetPgx0, B + BoffsetPgx1, scale);
}

//将右矩阵的128*256push进gstf里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_aw256_f32_T(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, float scale) {
    uint BoffsetPgx0 = (bw * iaw + ibw * 128) / 32;
    uint BoffsetPgx1 = BoffsetPgx0 + bw * 4;
    push_gstf_2pgx(B + BoffsetPgx0, B + BoffsetPgx1, scale);
}


//将右矩阵的128*128push进gstf里。输入需要是经过HBMtoVmem2函数特殊排列的
inline void push_aw128_f32_T(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, float scale) {
    uint BoffsetPgx = (bw * iaw + ibw * 128) / 32;
    push_gstf_1pgxw(B + BoffsetPgx, min(aw - iaw, 128), scale);
}


inline void push_aw128_f32(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, float scale) {
    int cur_awn = min((aw - iaw + 7), 128) / 8;
    int BoffsetPgx0 = (aw * ibw + iaw * 128) / 32;
    float8_128 zero = 0;
    for (int i = 15; i >= cur_awn; i--) {
        push_gsnf(zero, 0);
    }
    if (1) {
        int i = cur_awn - 1;
        int h = min((aw - iaw) - i * 8, 8);
        int mask = pre_exp2(h);
        float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + i * 32, 1, mask, min(bw - ibw, 128), 0);
        push_gsnf(gain_pgx0 * scale, 0);
    }
    for (int i = cur_awn - 2; i >= 0; i--) {
        float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + i * 32, 1, 255, min(bw - ibw, 128), 0);
        push_gsnf(gain_pgx0 * scale, 0);
    }
}

//默认gain里已有数据，只做高度为ah，aw为128的matmul。
//左矩阵和输出均为bf16,C存放f32格式的结果，D存放bf16格式的结果。
inline void matmul_LHS_aw128_bf16(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int bf_ibw, int add_src_flag, int is_last_aw, float8_128 scale) {
    if (is_last_aw && iaw + 128 >= aw)
        matmul_gain_no_pack_opt_store_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, scale);
    else
        matmul_gain_no_pack_opt_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, scale);
}

//默认gain里已有数据，只做高度为ah，aw为128的matmul。
//左矩阵为bf16,输出为f32,C存放f32格式的结果。
inline void matmul_LHS_aw128_bf16_out_f32(SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int bf_ibw, int add_src_flag, float8_128 scale) {
    matmul_gain_no_pack_opt_bf16(A, C, C, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, scale);
}

inline void matmul_gain_sdpa(SIM_X86::tensor correct_item, SIM_X86::tensor A, SIM_X86::tensor C, int ah, int aw, int bw, int iaw, int ibw) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
#pragma clang loop unroll_count(4)
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        m_matmul_single(left0, 0, 0);
    }
#pragma clang loop unroll_count(4)
    for (int i = 12; i < n; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;

        int Coffset = (i - 12) * 32;
        float8_128 res = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        if (iaw == 0) {
            int CorrectionOffset = (((i - 12) * 8) * 128) / 32;
            float8_128 correction = load8_128_stride_with_ldmask(CorrectionOffset, 1, 255, correct_item);
            res = res * correction;
        }
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);

        m_matmul_single(left0, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);

        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
#pragma clang loop unroll_count(2)
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        if (iaw == 0) {
            int CorrectionOffset = ((i * 8) * 128) / 32;
            float8_128 correction = load8_128_stride_with_ldmask(CorrectionOffset, 1, 255, correct_item);
            res = res * correction;
        }
        float8_128 ret0 = m_pop_mrf(0);
        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        if (iaw == 0) {
            int CorrectionOffset = ((i * 8) * 128) / 32;
            float8_128 correction = load8_128_stride_with_ldmask(CorrectionOffset, 1, 255, correct_item);
            res = res * correction;
        }
        float8_128 ret0 = m_pop_mrf(0);
        res = res + ret0;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);
    }
}

inline void matmul_gain_2pgx_sdpa(SIM_X86::tensor correct_item, SIM_X86::tensor A, SIM_X86::tensor C, int ah, int bw, int iaw, int ibw) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
#pragma clang loop unroll_count(4)
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);

        m_matmul_single(left0, 0, 0);
        m_matmul_single(left1, 0, 1);
    }
#pragma clang loop unroll_count(4)
    for (int i = 12; i < n; i++) {
        int AoffsetPgx0 = ah * iaw / 32 + i * 32;
        int AoffsetPgx1 = ah * (iaw + 128) / 32 + i * 32;
        int Coffset = (i - 12) * 32;
        float8_128 res = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        if (iaw == 0) {
            int CorrectionOffset = (((i - 12) * 8) * 128) / 32;
            float8_128 correction = load8_128_stride_with_ldmask(CorrectionOffset, 1, 255, correct_item);
            res = res * correction;
        }

        float8_128 left0 = v_f32_ld_tnsr_st_msk(AoffsetPgx0, A, 1, 255);
        float8_128 left1 = v_f32_ld_tnsr_st_msk(AoffsetPgx1, A, 1, 255);

        m_matmul_single(left0, 0, 0);
        m_matmul_single(left1, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
#pragma clang loop unroll_count(2)
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        if (iaw == 0) {
            int CorrectionOffset = ((i * 8) * 128) / 32;
            float8_128 correction = load8_128_stride_with_ldmask(CorrectionOffset, 1, 255, correct_item);
            res = res * correction;
        }
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        if (iaw == 0) {
            int CorrectionOffset = ((i * 8) * 128) / 32;
            float8_128 correction = load8_128_stride_with_ldmask(CorrectionOffset, 1, 255, correct_item);
            res = res * correction;
        }
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);
    }
}

//一个完整的vmem矩阵乘法，需要所有tensor都能一次性被vmem容纳。
//左右矩阵与输出均为bf16
inline void matmul_all_bf16(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, int is_last_aw, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    int bf_ibw = 0;
    SIM_X86::tensor D = C;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        int bf_iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_bf16(B, aw, iaw, ibw, bf_ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            //if (Lscale == 1.0) matmul_LHS_aw256_pipeline_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag);
            matmul_LHS_aw256_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, is_last_aw, Lscale);
            bf_iaw += 128;
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_bf16(B, aw, bw, iaw, ibw, bf_ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            matmul_LHS_aw128_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, is_last_aw, Lscale);
        }
        C = C + ah * 4;
        if (ibw % 256) {
            bf_ibw += 128;
            D = D + ah * 4;
        }
    }
}

//一个完整的vmem矩阵乘法，需要所有tensor都能一次性被vmem容纳。
//输入输出均为bf16，右矩阵转置
inline void matmul_all_bf16_RHST(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, int is_last_aw, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    int bw128 = bw & 0xffffff80;
    SIM_X86::tensor D = C;

    for (int ibw = 0; ibw < bw128; ibw += 128) {
        int iaw = 0;
        int bf_iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_T_bf16(B, aw, bw, iaw, ibw, bf_iaw, Rscale);
            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            //if (Lscale == 1.0) matmul_LHS_aw256_pipeline_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag);
            matmul_LHS_aw256_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, is_last_aw, Lscale);
            bf_iaw += 128;
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_T_bf16(B, aw, bw, iaw, ibw, bf_iaw, Rscale);
            m_fakemul(v_u32_move_f(0), 1, 0);
            matmul_LHS_aw128_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, 0, add_src_flag, is_last_aw, Lscale);
        }
        C += ah * 4;
        if (ibw % 256) {
            D = D + ah * 4;
        }
    }
    if ((bw - bw128) != 0) {
        int iaw = 0;
        int bf_iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_T_bf16(B, aw, bw, iaw, bw128, bf_iaw, Rscale);
            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            //if (Lscale == 1.0) matmul_LHS_aw256_pipeline_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, bw128, add_src_flag);
            matmul_LHS_aw256_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, bw128, add_src_flag, is_last_aw, Lscale);
            bf_iaw += 128;
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_T_bf16(B, aw, bw, iaw, bw128, bf_iaw, Rscale);
            m_fakemul(v_u32_move_f(0), 1, 0);
            matmul_LHS_aw128_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, bw128, 0, add_src_flag, is_last_aw, Lscale);
        }
    }
}

//一个完整的vmem矩阵乘法，需要所有tensor都能一次性被vmem容纳。
//左右矩阵均为bf16,输出为f32
inline void matmul_input_bf16_out_f32(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    int bf_ibw = 0;

    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        int bf_iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_bf16(B, aw, iaw, ibw, bf_ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag);
            else matmul_LHS_aw256_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, Lscale);
            bf_iaw += 128;
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_bf16(B, aw, bw, iaw, ibw, bf_ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            matmul_LHS_aw128_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, Lscale);
        }
        C = C + ah * 4;
        if (ibw % 256) {
            bf_ibw += 128;
        }
    }
}

//一个完整的vmem矩阵乘法，需要所有tensor都能一次性被vmem容纳。
//输入矩阵均为bf16,输出矩阵为f32，右矩阵转置
inline void matmul_input_bf16_out_f32_RHST(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    int bw128 = bw & 0xffffff80;
    for (int ibw = 0; ibw < bw128; ibw += 128) {
        int iaw = 0;
        int bf_iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_T_bf16(B, aw, bw, iaw, ibw, bf_iaw, Rscale);
            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag);
            else matmul_LHS_aw256_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, Lscale);
            bf_iaw += 128;
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_T_bf16(B, aw, bw, iaw, ibw, bf_iaw, Rscale);
            m_fakemul(v_u32_move_f(0), 1, 0);
            matmul_LHS_aw128_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, ibw, 0, add_src_flag, Lscale);
        }
        C += ah * 4;
    }
    if ((bw - bw128) != 0) {
        int iaw = 0;
        int bf_iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_T_bf16(B, aw, bw, iaw, bw128, bf_iaw, Rscale);
            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, bw128, add_src_flag);
            else matmul_LHS_aw256_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, bw128, add_src_flag, Lscale);
            bf_iaw += 128;
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_T_bf16(B, aw, bw, iaw, bw128, bf_iaw, Rscale);
            m_fakemul(v_u32_move_f(0), 1, 0);
            matmul_LHS_aw128_bf16_out_f32(A, C, ah, aw, bw, iaw, bf_iaw, bw128, 0, add_src_flag, Lscale);
        }
    }
}

//LHS与Out是f32，RHS是bf16。结果需与correct_item相加，用于sdpa_bf16
inline void matmul_sdpa_bf16(SIM_X86::tensor correct_item, SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw) {
    int aw256 = aw & 0xffffff00;
    int bf_ibw = 0;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_bf16(B, aw, iaw, ibw, bf_ibw, 1.0);
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            matmul_gain_2pgx_sdpa(correct_item, A, C, ah, bw, iaw, ibw);
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_bf16(B, aw, bw, iaw, ibw, bf_ibw, 1.0);
            m_fakemul(v_u32_move_b(0), 0, 0);
            matmul_gain_sdpa(correct_item, A, C, ah, aw, bw, iaw, ibw);
        }
        if (ibw % 256) bf_ibw += 128;
        C += ah * 4;
    }
}

//一个完整的vmem矩阵乘法，需要所有tensor都能一次性被vmem容纳。
//右矩阵矩阵为bf16,输出和左矩阵矩阵为f32
inline void matmul_RHS_bf16_LHS_out_f32(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    int bf_ibw = 0;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_bf16(B, aw, iaw, ibw, bf_ibw, Rscale);
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_f32(A, C, ah, bw, iaw, ibw, add_src_flag);
            else matmul_LHS_aw256_f32(A, C, ah, bw, iaw, ibw, add_src_flag, Lscale);
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_bf16(B, aw, bw, iaw, ibw, bf_ibw, Rscale);
            m_fakemul(v_u32_move_f(0), 0, 0);
            matmul_gain_opt_rest(A, C, ah, aw, bw, iaw, ibw, ah, add_src_flag, Lscale);
        }
        C = C + ah * 4;
        if (ibw % 256) {
            bf_ibw += 128;
        }
    }
}

//一个完整的vmem矩阵乘法，需要所有tensor都能一次性被vmem容纳。
//左右矩阵与输出均为f32
inline void matmul_all_f32(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;

    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_f32(B, aw, iaw, ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_f32(A, C, ah, bw, iaw, ibw, add_src_flag);
            else matmul_LHS_aw256_f32(A, C, ah, bw, iaw, ibw, add_src_flag, Lscale);
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_f32(B, aw, bw, iaw, ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            matmul_gain_opt_rest(A, C, ah, aw, bw, iaw, ibw, ah, add_src_flag, Lscale);
        }
        C = C + ah * 4;
    }
}

//一个完整的vmem矩阵乘法，需要所有tensor都能一次性被vmem容纳。
//左右矩阵与输出均为f32,右矩阵转置
inline void matmul_all_f32_RHST(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    uint bw128 = bw & 0xffffff80;
    uint last_bw = bw - bw128;
    uint last_bwn = (last_bw + 7) / 8;
    float8_128 zero = 0;

    for (int ibw = 0; ibw < bw128; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            // push_aw256_f32_T(B, aw, bw, iaw, ibw, Rscale);
            uint BoffsetPgx0 = (bw * iaw + ibw * 128) / 32;
            uint BoffsetPgx1 = BoffsetPgx0 + bw * 4;
            push_gstf_2pgx(B + BoffsetPgx0, B + BoffsetPgx1, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_f32(A, C, ah, bw, iaw, ibw, add_src_flag);
            else matmul_LHS_aw256_f32(A, C, ah, bw, iaw, ibw, add_src_flag, Lscale);
        }
        for (;iaw < aw; iaw += 128) {
            // push_aw128_f32_T(B, aw, bw, iaw, ibw, Rscale);
            uint BoffsetPgx = (bw * iaw + ibw * 128) / 32;
            push_gstf_1pgxw(B + BoffsetPgx, min(aw - iaw, 128), Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 1, 0);
            matmul_gain_opt_rest(A, C, ah, aw, bw, iaw, ibw, ah, add_src_flag, Lscale);
        }
        C = C + ah * 4;
    }
    if (last_bw){
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            // push_aw256_f32_T(B, aw, bw, iaw, bw128, Rscale);
            int BoffsetPgx0 = (bw * iaw + bw128 * 128) / 32;
            int BoffsetPgx1 = BoffsetPgx0 + bw * 4;
            for (int i = 15; i >= last_bwn; i--) {
                push_gstf(zero, 0);
                push_gstf(zero, 1);
            }

            for (int i = last_bwn - 1; i >= 0; i--) {
                float8_128 gain_pgx0 = load8_128_stride_ldmk(BoffsetPgx0 + i * 32, 1, B, 255);
                float8_128 gain_pgx1 = load8_128_stride_ldmk(BoffsetPgx1 + i * 32, 1, B, 255);
                push_gstf(gain_pgx0 * Rscale, 0);
                push_gstf(gain_pgx1 * Rscale, 1);
            }
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 1, 0);
            m_fakemul(v_u32_move_b(0), 1, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_f32(A, C, ah, bw, iaw, bw128, add_src_flag);
            else matmul_LHS_aw256_f32(A, C, ah, bw, iaw, bw128, add_src_flag, Lscale);
        }
        for (;iaw < aw; iaw += 128) {
            int BoffsetPgx = (bw * iaw + bw128 * 128) / 32;
            float8_128 zero = 0;
            for (int i = 15; i >= last_bwn; i--) {
                push_gstf(zero, 0);
            }
            int8_128 c = get_core_id();
            bool8_128 m = v_s32_cmp(LS, v_u32_and(c, v_u32_move_i(127)), v_u32_move_i(min(aw - iaw, 128)));
            for (int i = last_bwn - 1; i >= 0; i--) {
                float8_128 gain_pgx0 = load8_128_stride_ldmk(BoffsetPgx + i * 32, 1, B, 255);
                push_gstf(v_f32_sel(m, v_u32_move_f(0), gain_pgx0) * Rscale, 0);
            }
            // push_aw128_f32_T(B, aw, bw, iaw, bw128, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 1, 0);
            matmul_gain_opt_rest(A, C, ah, aw, bw, iaw, bw128, ah, add_src_flag, Lscale);
        }
        C = C + ah * 4;
    }
}

//全部是f32。结果需与correct_item相加，用于sdpa
inline void matmul_sdpa_f32(SIM_X86::tensor correct_item, SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw) {
    int aw256 = aw & 0xffffff00;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_f32(B, aw, iaw, ibw, 1.0);
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            matmul_gain_2pgx_sdpa(correct_item, A, C, ah, bw, iaw, ibw);
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_f32(B, aw, bw, iaw, ibw, 1.0);
            m_fakemul(v_u32_move_b(0), 0, 0);
            matmul_gain_sdpa(correct_item, A, C, ah, aw, bw, iaw, ibw);
        }
        C += ah * 4;
    }
}


inline float8_128 pack_i8(float8_128 a, float8_128 b) {
    return as_float((as_int(a) << 8 & 0x0000FF00) | (as_int(b) & 0x000000FF));
}

inline float8_128 pack_i8_4(float8_128 a, float8_128 b, float8_128 c, float8_128 d) {
    return as_float(as_int(a) << 24 | (as_int(b) << 16 & 0x00FF0000) | (as_int(c) << 8 & 0x0000FF00) | (as_int(d) & 0x000000FF));
}

inline float8_128 transfer_RHS_vector(float8_128 input1, bool ishi) {
    float8_128 input2 = input1;
    input2 = v_row_rotate(input2, 0);
    int8_128 input2_i = as_int(input2);
    int8_128 input1_i = as_int(input1);
    if (!ishi) {
        input2_i = v_u32_shl(input2_i, v_u32_move_i(16));
    }
    else {
        input1_i = v_u32_shr(input1_i, v_u32_move_i(16));
    }
    input1_i &= 0xFFFF;
    input2_i &= 0xFFFF0000;
    return as_float(input1_i | input2_i);
}

inline void store_trans_8_128_first(SIM_X86::tensor A, float8_128 input, int offset) {
    v_f32_st_tnsr_st_msk(offset / 32, A, 1, 1, input);
    input = v_row_rotate(input, 0);
    input = v_row_rotate(input, 0);
    v_f32_st_tnsr_st_msk((offset + 128) / 32, A, 1, 1, input);
    input = v_row_rotate(input, 0);
    input = v_row_rotate(input, 0);
    v_f32_st_tnsr_st_msk((offset + 256) / 32, A, 1, 1, input);
    input = v_row_rotate(input, 0);
    input = v_row_rotate(input, 0);
    v_f32_st_tnsr_st_msk((offset + 384) / 32, A, 1, 1, input);
}

inline void store_trans_8_128(SIM_X86::tensor A, float8_128 input, int offset) {
    v_f32_st_tnsr_st_msk(offset / 32, A, 1, 1, input);
    v_f32_st_tnsr_st_msk((offset - 128) / 32, A, 1, 4, input);
    v_f32_st_tnsr_st_msk((offset - 256) / 32, A, 1, 16, input);
    v_f32_st_tnsr_st_msk((offset - 384) / 32, A, 1, 64, input);
}

inline void transfer_RHS_bw256(SIM_X86::tensor A, SIM_X86::tensor B, int len, int ibw) {
    int shift_num = 0;
    if ((ibw / 256) % 2) shift_num = 16;
    if (1) {
        float8_128 x0 = load8_128_stride(0, 1, A);
        float8_128 x1 = load8_128_stride(32, 1, A);
        x0 = as_float(v_u32_shr(as_int(x0), shift_num));
        x1 = as_float(v_u32_shr(as_int(x1), shift_num));
        float8_128 a = transfer_RHS_vector(x0, 0);
        float8_128 b = transfer_RHS_vector(x1, 0);
        store_trans_8_128_first(B, a, 0);
        store_trans_8_128(B, b, 512);
    }
#pragma clang loop unroll_count(8)
    for (int i = 1; i < ((len + 2047) / 2048); i += 1) {
        float8_128 x0 = load8_128_stride(i * 64, 1, A);
        float8_128 x1 = load8_128_stride(i * 64 + 32, 1, A);
        x0 = as_float(v_u32_shr(as_int(x0), shift_num));
        x1 = as_float(v_u32_shr(as_int(x1), shift_num));
        float8_128 a = transfer_RHS_vector(x0, 0);
        float8_128 b = transfer_RHS_vector(x1, 0);
        store_trans_8_128(B, a, i * 1024);
        store_trans_8_128(B, b, 512 + i * 1024);
    }
}

//which = 0: lower bits 0-7; which = 1: lower bits 8-15; which = 2: lower bits 16-23; which = 3: lower bits 24-31
inline float8_128 transfer_left_vector(float8_128 x, int which) {
    int8_128 x1;
    int8_128 x0;
    if (which == 0) {
        x1 = as_int(x) << 8;
        x0 = as_int(x) & 0x000000FF;
    }
    else if (which == 1) {
        x1 = as_int(x) & 0x0000FF00;
        x0 = (as_int(x) >> 8) & 0x000000FF;
    }
    else if (which == 2) {
        x1 = v_u32_shr(as_int(x), 16) & 0x000000FF;
        x0 = (x1 << 8) & 0x0000FF00;
    }
    else {
        x1 = v_u32_shr(as_int(x), 24) & 0x000000FF;
        x0 = (x1 << 8) & 0x0000FF00;
    }
    return as_float(x0 | x1);
}

//Deal with overflow and underflow.
//Res should in range[-2048, 2047]
inline float8_128 add_clamp(float8_128 a, float8_128 b) {
    // int8_128 masked_a = as_int(a) & 0x00FF00FF;
    // int8_128 masked_b = as_int(b) & 0x00FF00FF;
    // return as_float(masked_a + masked_b);
    int8_128 a_0 = v_s32_shrar(as_int(a) << 16, 16);
    int8_128 b_0 = v_s32_shrar(as_int(b) << 16, 16);
    int8_128 add_res_0 = a_0 + b_0;
    bool8_128 mask_gt_0 = v_s32_cmp(GT, add_res_0, 2047);
    bool8_128 mask_ls_0 = v_s32_cmp(LS, add_res_0, -2048);
    add_res_0 = v_s32_sel(mask_gt_0, add_res_0, 2047);
    add_res_0 = v_s32_sel(mask_ls_0, add_res_0, -2048);

    return as_float(add_res_0);
}

inline float8_128 transfer_res_vector(float8_128 x) {
    int8_128 x1 = (as_int(x) >> 8) & 0x0000FF00;
    int8_128 x0 = as_int(x) & 0x000000FF;
    return as_float(x0 | x1);
}

inline void push_1pgx_int8(SIM_X86::tensor B0, int mask, int which, int h, int w) {
    int shift_num = which * 8;
    h = (h + 1) / 2;
    int last_awn = (h + 7) / 8;
    for (int i = 8; i >= last_awn; i--) {
        packed_push(v_u32_move_f(0.0), 0, 0);
    }
    if (1) {
        int i = last_awn - 1;
        int ih = min(h - i * 8, 8);
        int ldmk = pre_exp2(ih);
        float8_128 gain_pgx0 = load8_k(B0 + i * 32, 1, ldmk, w, 0);
        gain_pgx0 = as_float(v_u32_shr(as_int(gain_pgx0), shift_num) & mask);
        packed_push(gain_pgx0, 0, 0);
    }
    for (int i = last_awn - 2; i >= 0; i--) {
        float8_128 gain_pgx0 = load8_k(B0 + i * 32, 1, 255, w, 0);
        gain_pgx0 = as_float(v_u32_shr(as_int(gain_pgx0), shift_num) & mask);
        packed_push(gain_pgx0, 0, 0);
    }

}

//push lower bits 0-7 to PGX0 and lower bits 8-15 to PGX1
inline void push_2pgx_int8(SIM_X86::tensor B0, SIM_X86::tensor B1, int which) {
    float8_128 pgx0_gain_0 = load8_128_stride(32 * 7, 1, B0);
    float8_128 pgx1_gain_0 = load8_128_stride(32 * 7, 1, B1);
    float8_128 pgx0_gain_1 = load8_128_stride(32 * 6, 1, B0);
    float8_128 pgx1_gain_1 = load8_128_stride(32 * 6, 1, B1);
    float8_128 pgx0_gain_2 = load8_128_stride(32 * 5, 1, B0);
    float8_128 pgx1_gain_2 = load8_128_stride(32 * 5, 1, B1);
    float8_128 pgx0_gain_3 = load8_128_stride(32 * 4, 1, B0);
    float8_128 pgx1_gain_3 = load8_128_stride(32 * 4, 1, B1);

    pgx0_gain_0 = as_float(v_u32_shr(as_int(pgx0_gain_0), which * 8) & 0x00FF00FF);
    pgx1_gain_0 = as_float(v_u32_shr(as_int(pgx1_gain_0), which * 8) & 0x00FF00FF);
    pgx0_gain_1 = as_float(v_u32_shr(as_int(pgx0_gain_1), which * 8) & 0x00FF00FF);
    pgx1_gain_1 = as_float(v_u32_shr(as_int(pgx1_gain_1), which * 8) & 0x00FF00FF);
    pgx0_gain_2 = as_float(v_u32_shr(as_int(pgx0_gain_2), which * 8) & 0x00FF00FF);
    pgx1_gain_2 = as_float(v_u32_shr(as_int(pgx1_gain_2), which * 8) & 0x00FF00FF);
    pgx0_gain_3 = as_float(v_u32_shr(as_int(pgx0_gain_3), which * 8) & 0x00FF00FF);
    pgx1_gain_3 = as_float(v_u32_shr(as_int(pgx1_gain_3), which * 8) & 0x00FF00FF);

    packed_push(pgx0_gain_0, 0, 0);
    packed_push(pgx1_gain_0, 0, 1);
    packed_push(pgx0_gain_1, 0, 0);
    packed_push(pgx1_gain_1, 0, 1);
    packed_push(pgx0_gain_2, 0, 0);
    packed_push(pgx1_gain_2, 0, 1);
    packed_push(pgx0_gain_3, 0, 0);
    packed_push(pgx1_gain_3, 0, 1);

    pgx0_gain_0 = load8_128_stride(32 * 3, 1, B0);
    pgx1_gain_0 = load8_128_stride(32 * 3, 1, B1);
    pgx0_gain_1 = load8_128_stride(32 * 2, 1, B0);
    pgx1_gain_1 = load8_128_stride(32 * 2, 1, B1);
    pgx0_gain_2 = load8_128_stride(32 * 1, 1, B0);
    pgx1_gain_2 = load8_128_stride(32 * 1, 1, B1);
    pgx0_gain_3 = load8_128_stride(32 * 0, 1, B0);
    pgx1_gain_3 = load8_128_stride(32 * 0, 1, B1);

    pgx0_gain_0 = as_float(v_u32_shr(as_int(pgx0_gain_0), which * 8) & 0x00FF00FF);
    pgx1_gain_0 = as_float(v_u32_shr(as_int(pgx1_gain_0), which * 8) & 0x00FF00FF);
    pgx0_gain_1 = as_float(v_u32_shr(as_int(pgx0_gain_1), which * 8) & 0x00FF00FF);
    pgx1_gain_1 = as_float(v_u32_shr(as_int(pgx1_gain_1), which * 8) & 0x00FF00FF);
    pgx0_gain_2 = as_float(v_u32_shr(as_int(pgx0_gain_2), which * 8) & 0x00FF00FF);
    pgx1_gain_2 = as_float(v_u32_shr(as_int(pgx1_gain_2), which * 8) & 0x00FF00FF);
    pgx0_gain_3 = as_float(v_u32_shr(as_int(pgx0_gain_3), which * 8) & 0x00FF00FF);
    pgx1_gain_3 = as_float(v_u32_shr(as_int(pgx1_gain_3), which * 8) & 0x00FF00FF);


    packed_push(pgx0_gain_0, 0, 0);
    packed_push(pgx1_gain_0, 0, 1);
    packed_push(pgx0_gain_1, 0, 0);
    packed_push(pgx1_gain_1, 0, 1);
    packed_push(pgx0_gain_2, 0, 0);
    packed_push(pgx1_gain_2, 0, 1);
    packed_push(pgx0_gain_3, 0, 0);
    packed_push(pgx1_gain_3, 0, 1);

}

inline void matmul_gain_aw128_store_int8(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int int8_iaw, int ibw, int int_ibw, int add_src_flag) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int which = (iaw / 128) % 4;
    int res_num = (ibw % 512) / 128;
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x1 = transfer_left_vector((left0), which);
        m_matmul_int8_lo_single(x1, 0, 0);
    }
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x1 = transfer_left_vector((left0), which);
        m_matmul_int8_lo_single(x1, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);
        res0 = add_clamp(res0, ret0);
        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (res_num == 3) {
            float8_128 res0_2 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 12);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(res0, res0_2, res0_1, res0_0));
        }
        else if (res_num == 2) {
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(0, res0, res0_1, res0_0));
        }
        else if (res_num == 1) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8(res0, res0_lo));
        }
        else if (res_num == 0) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, res0);
        }
    }
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = add_clamp(res0, ret0);
        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (res_num == 3) {
            float8_128 res0_2 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 12);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(res0, res0_2, res0_1, res0_0));
        }
        else if (res_num == 2) {
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(0, res0, res0_1, res0_0));
        }
        else if (res_num == 1) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8(res0, res0_lo));
        }
        else if (res_num == 0) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, res0);
        }
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, mask);
        if (res_num == 3) {
            float8_128 res0_2 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 8);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 12);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, pack_i8_4(res0, res0_2, res0_1, res0_0));
        }
        else if (res_num == 2) {
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 8);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, pack_i8_4(0, res0, res0_1, res0_0));
        }
        else if (res_num == 1) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, pack_i8(res0, res0_lo));
        }
        else if (res_num == 0) {
            store8_128_stride_with_stmask(Coffset, 1, mask, D, res0);
        }
    }
}

inline void matmul_gain_aw128_int8(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int int8_iaw, int ibw, int int_ibw, int add_src_flag) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int which = (iaw / 128) % 4;
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x1 = transfer_left_vector((left0), which);
        m_matmul_int8_lo_single(x1, 0, 0);
    }
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x1 = transfer_left_vector((left0), which);
        m_matmul_int8_lo_single(x1, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);

        res0 = add_clamp(res0, ret0);
        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
    }
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, mask);
    }
}

inline void matmul_gain_aw256_store_int8(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int int8_iaw, int ibw, int int_ibw, int add_src_flag) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int which = ((iaw / 256) % 2) ? 2 : 0;
    int res_num = (ibw % 512) / 128;
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x0 = transfer_left_vector(left0, which);
        float8_128 x1 = transfer_left_vector(left0, which + 1);

        m_matmul_int8_lo_single(x0, 0, 0);
        m_matmul_int8_lo_single(x1, 0, 1);
    }
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x0 = transfer_left_vector(left0, which);
        float8_128 x1 = transfer_left_vector(left0, which + 1);
        m_matmul_int8_lo_single(x0, 0, 0);
        m_matmul_int8_lo_single(x1, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);


        ret0 = add_clamp(ret0, ret1);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (res_num == 3) {
            float8_128 res0_2 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 12);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(res0, res0_2, res0_1, res0_0));
        }
        else if (res_num == 2) {
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(0, res0, res0_1, res0_0));
        }
        else if (res_num == 1) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8(res0, res0_lo));
        }
        else if (res_num == 0) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, res0);
        }
    }
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        ret0 = add_clamp(ret0, ret1);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (res_num == 3) {
            float8_128 res0_2 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 12);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(res0, res0_2, res0_1, res0_0));
        }
        else if (res_num == 2) {
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 8);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8_4(0, res0, res0_1, res0_0));
        }
        else if (res_num == 1) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, pack_i8(res0, res0_lo));
        }
        else if (res_num == 0) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, res0);
        }
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        ret0 = add_clamp(ret0, ret1);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, mask);
        if (res_num == 3) {
            float8_128 res0_2 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 8);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 12);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, pack_i8_4(res0, res0_2, res0_1, res0_0));
        }
        else if (res_num == 2) {
            float8_128 res0_1 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            float8_128 res0_0 = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 8);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, pack_i8_4(0, res0, res0_1, res0_0));
        }
        else if (res_num == 1) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, pack_i8(res0, res0_lo));
        }
        else if (res_num == 0) {
            store8_128_stride_with_stmask(Coffset, 1, mask, D, res0);
        }
    }
}

inline void matmul_gain_aw256_int8(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int int8_iaw, int ibw, int int_ibw, int add_src_flag) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int which = ((iaw / 256) % 2) ? 2 : 0;
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x0 = transfer_left_vector(left0, which);
        float8_128 x1 = transfer_left_vector(left0, which + 1);

        m_matmul_int8_lo_single(x0, 0, 0);
        m_matmul_int8_lo_single(x1, 0, 1);
    }
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        int AoffsetPgx0 = ah * int8_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        float8_128 x0 = transfer_left_vector(left0, which);
        float8_128 x1 = transfer_left_vector(left0, which + 1);
        m_matmul_int8_lo_single(x0, 0, 0);
        m_matmul_int8_lo_single(x1, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        ret0 = add_clamp(ret0, ret1);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
    }
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        ret0 = add_clamp(ret0, ret1);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        ret0 = add_clamp(ret0, ret1);
        res0 = add_clamp(res0, ret0);

        store8_128_stride_stmk(Coffset, stride, C, res0, mask);
    }
}

inline void matmul_int8(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int add_src_flag) {
    int aw256 = aw & 0xffffff00;
    int RHS_offset = 0;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0, int8_iaw = 0;
        if (ibw % 512 == 0 && ibw != 0) {
            D = D + ah * 4;
        }
        for (; iaw < aw256; iaw += 256) {
            if (iaw % 512 == 0 && iaw != 0) int8_iaw += 128;
            push_2pgx_int8(B + iaw * 128 / 64 + RHS_offset * ALIGN128(aw) / 64, B + iaw * 128 / 64 + 2 * 128 + RHS_offset * ALIGN128(aw) / 64, (ibw / 128) % 2);
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            if (iaw == aw - 256) matmul_gain_aw256_store_int8(A, C, D, ah, aw, bw, iaw, int8_iaw, ibw, 0, add_src_flag);
            else matmul_gain_aw256_int8(A, C, C, ah, aw, bw, iaw, int8_iaw, ibw, 0, add_src_flag);
        }
        for (;iaw < aw; iaw += 128) {
            push_1pgx_int8(B + iaw * 128 / 64 + RHS_offset * ALIGN128(aw) / 64, 0x00FF00FF, (ibw / 128) % 2, min(aw - iaw, 128), min(bw - ibw, 128));
            m_fakemul(v_u32_move_b(0), 0, 0);
            if (ibw == ALIGN128(bw) - 128 || ((ibw % 512) == 384)) matmul_gain_aw128_store_int8(A, C, D, ah, aw, bw, iaw, int8_iaw, ibw, 0, add_src_flag);
            else matmul_gain_aw128_int8(A, C, C, ah, aw, bw, iaw, int8_iaw, ibw, 0, add_src_flag);
        }
        C += ah * 4;
        if (ibw % 256) RHS_offset += 128;
    }
}


inline void push_gain_2pgx_RHS_ldstride(SIM_X86::tensor B0, SIM_X86::tensor B1, int bw, float scale){
    bw = ALIGN128(bw);
    int stride = bw / 128;
    int once_offset = 8 * bw / 32;
    float8_128 pgx0_gain_0 = load8_128_stride(15 * once_offset, stride, B0);
    float8_128 pgx1_gain_0 = load8_128_stride(15 * once_offset, stride, B1);
    float8_128 pgx0_gain_1 = load8_128_stride(14 * once_offset, stride, B0);
    float8_128 pgx1_gain_1 = load8_128_stride(14 * once_offset, stride, B1);
    float8_128 pgx0_gain_2 = load8_128_stride(13 * once_offset, stride, B0);
    float8_128 pgx1_gain_2 = load8_128_stride(13 * once_offset, stride, B1);
    float8_128 pgx0_gain_3 = load8_128_stride(12 * once_offset, stride, B0);
    float8_128 pgx1_gain_3 = load8_128_stride(12 * once_offset, stride, B1);
    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;
    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);
    pgx0_gain_0 = load8_128_stride(11 * once_offset, stride, B0);
    pgx1_gain_0 = load8_128_stride(11 * once_offset, stride, B1);
    pgx0_gain_1 = load8_128_stride(10 * once_offset, stride, B0);
    pgx1_gain_1 = load8_128_stride(10 * once_offset, stride, B1);
    pgx0_gain_2 = load8_128_stride(9 * once_offset, stride, B0);
    pgx1_gain_2 = load8_128_stride(9 * once_offset, stride, B1);
    pgx0_gain_3 = load8_128_stride(8 * once_offset, stride, B0);
    pgx1_gain_3 = load8_128_stride(8 * once_offset, stride, B1);
    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;
    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);
    pgx0_gain_0 = load8_128_stride(7 * once_offset, stride, B0);
    pgx1_gain_0 = load8_128_stride(7 * once_offset, stride, B1);
    pgx0_gain_1 = load8_128_stride(6 * once_offset, stride, B0);
    pgx1_gain_1 = load8_128_stride(6 * once_offset, stride, B1);
    pgx0_gain_2 = load8_128_stride(5 * once_offset, stride, B0);
    pgx1_gain_2 = load8_128_stride(5 * once_offset, stride, B1);
    pgx0_gain_3 = load8_128_stride(4 * once_offset, stride, B0);
    pgx1_gain_3 = load8_128_stride(4 * once_offset, stride, B1);
    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;
    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);
    
    pgx0_gain_0 = load8_128_stride(3 * once_offset, stride, B0);
    pgx1_gain_0 = load8_128_stride(3 * once_offset, stride, B1);
    pgx0_gain_1 = load8_128_stride(2 * once_offset, stride, B0);
    pgx1_gain_1 = load8_128_stride(2 * once_offset, stride, B1);
    pgx0_gain_2 = load8_128_stride(1 * once_offset, stride, B0);
    pgx1_gain_2 = load8_128_stride(1 * once_offset, stride, B1);
    pgx0_gain_3 = load8_128_stride(0 * once_offset, stride, B0);
    pgx1_gain_3 = load8_128_stride(0 * once_offset, stride, B1);
    pgx0_gain_0 = pgx0_gain_0 * scale;
    pgx1_gain_0 = pgx1_gain_0 * scale;
    pgx0_gain_1 = pgx0_gain_1 * scale;
    pgx1_gain_1 = pgx1_gain_1 * scale;
    pgx0_gain_2 = pgx0_gain_2 * scale;
    pgx1_gain_2 = pgx1_gain_2 * scale;
    pgx0_gain_3 = pgx0_gain_3 * scale;
    pgx1_gain_3 = pgx1_gain_3 * scale;
    
    push_gsnf(pgx0_gain_0, 0);
    push_gsnf(pgx1_gain_0, 1);
    push_gsnf(pgx0_gain_1, 0);
    push_gsnf(pgx1_gain_1, 1);
    push_gsnf(pgx0_gain_2, 0);
    push_gsnf(pgx1_gain_2, 1);
    push_gsnf(pgx0_gain_3, 0);
    push_gsnf(pgx1_gain_3, 1);
}
//将右矩阵的128*128push进gsnf里。
inline void push_aw256_f32_RHS_ldstride(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, float scale){
    int BoffsetPgx0 = (iaw * ALIGN128(bw) + ibw) / 32;
    int BoffsetPgx1 = BoffsetPgx0 + ALIGN128(bw) * 128 / 32;
    push_gain_2pgx_RHS_ldstride(B + BoffsetPgx0, B + BoffsetPgx1, bw, scale);
}
inline void push_aw128_f32_stride(SIM_X86::tensor B, int aw, int bw, int iaw, int ibw, float scale) {
    int cur_awn = min((aw - iaw + 7), 128) / 8;
    int bw128 = ALIGN128(bw);
    int stride = bw128 / 128;
    int BoffsetPgx0 = (iaw * ALIGN128(bw) + ibw) / 32;
    int once_offset = 8 * bw128 / 32;
    float8_128 zero = 0;
    for (int i = 15; i >= cur_awn; i--) {
        push_gsnf(zero, 0);
    }
    if (1) {
        int i = cur_awn - 1;
        int h = min((aw - iaw) - i * 8, 8);
        int mask = pre_exp2(h);
        float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + i * once_offset, stride, mask, min(bw - ibw, 128), 0);
        push_gsnf(gain_pgx0 * scale, 0);
    }
    for (int i = cur_awn - 2; i >= 0; i--) {
        float8_128 gain_pgx0 = load8_k(B + BoffsetPgx0 + i * once_offset, stride, 255, min(bw - ibw, 128), 0);
        push_gsnf(gain_pgx0 * scale, 0);
    }
}
inline void matmul_all_f32_RHS_ldstride(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_f32_RHS_ldstride(B, aw, bw, iaw, ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            if (Lscale == 1.0) matmul_LHS_aw256_pipeline_f32(A, C, ah, bw, iaw, ibw, add_src_flag);
            else matmul_LHS_aw256_f32(A, C, ah, bw, iaw, ibw, add_src_flag, Lscale);
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_f32_stride(B, aw, bw, iaw, ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            matmul_gain_opt_rest(A, C, ah, aw, bw, iaw, ibw, ah, add_src_flag, Lscale);
        }
        C = C + ah * 4;
    }
}


inline void load_mat_0123_h_with_sync(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
    int idx2, int idx3, int vmemH, int vmemW) {
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    for (int i = 0; i < vmemW; i += 128) {
        dlc_sync(dlc_dma(tensor_slice(src, (offset + i) / 32), HBM, tensor_slice(dst, i * vmemH / 32), VMEM,
            vmemH * 128, dim3, 128, 128, 7));
    }
}

inline void store_mat_0123_h_with_sync(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
    int idx2, int idx3, int vmemH, int vmemW) {
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    int len = vmemH * 128;
    for (int i = 0; i < ALIGN128(vmemW); i += 128) {
        dlc_sync(dlc_dma(src + i * vmemH / 32, VMEM, dst + (i + offset) / 32, HBM, len, 128, dim3, 128, 7));
    }
}

inline int load_mat_0123_h_without_sync(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
    int idx2, int idx3, int vmemH, int vmemW) {
        const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    int i = 0;
    for (; i < vmemW - 128; i += 128) {
        dlc_dma(tensor_slice(src, (offset + i) / 32), HBM, tensor_slice(dst, i * vmemH / 32), VMEM,
            vmemH * 128, dim3, 128, 128, 7);
    }
    return dlc_dma(tensor_slice(src, (offset + i) / 32), HBM, tensor_slice(dst, i * vmemH / 32), VMEM,
        vmemH * 128, dim3, 128, 128, 7);
}

inline int store_mat_0123_h_without_sync(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
    int idx2, int idx3, int vmemH, int vmemW) {
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    int len = vmemH * 128;
    int i = 0;
    for (; i + 128 < ALIGN128(vmemW); i += 128) {
        dlc_dma(src + i * vmemH / 32, VMEM, dst + (i + offset) / 32, HBM, len, 128, dim3, 128, 7);
    }
    return dlc_dma(src + i * vmemH / 32, VMEM, dst + (i + offset) / 32, HBM, len, 128, dim3, 128, 7);
}

//pingpong version of matmul_all_bf16
//usage:sync = matmul_all_bf16_pingpong(input0_hbm, input1_hbm, output_hbm, input0, input1, output,
//          process_ah, process_aw, process_bw, AH, AW, BW, i, j, k, aw + k >= AW, 1.0, 1.0);
//then dlc_sync(sync); before program's end.
//No other DMAs outside anymore, in the function it will load from HBM each 128 columns an iteration.
inline int matmul_all_bf16_pingpong(SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0, SIM_X86::tensor input1, SIM_X86::tensor output,
    int process_ah, int process_aw, int process_bw, int AH, int AW, int BW, int i, int j, int k, int is_last_aw, float Lscale, float Rscale) {
    int aw256 = process_aw & 0xffffff00;
    int bf_ibw = 0;
    SIM_X86::tensor output_ptr = output;
    SIM_X86::tensor output_tmp_ptr = output;
    int bf_AW = ALIGN256(AW) / 2;
    int bf_BW = ALIGN256(BW) / 2;
    int bf_k = ALIGN256(k) / 2;
    int bf_j = ALIGN256(j) / 2;
    int sync_RHS = DONE, sync_LHS_256 = DONE, sync_LHS_128 = DONE, sync_out = DONE;
    for (int ibw = 0; ibw < process_bw; ibw += 128) {
        int iaw = 0;
        int bf_iaw = 0;
        if (ibw == 0) {
            load_mat_0123_h_with_sync(input1_hbm, input1 + (bf_ibw * process_aw) / 32, 1, 1, AW, bf_BW, 0, 0, k, bf_j + bf_ibw,
                process_aw, 128);
        }
        if ((ibw + 128) < process_bw && ibw % 256 == 0) {
            dlc_sync(sync_RHS);
            int next_bf_ibw = bf_ibw + 128;
            sync_RHS = load_mat_0123_h_without_sync(input1_hbm, input1 + (next_bf_ibw * process_aw) / 32, 1, 1, AW, bf_BW, 0, 0, k, bf_j + next_bf_ibw,
                process_aw, 128);
        }
        else if (ibw + 128 >= process_bw) {
            dlc_sync(sync_RHS);
        }
        if (ibw == 0) {
            for (; iaw < aw256; iaw += 256) {
                if (iaw == 0) {
                    load_mat_0123_h_with_sync(input0_hbm, input0 + (bf_iaw * process_ah) / 32, 1, 1, AH, bf_AW, 0, 0, i, bf_k + bf_iaw,
                        process_ah, 128);
                }
                if ((ibw + 128) < process_bw && ibw % 256 == 0) {
                    dlc_sync(sync_LHS_256);
                    int next_bf_iaw = bf_iaw + 128;
                    sync_LHS_256 = load_mat_0123_h_without_sync(input0_hbm, input0 + (next_bf_iaw * process_ah) / 32, 1, 1, AH, bf_AW, 0, 0, i, bf_k + next_bf_iaw,
                        process_ah, 128);
                }
                else if (ibw + 128 >= process_bw) {
                    dlc_sync(sync_LHS_256);
                }
                push_aw256_bf16(input1, process_aw, iaw, ibw, bf_ibw, Rscale);
                //load datas in gain to GMR
                m_fakemul(v_u32_move_b(0), 0, 0);
                m_fakemul(v_u32_move_b(0), 0, 1);
                matmul_LHS_aw256_bf16(input0, output_ptr, output_tmp_ptr, process_ah, process_aw, process_bw, iaw, bf_iaw, ibw, k, is_last_aw, Lscale);
                bf_iaw += 128;
            }
            for (;iaw < process_aw; iaw += 128) {
                if (iaw == aw256) {
                    load_mat_0123_h_with_sync(input0_hbm, input0 + (bf_iaw * process_ah) / 32, 1, 1, AH, bf_AW, 0, 0, i, bf_k + bf_iaw,
                        process_ah, 128);
                    if ((ibw + 128) < process_bw && ibw % 256 == 0) {
                        dlc_sync(sync_LHS_128);
                        int next_bf_iaw = bf_iaw + 128;
                        sync_LHS_128 = load_mat_0123_h_without_sync(input0_hbm, input0 + (next_bf_iaw * process_ah) / 32, 1, 1, AH, bf_AW, 0, 0, i, bf_k + next_bf_iaw,
                            process_ah, 128);
                    }
                    else if (ibw + 128 >= process_bw) {
                        dlc_sync(sync_LHS_128);
                    }
                }
                push_aw128_bf16(input1, process_aw, process_bw, iaw, ibw, bf_ibw, Rscale);
                //load datas in gain to GMR
                m_fakemul(v_u32_move_b(0), 0, 0);
                matmul_LHS_aw128_bf16(input0, output_ptr, output_tmp_ptr, process_ah, process_aw, process_bw, iaw, bf_iaw, ibw, bf_ibw, k, is_last_aw, Lscale);
            }
        }
        else {
            for (; iaw < aw256; iaw += 256) {
                push_aw256_bf16(input1, process_aw, iaw, ibw, bf_ibw, Rscale);
                //load datas in gain to GMR
                m_fakemul(v_u32_move_b(0), 0, 0);
                m_fakemul(v_u32_move_b(0), 0, 1);
                matmul_LHS_aw256_bf16(input0, output_ptr, output_tmp_ptr, process_ah, process_aw, process_bw, iaw, bf_iaw, ibw, k, is_last_aw, Lscale);
                bf_iaw += 128;
            }
            for (;iaw < process_aw; iaw += 128) {
                push_aw128_bf16(input1, process_aw, process_bw, iaw, ibw, bf_ibw, Rscale);
                //load datas in gain to GMR
                m_fakemul(v_u32_move_b(0), 0, 0);
                matmul_LHS_aw128_bf16(input0, output_ptr, output_tmp_ptr, process_ah, process_aw, process_bw, iaw, bf_iaw, ibw, bf_ibw, k, is_last_aw, Lscale);
            }
        }
        if ((ibw % 256 || ibw + 128 >= process_bw) && is_last_aw) {
            dlc_sync(sync_out);
            sync_out = store_mat_0123_h_without_sync(output_tmp_ptr, output_hbm, 1, 1, AH, bf_BW, 0, 0, i, bf_j + bf_ibw, process_ah, 128);
        }

        output_ptr = output_ptr + process_ah * 4;
        if (ibw % 256) {
            bf_ibw += 128;
            output_tmp_ptr = output_tmp_ptr + process_ah * 4;
        }
    }
    return sync_out;
}

inline void matmul_gain_2pgx_bf16_no_extra_space(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag, float8_128 scale) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int bw128 = ALIGN128(bw);
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_128_stride(AoffsetPgx0, 1, A);
        short8_128 x1 = unpack_16b(as_int(left0), 0);
        short8_128 x2 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
        m_matmul_single(bfloat16_to_float(x2) * scale, 0, 1);
    }

#pragma unroll
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);

        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_128_stride(AoffsetPgx0, 1, A);
        short8_128 x1 = unpack_16b(as_int(left0), 0);
        short8_128 x2 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
        m_matmul_single(bfloat16_to_float(x2) * scale, 0, 1);

        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);

        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, 1, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res)));
        }
    }

#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, 1, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res)));
        }
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        float8_128 ret1 = m_pop_mrf(1);
        res = res + ret0;
        res = res + ret1;
        store8_128_stride_stmk(Coffset, stride, C, res, mask);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, 1, mask, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(res, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(v_u32_move_f(0), res)));
        }
    }
}

inline void matmul_gain_1pgx_bf16_no_extra_space(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int bf_ibw, int add_src_flag, float8_128 scale) {
    int n = (ah + 7) / 8;
    int m = min(12, n);
    int stride = 1;
    int bw128 = ALIGN128(bw);
    int is_hi = (iaw / 128) % 2;
#pragma unroll
    for (int i = 0; i < m; i++) {
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        short8_128 x1;
        if (is_hi == 0) x1 = unpack_16b(as_int(left0), 0);
        else x1 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);
    }
#pragma unroll
    for (int i = 12; i < n; i++) {
        int Coffset = (i - 12) * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        int AoffsetPgx0 = ah * bf_iaw / 32 + i * 32;
        float8_128 left0 = load8_k(A + AoffsetPgx0, 1, 255, min(aw - iaw, 128), 0);
        short8_128 x1;
        if (is_hi == 0) x1 = unpack_16b(as_int(left0), 0);
        else x1 = unpack_16b(as_int(left0), 1);
        m_matmul_single(bfloat16_to_float(x1) * scale, 0, 0);

        float8_128 ret0 = m_pop_mrf(0);

        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
        }
    }
#pragma unroll
    for (int i = n - m; i < n - 1; i++) {
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, 255, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, 255);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, 255, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(res0, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, 255, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
        }
    }
    if (m != 0) {
        int i = n - 1;
        int h = min(ah - i * 8, 8);
        int mask = pre_exp2(h);
        int Coffset = i * 32;
        float8_128 res0 = (iaw == 0 && add_src_flag == 0)
            ? v_u32_move_f(0.0)
            : load8_128_stride_with_ldmask(Coffset, stride, mask, C);
        float8_128 ret0 = m_pop_mrf(0);
        res0 = res0 + ret0;

        store8_128_stride_stmk(Coffset, stride, C, res0, mask);
        if (ibw % 256) {
            float8_128 res0_lo = load8_128_stride_with_ldmask(Coffset, stride, mask, C - ah * 4);
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(res0, res0_lo)));
        }
        else if (ibw == bw128 - 128) {
            store8_128_stride_with_stmask(Coffset, 1, mask, D, as_float(float_to_bfloat16(v_u32_move_f(0), res0)));
        }
    }
}

inline void matmul_LHS_aw256_bf16_no_extra_space(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int add_src_flag, int is_last_aw, float8_128 scale) {
    if (is_last_aw && iaw + 256 >= aw)
        matmul_gain_2pgx_no_pack_opt_store_rest_bf16(A, C, D, ah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, scale);
        // matmul_gain_2pgx_bf16_no_extra_space(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, scale);
    else
        matmul_gain_2pgx_no_pack_opt_rest_bf16(A, C, D, ah, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, scale);
}

inline void matmul_LHS_aw128_bf16_no_extra_space(SIM_X86::tensor A, SIM_X86::tensor C, SIM_X86::tensor D, int ah, int aw, int bw, int iaw, int bf_iaw, int ibw, int bf_ibw, int add_src_flag, int is_last_aw, float8_128 scale) {
    if (is_last_aw && iaw + 128 >= aw)
        matmul_gain_no_pack_opt_store_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, scale);
        // matmul_gain_1pgx_bf16_no_extra_space(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, scale);
    else
        matmul_gain_no_pack_opt_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, scale);
}

inline void matmul_all_bf16_no_extra_space(SIM_X86::tensor A, SIM_X86::tensor B, SIM_X86::tensor C, int ah, int aw, int bw, int add_src_flag, int is_last_aw, float Lscale, float Rscale) {
    int aw256 = aw & 0xffffff00;
    int bf_ibw = 0;
    SIM_X86::tensor D = C;
    for (int ibw = 0; ibw < bw; ibw += 128) {
        int iaw = 0;
        int bf_iaw = 0;
        for (; iaw < aw256; iaw += 256) {
            push_aw256_bf16(B, aw, iaw, ibw, bf_ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            m_fakemul(v_u32_move_b(0), 0, 1);
            matmul_LHS_aw256_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, add_src_flag, is_last_aw, Lscale);
            bf_iaw += 128;
        }
        for (;iaw < aw; iaw += 128) {
            push_aw128_bf16(B, aw, bw, iaw, ibw, bf_ibw, Rscale);
            //load datas in gain to GMR
            m_fakemul(v_u32_move_b(0), 0, 0);
            matmul_LHS_aw128_bf16(A, C, D, ah, aw, bw, iaw, bf_iaw, ibw, bf_ibw, add_src_flag, is_last_aw, Lscale);
        }
        C = C + ah * 4;
        if (ibw % 256) {
            bf_ibw += 128;
            D = D + ah * 4;
        }
    }
}
#endif
