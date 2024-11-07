#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "align.h"
#include "bf16.h"
#include "dma.h"
#include "ldst.h"


inline int padding128(int a) { return (a + 127) & 0xffffff80; }
inline int padding256(int a) { return (a + 255) & 0xffffff00; }
inline int _max(int a, int b) { return (a > b) ? a : b; }

inline float8_128 load8_k(SIM_X86::tensor t, int st, int ldmk, int w) { return load8_128_stride_ldmk(0, st, t, ldmk); }

inline float8_128 loadmin8_k(SIM_X86::tensor t, int st, int h, int ldmk, int w) {
    float8_128 v = load8_128_stride_ldmk(0, st, t, ldmk);
    return v;
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

inline float128_128 loadh_k(SIM_X86::tensor t, int st, int h, int w) {
    float8_128 data0 = v_u32_move_f(0.0);
    float8_128 data1 = v_u32_move_f(0.0);
    float8_128 data2 = v_u32_move_f(0.0);
    float8_128 data3 = v_u32_move_f(0.0);
    float8_128 data4 = v_u32_move_f(0.0);
    float8_128 data5 = v_u32_move_f(0.0);
    float8_128 data6 = v_u32_move_f(0.0);
    float8_128 data7 = v_u32_move_f(0.0);
    float8_128 data8 = v_u32_move_f(0.0);
    float8_128 data9 = v_u32_move_f(0.0);
    float8_128 data10 = v_u32_move_f(0.0);
    float8_128 data11 = v_u32_move_f(0.0);
    float8_128 data12 = v_u32_move_f(0.0);
    float8_128 data13 = v_u32_move_f(0.0);
    float8_128 data14 = v_u32_move_f(0.0);
    float8_128 data15 = v_u32_move_f(0.0);
    int nh = (h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        const int i = (x) * 8;                                                                               \
        const int cur_h = min(h - i, 8);                                                                     \
        const int ldmk = (1 << cur_h) - 1;                                                                   \
        data##x = loadmin8_k(tensor_slice(t, i * st * 128 / 32), st, cur_h, ldmk, w);                        \
    }
    SWITCH_CASES_REV(nh)
#undef CASE_ITEM
    return v_concat_16(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11,
                       data12, data13, data14, data15);
}

inline float128_128 loadh_k_with_stride(SIM_X86::tensor src, int _h, int _w, int h, int input0_stride) {
    float8_128 data0 = v_u32_move_f(0.0);
    float8_128 data1 = v_u32_move_f(0.0);
    float8_128 data2 = v_u32_move_f(0.0);
    float8_128 data3 = v_u32_move_f(0.0);
    float8_128 data4 = v_u32_move_f(0.0);
    float8_128 data5 = v_u32_move_f(0.0);
    float8_128 data6 = v_u32_move_f(0.0);
    float8_128 data7 = v_u32_move_f(0.0);
    float8_128 data8 = v_u32_move_f(0.0);
    float8_128 data9 = v_u32_move_f(0.0);
    float8_128 data10 = v_u32_move_f(0.0);
    float8_128 data11 = v_u32_move_f(0.0);
    float8_128 data12 = v_u32_move_f(0.0);
    float8_128 data13 = v_u32_move_f(0.0);
    float8_128 data14 = v_u32_move_f(0.0);
    float8_128 data15 = v_u32_move_f(0.0);
    int nh = (h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        int cur_h = min(h - (x) * 8, 8);                                                                     \
        int offset = _h * input0_stride / 32 + _w / 32 + (x) * 8 * input0_stride / 32;                       \
        data##x = load8_128_stride_ldmk(0, input0_stride / 128, src + offset, (1 << cur_h) - 1);             \
    } 
    SWITCH_CASES_REV(nh)
#undef CASE_ITEM
    return v_concat_16(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11,
                       data12, data13, data14, data15);
}

// [h, padding_w]
inline float128_128 loadh_k_T(SIM_X86::tensor t, int padding_h, int h, int padding_w, int w, int i, int j) {
    int addr = 128 * j + i * 128 * padding_w;
    int st = padding_w / 128;
    int cur_w = min(w - 128 * j, 128);
    int cur_h = min(h - 128 * i, 128);

    float8_128 __attribute__((address_space(2))) data[16];
    
    int _cur_h = min(cur_h, 8);
    int ldmk = (1 << _cur_h) - 1;
    data[0] = loadmin8_k(tensor_slice(t, addr / 32), st, cur_h, ldmk, cur_w);
    m_transpose_start(data[0], 128, 0);
    
    int nh = (cur_h + 7) / 8;
    for(int index = 1; index < nh - 1; index++) {
        int _index = index * 8;
        _cur_h = min(cur_h - _index, 8);
        ldmk = (1 << _cur_h) - 1;
        data[index] = loadmin8_k(tensor_slice(t, (addr + _index * st * 128) / 32), st, cur_h, ldmk, cur_w);
        m_transpose_mid(data[index], 0);
    }

    _cur_h = min(_max(cur_h - (nh - 1) * 8, 0), 8);
    ldmk = (1 << _cur_h) - 1;
    data[15] = loadmin8_k(tensor_slice(t, (addr + (nh - 1) * 8 * st * 128) / 32), st, cur_h, ldmk, cur_w);
    m_transpose_end(data[15], 0);

    float8_128 __attribute__((address_space(2))) res[16];
    for(int i = 0; i < 16; i++) {
        res[i] = m_pop_trf(0);
    }

    return v_concat_16(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], 
                       res[10], res[11], res[12], res[13], res[14], res[15]);
}

inline void store128_128_ex_permute(SIM_X86::tensor t, int h, int w, int ih, int iw, float128_128 v) {
    int pw = (w + 127) & 0xffffff80;
//     int cur_w = min(w - iw * 128, 128);
    int cur_h = min(h - ih * 128, 128);
    int kS = (cur_h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        int i = (x);                                                                                         \
        int cur_sth = min(cur_h - i * 8, 8);                                                                 \
        store8_128_stride_stmk(((ih * 128 + i * 8) * pw + iw * 128) / 32, pw / 128, t, sub_vector(v, x),     \
                               (1 << cur_sth) - 1);                                                          \
    }
    SWITCH_CASES_REV(kS)
#undef CASE_ITEM
}

inline void store128_128_ex_permute_with_stride2(SIM_X86::tensor t, int h, int w, int ih, int iw, int stride_dst, float128_128 v) {
//     int pw = (w + 127) & 0xffffff80;
//     int cur_w = min(w - iw * 128, 128);
    int cur_h = min(h - ih * 128, 128);
    int kS = (cur_h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        int i = (x);                                                                                         \
        int cur_sth = min(cur_h - i * 8, 8);                                                                 \
        store8_128_stride_stmk(((ih * 128 + i * 8) * stride_dst + iw * 128) / 32, stride_dst / 128,          \
                               t, sub_vector(v, x), (1 << cur_sth) - 1);                                     \
    }
    SWITCH_CASES_REV(kS)
#undef CASE_ITEM
}

inline void store128_128_ex_permute_with_stride(SIM_X86::tensor dst, int _h, int _w, int h, int output_stride, float128_128 v) {
    int kS = (h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        int cur_h = min(h - (x) * 8, 8);                                                                     \
        int offset = _h * output_stride / 32 + _w / 32 + (x) * 8 * output_stride / 32;                       \
        store8_128_stride_stmk(0, output_stride / 128, dst + offset, sub_vector(v, x), (1 << cur_h) - 1);    \
    }
    SWITCH_CASES_REV(kS)
#undef CASE_ITEM
}

inline void tile_transfer(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int height, int width, int dstaddr) {
    for (int i = 0; i < height; i += 8) {
        int cur_h = min(height - i, 8);
        for (int j = 0; j < width; j += 128) {
            float8_128 d =
                load8_128_stride_ldmk((srcaddr + j + i * width) / 32, width / 128, src, (1 << cur_h) - 1);
            store8_128_stride_stmk((dstaddr + j + i * width) / 32, width / 128, dst, d, (1 << cur_h) - 1);
        }
    }
}

inline void pop_nws0_store128_128_ex_permute(SIM_X86::tensor t, int h, int w, int ih, int iw) {
    int pw = (w + 127) & 0xffffff80;
//     int cur_w = min(w - iw * 128, 128);
    int cur_h = min(h - ih * 128, 128);
    int kS = (cur_h + 7) / 8;
    for(int i = 0; i < kS; i++) {
        int cur_sth = min(cur_h - i * 8, 8);
        [[maybe_unused]]float8_128 x = m_pop_trf(0);
        store8_128_stride_stmk(((ih * 128 + i * 8) * pw + iw * 128) / 32, pw / 128, t, x, (1 << cur_sth) - 1);   
    }
    for(int i = kS; i < 16; i++) {
        __attribute__((unused)) float8_128 x = m_pop_trf(0);
    }
}

inline void pop_nws1_store128_128_ex_permute(SIM_X86::tensor t, int h, int w, int ih, int iw) {
    int pw = (w + 127) & 0xffffff80;
//     int cur_w = min(w - iw * 128, 128);
    int cur_h = min(h - ih * 128, 128);
    int kS = (cur_h + 7) / 8;
    for(int i = 0; i < kS; i++) {
        int cur_sth = min(cur_h - i * 8, 8);
        store8_128_stride_stmk(((ih * 128 + i * 8) * pw + iw * 128) / 32, pw / 128, t, m_pop_trf(1), (1 << cur_sth) - 1);   
    }
    for(int i = kS; i < 16; i++) {
        __attribute__((unused))float8_128 x = m_pop_trf(1);
    }
}

// inline void tile_trans_transfer(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr) {
//     int paddingH = padding128(src_h);
//     int paddingW = padding128(src_w);
//     for (int i = 0; i < src_h; i += 128) {
//         for (int j = 0; j < src_w; j += 128) {
//             float128_128 v = loadh_k_T(tensor_slice(src, srcaddr / 32), paddingH, src_h, paddingW, src_w,
//                                        i / 128, j / 128);
//             store128_128_ex_permute(tensor_slice(dst, dstaddr / 32), src_w, src_h, j / 128, i / 128, v);
//         }
//     }
// }

inline void tile_trans_transfer(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr) {
    int paddingH = padding128(src_h);
    int paddingW = padding128(src_w);
    int i = 0;
    for (; i + 128 <= src_h; i += 128) {
        int j = 0;
        for (; j + 128 < src_w; j += 256) {
            int addr0 = j + i * paddingW;
            int addr1 = j + 128 + i * paddingW;
            int st = paddingW / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];
            
            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_start(data_b[0], 128, 1);
            
            for(int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), 255);
                m_transpose_mid(data_b[i], 1);
            }
        
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 120 * st * 128 / 32), 255);
            m_transpose_end(data_b[15], 1);
            
            int store_addr0 = j * paddingH + i;
            int store_addr1 = (j + 128) * paddingH + i;
            int store_st = paddingH / 128;
            
            int cur_h = min(src_w - j - 128, 128);
            int kS = (cur_h + 7) / 8;
            #pragma clang loop unroll_count(2)
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, 255);   
                float8_128 x1 = m_pop_trf(1);
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr1) / 32), x1, (1 << cur_sth) - 1);   
            }
            #pragma clang loop unroll_count(2)
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x1 = m_pop_trf(1);
                float8_128 x = m_pop_trf(0);
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x, 255);   
            }
        }
        int tail_j = src_w - j;
        if(tail_j <= 0) continue;
        int addr0 = j + i * paddingW;
        int st = paddingW / 128;
        SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
        float8_128 data_a[16];
        
        data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
        m_transpose_start(data_a[0], 128, 0);
        
        for(int ii = 1; ii < 15; ii++) {
            int _i = ii * 8;
            data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
            m_transpose_mid(data_a[i], 0);
        }
    
        data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
        m_transpose_end(data_a[15], 0);
        
        int store_addr0 = j * paddingH + i;
        int store_st = paddingH / 128;
        
        int cur_h = min(src_w - j, 128);
        int kS = (cur_h + 7) / 8;
        #pragma clang loop unroll_count(2)
        for(int i = 0; i < kS; i++) {
            int cur_sth = min(cur_h - i * 8, 8);
            float8_128 x0 = m_pop_trf(0);
            store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, (1 << cur_sth) - 1);   
        }
        #pragma clang loop unroll_count(2)
        for(int i = kS; i < 16; i++) {
            __attribute__((unused))float8_128 x = m_pop_trf(0);
        }
    }
    if(src_h - i <= 0) return;
    for (int j = 0; j < src_w; j += 128) {
        float128_128 v = loadh_k_T(tensor_slice(src, srcaddr / 32), paddingH, src_h, paddingW, src_w, i / 128, j / 128);
        store128_128_ex_permute(tensor_slice(dst, dstaddr / 32), src_w, src_h, j / 128, i / 128, v);
    }
}

// e.g. [4, (3, ..., 333)] => [(3, ..., 333), 4]
// src_h = 4
// mid = 3 * ...
// last_w = 333
inline void tile_trans_transfer_mid(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int mid, int last_w, int dstaddr) {
    int src_w = padding128(last_w) * mid;
    int paddingH = padding128(src_h);
    int paddingW = padding128(src_w);
    int lastH =  last_w % 128;
    int mid_num = (last_w - 1) / 128;
    if((last_w & 127) == 0) lastH = 128;
    int i = 0;
    for (; i + 128 <= src_h; i += 128) {
        int storedCountH = 0;
        int j = 0;
        for (; j + 128 < src_w; j += 256) {
            int addr0 = j + i * paddingW;
            int addr1 = j + 128 + i * paddingW;
            int st = paddingW / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];
            
            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_start(data_b[0], 128, 1);
            
            for(int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), 255);
                m_transpose_mid(data_b[i], 1);
            }
        
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 120 * st * 128 / 32), 255);
            m_transpose_end(data_b[15], 1);
            
            int store_addr0 = storedCountH * paddingH + i;
            int cur_h0;
            if(mid_num){
                cur_h0 = 128;
                mid_num--;
            }
            else{
                cur_h0 = lastH;
                mid_num = (last_w - 1) / 128;
            }
            storedCountH += cur_h0;
            int store_addr1 = storedCountH * paddingH + i;
            int cur_h1;
            if(mid_num){
                cur_h1 = 128;
                mid_num--;
            }
            else{
                cur_h1 = lastH;
                mid_num = (last_w - 1) / 128;
            }
            storedCountH += cur_h1;
            int store_st = paddingH / 128;
            
            int kS = (cur_h0 + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h0 - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, (1 << cur_sth) - 1);   
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x = m_pop_trf(0);
            }
            kS = (cur_h1 + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h1 - i * 8, 8);
                float8_128 x0 = m_pop_trf(1);
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr1) / 32), x0, (1 << cur_sth) - 1);   
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x = m_pop_trf(1);
            }
        }
        int tail_j = src_w - j;
        if(tail_j <= 0) continue;
        int addr0 = j + i * paddingW;
        int st = paddingW / 128;
        SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
        float8_128 data_a[16];
        
        data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
        m_transpose_start(data_a[0], 128, 0);
        
        for(int ii = 1; ii < 15; ii++) {
            int _i = ii * 8;
            data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
            m_transpose_mid(data_a[i], 0);
        }
    
        data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
        m_transpose_end(data_a[15], 0);
        
        int store_addr0 = storedCountH * paddingH + i;
        int store_st = paddingH / 128;
        if(mid_num){
            mid_num--;
        }
        else{
            mid_num = (last_w - 1) / 128;
        }
        
        int cur_h = lastH;
        int kS = (cur_h + 7) / 8;
        #pragma clang loop unroll_count(2)
        for(int i = 0; i < kS; i++) {
            int cur_sth = min(cur_h - i * 8, 8);
            float8_128 x0 = m_pop_trf(0);
            store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, (1 << cur_sth) - 1);   
        }
        #pragma clang loop unroll_count(2)
        for(int i = kS; i < 16; i++) {
            __attribute__((unused))float8_128 x = m_pop_trf(0);
        }
    }
    if(src_h - i <= 0) return;
    int rest_h = src_h - i;
    int storedCountH = 0;
    int push_num = (rest_h + 7) / 8;
    int j = 0;
    for (; j + 128 < src_w; j += 256) {
        int addr0 = j + i * paddingW;
        int addr1 = j + 128 + i * paddingW;
        int st = paddingW / 128;
        SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
        SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
        float8_128 data_a[16];
        float8_128 data_b[16];
        
        data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
        m_transpose_start(data_a[0], 128, 0);
        data_b[0] = load8_128_stride_ldmk(0, st, t1, 255);
        m_transpose_start(data_b[0], 128, 1);
        
        for(int ii = 1; ii < push_num - 1; ii++) {
            int _i = ii * 8;
            data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
            m_transpose_mid(data_a[i], 0);
            data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), 255);
            m_transpose_mid(data_b[i], 1);
        }
    
        data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), 255);
        m_transpose_end(data_a[15], 0);
        data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, (push_num - 1) * 8 * st * 128 / 32), 255);
        m_transpose_end(data_b[15], 1);
        
        int store_addr0 = storedCountH * paddingH + i;
        int cur_h0;
        if(mid_num){
            cur_h0 = 128;
            mid_num--;
        }
        else{
            cur_h0 = lastH;
            mid_num = (last_w - 1) / 128;
        }
        storedCountH += cur_h0;
        int store_addr1 = storedCountH * paddingH + i;
        int cur_h1;
        if(mid_num){
            cur_h1 = 128;
            mid_num--;
        }
        else{
            cur_h1 = lastH;
            mid_num = (last_w - 1) / 128;
        }
        storedCountH += cur_h1;
        int store_st = paddingH / 128;
        
        int kS = (cur_h0 + 7) / 8;
        for(int i = 0; i < kS; i++) {
            int cur_sth = min(cur_h0 - i * 8, 8);
            float8_128 x0 = m_pop_trf(0);
            store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, (1 << cur_sth) - 1);   
        }
        for(int i = kS; i < 16; i++) {
            __attribute__((unused))float8_128 x = m_pop_trf(0);
        }
        kS = (cur_h1 + 7) / 8;
        for(int i = 0; i < kS; i++) {
            int cur_sth = min(cur_h1 - i * 8, 8);
            float8_128 x0 = m_pop_trf(1);
            store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr1) / 32), x0, (1 << cur_sth) - 1);   
        }
        for(int i = kS; i < 16; i++) {
            __attribute__((unused))float8_128 x = m_pop_trf(1);
        }
    }
    int tail_j = src_w - j;
    if(tail_j <= 0) return;
    int addr0 = j + i * paddingW;
    int st = paddingW / 128;
    SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
    float8_128 data_a[16];
    
    data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
    m_transpose_start(data_a[0], 128, 0);
    
    for(int ii = 1; ii < push_num - 1; ii++) {
        int _i = ii * 8;
        data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
        m_transpose_mid(data_a[i], 0);
    }

    data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), 255);
    m_transpose_end(data_a[15], 0);
    
    int store_addr0 = storedCountH * paddingH + i;
    int store_st = paddingH / 128;
    
    int cur_h = lastH;
    int kS = (cur_h + 7) / 8;
    #pragma clang loop unroll_count(2)
    for(int i = 0; i < kS; i++) {
        int cur_sth = min(cur_h - i * 8, 8);
        float8_128 x0 = m_pop_trf(0);
        store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, (1 << cur_sth) - 1);   
    }
    #pragma clang loop unroll_count(2)
    for(int i = kS; i < 16; i++) {
        __attribute__((unused))float8_128 x = m_pop_trf(0);
    }
}

inline void tile_trans_transfer_with_stride(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr, int stride_src, int stride_dst) {
    int i = 0;
    for (; i + 128 <= src_h; i += 128) {
        int j = 0;
        for (; j + 128 <= src_w; j += 256) {
            int addr0 = j + i * stride_src;
            int addr1 = j + 128 + i * stride_src;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];
            
            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_start(data_b[0], 128, 1);
            
            for(int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), 255);
                m_transpose_mid(data_b[i], 1);
            }
        
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 120 * st * 128 / 32), 255);
            m_transpose_end(data_b[15], 1);
            
            int store_addr0 = j * stride_dst + i;
            int store_addr1 = (j + 128) * stride_dst + i;
            int store_st = stride_dst / 128;
            
            int cur_h = min(src_w - j - 128, 128);
            int kS = (cur_h + 7) / 8;
            #pragma clang loop unroll_count(2)
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                store8_128_stride_stmk(i * 8 * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, 255);   
                float8_128 x1 = m_pop_trf(1);
                store8_128_stride_stmk(i * 8 * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr1) / 32), x1, (1 << cur_sth) - 1);   
            }
            #pragma clang loop unroll_count(2)
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x1 = m_pop_trf(1);
                float8_128 x = m_pop_trf(0);
                store8_128_stride_stmk(i * 8 * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x, 255);   
            }
        }
        int tail_j = src_w - j;
        if(tail_j <= 0) continue;
        int addr0 = j + i * stride_src;
        int st = stride_src / 128;
        SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
        float8_128 data_a[16];
        
        data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
        m_transpose_start(data_a[0], 128, 0);
        
        for(int ii = 1; ii < 15; ii++) {
            int _i = ii * 8;
            data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
            m_transpose_mid(data_a[i], 0);
        }
    
        data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
        m_transpose_end(data_a[15], 0);
        
        int store_addr0 = j * stride_dst + i;
        int store_st = stride_dst / 128;
        
        int cur_h = min(src_w - j, 128);
        int kS = (cur_h + 7) / 8;
        #pragma clang loop unroll_count(2)
        for(int i = 0; i < kS; i++) {
            int cur_sth = min(cur_h - i * 8, 8);
            float8_128 x0 = m_pop_trf(0);
            store8_128_stride_stmk(i * 8 * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr0) / 32), x0, (1 << cur_sth) - 1);   
        }
        #pragma clang loop unroll_count(2)
        for(int i = kS; i < 16; i++) {
            __attribute__((unused))float8_128 x = m_pop_trf(0);
        }
    }
    if(src_h - i <= 0) return;
    for (int j = 0; j < src_w; j += 128) {\
        float128_128 v = loadh_k_T(tensor_slice(src, srcaddr / 32), stride_dst, src_h, stride_src, src_w, i / 128, j / 128);
        store128_128_ex_permute_with_stride2(tensor_slice(dst, dstaddr / 32), src_w, src_h, j / 128, i / 128, stride_dst, v);
    }
}

inline void set_permute_for_tsps_bf16() {
    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);
}

// 在调用该函数前面set permute
// int8_128 coreid = get_core_id();
// int8_128 permute_odd = coreid * 2;
// int8_128 permute_even = permute_odd + 1;
// m_set_permute(permute_odd, 0);
// m_set_permute(permute_even, 1);
inline void tile_trans_transfer_bf16(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr, int stride_src, int stride_dst) {
    int i = 0;
    for (; i + 256 <= src_h; i += 256) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr1 = (i + 128) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], 128, 1);

            for(int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data1[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int addr2 = (i + 64) * stride_src + j / 2;
            int addr3 = (i + 192) * stride_src + j / 2;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);
            SIM_X86::tensor t3 = tensor_slice(src, srcaddr / 32 + addr3 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            float8_128 __attribute__((address_space(2))) data3[8];
            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);
            data3[0] = load8_128_stride_ldmk(0, st, t3, 255);
            m_transpose_packed_start(data3[0], 128, 1);

            for(int index = 1; index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
                data3[index] = load8_128_stride_ldmk(0, st, tensor_slice(t3, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data3[index], 1);
            }

            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data2[7], 0);
            data3[7] = load8_128_stride_ldmk(0, st, tensor_slice(t3, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data3[7], 1);

            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);

                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res23[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
    int rest_h = src_h - i;
    if(rest_h == 0) return ;
    if(rest_h <= 64) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int st = stride_src / 128;
            int push_num = (rest_h + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            int cur_sth = min(rest_h, 8);
            data0[0] = load8_128_stride_ldmk(0, st, t0, (1 << cur_sth) - 1);

            m_transpose_packed_start(data0[0], 128, 0);
        
            for(int index = 1; index < push_num - 1; index++) {
                cur_sth = min(rest_h - index * 8, 8);
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data0[index], 0);
            }

            cur_sth = min(rest_h - (push_num - 1) * 8, 8);
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data0[7], 0);
            
            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                x0 = __$F(__$S(x0) << 16);
                float8_128 zero = v_u32_move_b(0);
                res01[index] = __$F(float_to_bfloat16(zero, x0));
            }

            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            
            for(int index = 0; index < store_num; index++) {
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h <= 128) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr2 = (i + 64) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data2[8];
            int push_num = (rest_h - 64 + 7) / 8;
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            int cur_sth = min(rest_h - 64, 8);
            data2[0] = load8_128_stride_ldmk(0, st, t2, (1 << cur_sth) - 1);
            m_transpose_packed_start(data2[0], 128, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 64 - index * 8, 8);
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data2[index], 1);
            }

            for(int index = _max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 64 - (push_num - 1) * 8, 8);
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data2[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                float8_128 zero = v_u32_move_b(0);
                res01[index] = __$F(float_to_bfloat16(zero, x0));
                res23[index] = __$F(float_to_bfloat16(zero, x1));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {

                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h <= 192) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr1 = (i + 128) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            int cur_sth = min(rest_h - 128, 8);
            data1[0] = load8_128_stride_ldmk(0, st, t1, (1 << cur_sth) - 1);
            int push_num = (rest_h - 128 + 7) / 8;

            m_transpose_packed_start(data1[0], 128, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 128 - index * 8, 8);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data1[index], 1);
            }

            for(int index = _max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 128 - (push_num - 1) * 8, 8);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data1[7], 1);
            
            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int addr2 = (i + 64) * stride_src + j / 2;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);

            for(int index = 1; index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
            }

            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data2[7], 0);

            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                x0 = __$F(__$S(x0) << 16);
                float8_128 zero = v_u32_move_b(0);
                res23[index] = __$F(float_to_bfloat16(zero, x0));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h < 256) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr1 = (i + 128) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], 128, 1);

            for(int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data1[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int addr2 = (i + 64) * stride_src + j / 2;
            int addr3 = (i + 192) * stride_src + j / 2;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);
            SIM_X86::tensor t3 = tensor_slice(src, srcaddr / 32 + addr3 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            float8_128 __attribute__((address_space(2))) data3[8];
            int push_num = (rest_h - 192 + 7) / 8;

            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);
            int cur_sth = min(rest_h - 192, 8);
            data3[0] = load8_128_stride_ldmk(0, st, t3, (1 << cur_sth) - 1);
            m_transpose_packed_start(data3[0], 128, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
                cur_sth = min(rest_h - 192 - index * 8, 8);
                data3[index] = load8_128_stride_ldmk(0, st, tensor_slice(t3, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data3[index], 1);
            }

            for(int index = _max(push_num - 1, 1); index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
            }
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data2[7], 0);
            cur_sth = min(rest_h - 192 - (push_num - 1) * 8, 8);
            data3[7] = load8_128_stride_ldmk(0, st, tensor_slice(t3, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data3[7], 1);            

            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);

                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res23[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
}

// 在调用该函数前面set permute
// int8_128 coreid = get_core_id();
// int8_128 permute_odd = coreid * 2;
// int8_128 permute_even = permute_odd + 1;
// m_set_permute(permute_odd, 0);
// m_set_permute(permute_even, 1);
inline void tile_trans_transfer_bf16_to_f32_with_stride(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr,
                                            int stride_src, int stride_dst) {
    int i = 0;
    for (; i + 128 <= src_h; i += 128) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr1 = (i + 64) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];

            int trans_pack_width = (min(128, src_w - j) + 15) & 0xFFFFFFF0;
            int trans_pop = trans_pack_width / 8;

            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], trans_pack_width, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], trans_pack_width, 1);

            for(int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data1[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < trans_pop; index++) {
                res01[index] = m_pop_trf(0);
                res23[index] = m_pop_trf(1);
                res01[index] = __$F(__$S(res01[index]) << 16);
                res23[index] = __$F(__$S(res23[index]) << 16);
            }
            
            int store_addr = j * stride_dst + i;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
    int rest_h = src_h - i;
    if(rest_h == 0) return ;
    if(rest_h <= 64) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int st = stride_src / 128;
            int push_num = (rest_h + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            
            int trans_pack_width = (min(128, src_w - j) + 15) & 0xFFFFFFF0;
            int trans_pop = trans_pack_width / 8;

            int cur_sth = min(rest_h, 8);
            data0[0] = load8_128_stride_ldmk(0, st, t0, (1 << cur_sth) - 1);
            m_transpose_packed_start(data0[0], trans_pack_width, 0);
        
            for(int index = 1; index < push_num - 1; index++) {
                cur_sth = min(rest_h - index * 8, 8);
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data0[index], 0);
            }
            cur_sth = min(rest_h - (push_num - 1) * 8, 8);
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data0[7], 0);

            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < trans_pop; index++) {
                res01[index] = m_pop_trf(0);
                res01[index] = __$F(__$S(res01[index]) << 16);
            }

            int store_addr = j * stride_dst + i;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h <= 128) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr2 = (i + 64) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data2[8];

            int trans_pack_width = (min(128, src_w - j) + 15) & 0xFFFFFFF0;
            int trans_pop = trans_pack_width / 8;

            int push_num = (rest_h - 64 + 7) / 8;
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], trans_pack_width, 0);
            int cur_sth = min(rest_h - 64, 8);
            data2[0] = load8_128_stride_ldmk(0, st, t2, (1 << cur_sth) - 1);
            m_transpose_packed_start(data2[0], trans_pack_width, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 64 - index * 8, 8);
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data2[index], 1);
            }

            for(int index = _max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 64 - (push_num - 1) * 8, 8);
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data2[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < trans_pop; index++) {
                res01[index] = m_pop_trf(0);
                res23[index] = m_pop_trf(1);
                res01[index] = __$F(__$S(res01[index]) << 16);
                res23[index] = __$F(__$S(res23[index]) << 16);
            }
            
            int store_addr = j * stride_dst + i;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {

                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
}

inline void tile_trans_transfer_bf16_to_f32(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr) {
    tile_trans_transfer_bf16_to_f32_with_stride(src, dst, srcaddr, src_h, src_w, dstaddr, 
                                                padding256(src_w) / 2, padding128(src_h));
}

// e.g. [4, (3, ..., 333)] => [(3, ..., 333), 4]
// src_h = 4
// mid = 3 * ...
// last_w = 333
// 在调用该函数前面set permute
// int8_128 coreid = get_core_id();
// int8_128 permute_odd = coreid * 2;
// int8_128 permute_even = permute_odd + 1;
// m_set_permute(permute_odd, 0);
// m_set_permute(permute_even, 1);
inline void tile_trans_transfer_bf16_to_f32_mid(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int dstaddr, int src_h, int mid, int last_w) {
    int paddingH = padding128(src_h);
    int src_w = padding256(last_w) * mid;
    int paddingW = padding256(src_w) / 2;
    int lastH =  last_w % 256;
    int mid_num = (last_w - 1) / 256;
    if((last_w & 255) == 0) lastH = 256;
    int i = 0;
    for (; i + 128 <= src_h; i += 128) {
        int storedCountH = 0;
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * paddingW + j / 2;
            int addr1 = (i + 64) * paddingW + j / 2;
            int st = paddingW / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], 128, 1);

            for(int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data1[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                res01[index] = m_pop_trf(0);
                res23[index] = m_pop_trf(1);
                res01[index] = __$F(__$S(res01[index]) << 16);
                res23[index] = __$F(__$S(res23[index]) << 16);
            }
            
            int store_addr = storedCountH * paddingH + i;
            int store_st = paddingH / 128;
            int cur_h;
            if(mid_num){
                cur_h = 256;
                mid_num--;
            }
            else{
                cur_h = lastH;
                mid_num = (last_w - 1) / 256;
            }
            storedCountH += cur_h;
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * paddingH + 128 * paddingH) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
    int rest_h = src_h - i;
    if(rest_h == 0) return ;
    if(rest_h <= 64) {
        int storedCountH = 0;
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * paddingW + j / 2;
            int st = paddingW / 128;
            int push_num = (rest_h + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            int cur_sth = min(rest_h, 8);
            data0[0] = load8_128_stride_ldmk(0, st, t0, (1 << cur_sth) - 1);
            m_transpose_packed_start(data0[0], 128, 0);
        
            for(int index = 1; index < push_num - 1; index++) {
                cur_sth = min(rest_h - index * 8, 8);
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data0[index], 0);
            }
            cur_sth = min(rest_h - (push_num - 1) * 8, 8);
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data0[7], 0);

            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                res01[index] = m_pop_trf(0);
                res01[index] = __$F(__$S(res01[index]) << 16);
            }

            int store_addr = storedCountH * paddingH + i;
            int store_st = paddingH / 128;
            int cur_h;
            if(mid_num){
                cur_h = 256;
                mid_num--;
            }
            else{
                cur_h = lastH;
                mid_num = (last_w - 1) / 256;
            }
            storedCountH += cur_h;
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * paddingH + 128 * paddingH) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h <= 128) {
        int storedCountH = 0;
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * paddingW + j / 2;
            int addr2 = (i + 64) * paddingW + j / 2;
            int st = paddingW / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data2[8];
            int push_num = (rest_h - 64 + 7) / 8;
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            int cur_sth = min(rest_h - 64, 8);
            data2[0] = load8_128_stride_ldmk(0, st, t2, (1 << cur_sth) - 1);
            m_transpose_packed_start(data2[0], 128, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 64 - index * 8, 8);
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data2[index], 1);
            }

            for(int index = _max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 64 - (push_num - 1) * 8, 8);
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data2[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                res01[index] = m_pop_trf(0);
                res23[index] = m_pop_trf(1);
                res01[index] = __$F(__$S(res01[index]) << 16);
                res23[index] = __$F(__$S(res23[index]) << 16);
            }
            
            int store_addr = storedCountH * paddingH + i;
            int store_st = paddingH / 128;
            int cur_h;
            if(mid_num){
                cur_h = 256;
                mid_num--;
            }
            else{
                cur_h = lastH;
                mid_num = (last_w - 1) / 256;
            }
            storedCountH += cur_h;
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {

                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * paddingH + 128 * paddingH) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
}

inline void tile_trans_transfer_f32_to_bf16(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr) {
    int paddingH = padding256(src_h) / 2;
    int paddingW = padding128(src_w);
    int i = 0;
    for (; i + 256 <= src_h; i += 256) {
        for (int j = 0; j < src_w; j += 128) {
            int addr0 = j + i * paddingW;
            int addr1 = j + (i + 128) * paddingW;
            int st = paddingW / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];
            
            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_start(data_b[0], 128, 1);
            
            for(int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), 255);
                m_transpose_mid(data_b[i], 1);
            }
        
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 120 * st * 128 / 32), 255);
            m_transpose_end(data_b[15], 1);
            
            int store_addr = j * paddingH + i / 2;
            int store_st = paddingH / 128;
            
            int cur_h = min(src_w - j, 128);
            int kS = (cur_h + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                float8_128 res = __$F(float_to_bfloat16(x1, x0));
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), res, (1 << cur_sth) - 1);
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x0 = m_pop_trf(0);
                __attribute__((unused))float8_128 x1 = m_pop_trf(1);
            }
        }
    }
    int rest_h = src_h - i;
    if(rest_h == 0) return ;
    if(rest_h <= 128) {
        for (int j = 0; j < src_w; j += 128) {
            int addr0 = j + i * paddingW;
            int st = paddingW / 128;
            int push_num = (rest_h + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            float8_128 data_a[16];
            
            int cur_sth = min(rest_h, 8);
            data_a[0] = load8_128_stride_ldmk(0, st, t0, (1 << cur_sth) - 1);
            m_transpose_start(data_a[0], 128, 0);
            
            for(int ii = 1; ii < push_num - 1; ii++) {
                cur_sth = min(rest_h - ii * 8, 8);
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_mid(data_a[i], 0);
            }
        
            cur_sth = min(rest_h - (push_num - 1) * 8, 8);
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_end(data_a[15], 0);
            
            int store_addr = j * paddingH + i / 2;
            int store_st = paddingH / 128;
            
            int cur_h = min(src_w - j, 128);
            int kS = (cur_h + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = v_u32_move_b(0);
                float8_128 res = __$F(float_to_bfloat16(x1, x0));
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), res, (1 << cur_sth) - 1);
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x0 = m_pop_trf(0);
            }
        }
    } else {
        for (int j = 0; j < src_w; j += 128) {
            int addr0 = j + i * paddingW;
            int addr1 = j + (i + 128) * paddingW;
            int st = paddingW / 128;
            int push_num = (rest_h - 128 + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];
            
            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            int cur_sth = min(rest_h - 128, 8);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, (1 << cur_sth) - 1);
            m_transpose_start(data_b[0], 128, 1);
            
            for(int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                cur_sth = min(rest_h - 128 - ii * 8, 8);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_mid(data_b[i], 1);
            }
            
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            cur_sth = min(rest_h - 128 - (push_num - 1) * 8, 8);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, (push_num - 1) * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_end(data_b[15], 1);
            
            int store_addr = j * paddingH + i / 2;
            int store_st = paddingH / 128;
            
            int cur_h = min(src_w - j, 128);
            int kS = (cur_h + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                float8_128 res = __$F(float_to_bfloat16(x1, x0));
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), res, (1 << cur_sth) - 1);
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x0 = m_pop_trf(0);
                __attribute__((unused))float8_128 x1 = m_pop_trf(1);
            }
        }
    }
}

// [O, L, pR] => [O', L, pR]
// perm[4] == 4 && perm[3] == 3
inline void permute_keeplo2(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst) {
    oridim[0] = padding128(oridim[0]);
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int idx[5] = {0, 0, 0, 0, 0};
    for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
        for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
            for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];

                int len = oridim[0] * oridim[1];
                int ldst_msk = pre_exp2((len - len / 1024 * 1024) / 128);

                int i = 0;
                for (; i < len / 1024 * 1024; i += 1024) {
                    float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_base_src / 32);
                    v_f32_st_tnsr_b(i / 32, dst + offset_base_dst / 32, x);
                }
                if (ldst_msk) {
                    float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, src + offset_base_src / 32, 1, ldst_msk);
                    v_f32_st_tnsr_st_msk(i / 32, dst + offset_base_dst / 32, 1, ldst_msk, x);
                }
            }
        }
    }
}

// [O, L, pR] => [O', L, pR]
// perm[4] == 4 && perm[3] == 3
inline void permute_keeplo2_small(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem,
                                  int *oridim, int *perm, int *idx_origin, int cnt, int BLACK) {
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int dim01_len = oridim[0] * oridim[1];

    if (BLACK) {
        int idx[5];
        int i = 0;
        
        idx_origin[2]++;
        idx_origin[1] = 0;
        if (idx_origin[2] >= oridim[2]) {
            idx_origin[3]++;
            idx_origin[2] = 0;
            if (idx_origin[3] >= oridim[3]) {
                idx_origin[4]++;
                idx_origin[3] = 0;
            }
        }
        int offset_base_src = idx_origin[4] * oldst[4] + idx_origin[3] * oldst[3] + idx_origin[2] * oldst[2];
        HBM2VMem(input0_hbm + offset_base_src / 32, input0_vmem, cnt * dim01_len);

        for (idx[4] = idx_origin[4]; idx[4] < oridim[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim[2]; ++idx[2]) {
                    idx_origin[2] = 0;

                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];
                    dlc_dma(input0_vmem + i * dim01_len / 32, VMEM, output_hbm + offset_base_dst / 32, HBM, dim01_len, 128, 128, 128, 7);
                    i++;
                }
            }
        }
        Vmem2HBM(input0_vmem, output_hbm, 0);
    } else {
        int idx[5];
        int i = cnt - 1;
        
        HBM2VMem(input0_hbm, input0_vmem, cnt * dim01_len);

        for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
            idx_origin[4] = oridim[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                idx_origin[3] = oridim[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                    idx_origin[2] = oridim[2] - 1;

                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];
                    dlc_dma(input0_vmem + i * dim01_len / 32, VMEM, output_hbm + offset_base_dst / 32, HBM, dim01_len, 128, 128, 128, 7);
                    i--;
                }
            }
        }
        Vmem2HBM(input0_vmem, output_hbm, 0);
    }
}

// [O, L, pR] => [O', L, pR]
// perm[4] == 4 && perm[3] == 3
inline void permute_keeplo2_big(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize,
                                 int *oridim, int *perm, int *idx_origin, int BLACK) {
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};

    if (BLACK) {
        int idx[5];
        
        idx_origin[2]++;
        idx_origin[1] = 0;
        if (idx_origin[2] >= oridim[2]) {
            idx_origin[3]++;
            idx_origin[2] = 0;
            if (idx_origin[3] >= oridim[3]) {
                idx_origin[4]++;
                idx_origin[3] = 0;
            }
        }

        for (idx[4] = idx_origin[4]; idx[4] < oridim[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim[2]; ++idx[2]) {
                    idx_origin[2] = 0;

                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];
                    int hbmlen = oridim[0] * oridim[1];

                    // if (((int)(input0_hbm + offset_base_src / 32) - (int)(output_hbm + offset_base_dst / 32)) % 128 == 0) {
                    int diff = ((int)(input0_hbm + offset_base_src / 32) - (int)(output_hbm + offset_base_dst / 32));
                    if (diff - diff / 128 * 128 == 0) {
                        dlc_dma(input0_hbm + offset_base_src / 32, HBM,
                                output_hbm + offset_base_dst / 32, HBM, hbmlen, 128, 128, 128, 7);
                    } else {
                        for (int len = 0; len < hbmlen; len += VMEMSize) {
                            int vmemsize = min(hbmlen - len, VMEMSize);

                            int sync = dlc_dma(input0_hbm + offset_base_src / 32 + len / 32, HBM,
                                                input0_vmem, VMEM, vmemsize, 128, 128, 128, 7);
                            dlc_sync(sync);
                            int sync2 = dlc_dma(input0_vmem, VMEM,
                                                output_hbm + offset_base_dst / 32 + len / 32, HBM,
                                                vmemsize, 128, 128, 128, 7);
                            dlc_sync(sync2);
                        }
                    }
                }
            }
        }
        int sync = dlc_dma(input0_hbm, HBM, output_hbm, HBM, 0, 128, 128, 128, 7);
        dlc_sync(sync);
    } else {
        int idx[5];

        for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
            idx_origin[4] = oridim[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                idx_origin[3] = oridim[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                    idx_origin[2] = oridim[2] - 1;

                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];
                    int hbmlen = oridim[0] * oridim[1];

                    // if (((int)(input0_hbm + offset_base_src / 32) - (int)(output_hbm + offset_base_dst / 32)) % 128 == 0) {
                    int diff = ((int)(input0_hbm + offset_base_src / 32) - (int)(output_hbm + offset_base_dst / 32));
                    if (diff - diff / 128 * 128 == 0) {
                        dlc_dma(input0_hbm + offset_base_src / 32, HBM,
                                output_hbm + offset_base_dst / 32, HBM, hbmlen, 128, 128, 128, 7);
                    } else {
                        for (int len = 0; len < hbmlen; len += VMEMSize) {
                            int vmemsize = min(hbmlen - len, VMEMSize);

                            int sync = dlc_dma(input0_hbm + offset_base_src / 32 + len / 32, HBM,
                                                input0_vmem, VMEM, vmemsize, 128, 128, 128, 7);
                            dlc_sync(sync);
                            int sync2 = dlc_dma(input0_vmem, VMEM,
                                                output_hbm + offset_base_dst / 32 + len / 32, HBM,
                                                vmemsize, 128, 128, 128, 7);
                            dlc_sync(sync2);
                        }
                    }
                }
            }
        }
        int sync = dlc_dma(input0_hbm, HBM, output_hbm, HBM, 0, 128, 128, 128, 7);
        dlc_sync(sync);
    }
}

// [O, pR] => [O', pR]
// perm[4] == 4
inline void permute_keeplo1(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst) {
    oridim[0] = padding128(oridim[0]);
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int idx[5] = {0, 0, 0, 0, 0};
    for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
        for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
            for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                for (idx[1] = 0; idx[1] < oridim[1]; idx[1]++) {
                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                    int len = oridim[0];
                    int ldst_msk = pre_exp2((len - len / 1024 * 1024) / 128);

                    int i = 0;
                    for (; i < len / 1024 * 1024; i += 1024) {
                        float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_base_src / 32);
                        v_f32_st_tnsr_b(i / 32, dst + offset_base_dst / 32, x);
                    }
                    if (ldst_msk) {
                        float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, src + offset_base_src / 32, 1, ldst_msk);
                        v_f32_st_tnsr_st_msk(i / 32, dst + offset_base_dst / 32, 1, ldst_msk, x);
                    }
                }
            }
        }
    }
}

// [O, pR] => [O', pR]
// perm[4] == 4
inline void permute_keeplo1_small(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem,
                                  int *oridim, int *perm, int *idx_origin, int cnt, int BLACK) {
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int dim0_128 = oridim[0];

    if (BLACK) {
        int idx[5];
        int i = 0;
        
        idx_origin[1]++;
        if (idx_origin[1] >= oridim[1]) {
            idx_origin[2]++;
            idx_origin[1] = 0;
            if (idx_origin[2] >= oridim[2]) {
                idx_origin[3]++;
                idx_origin[2] = 0;
                if (idx_origin[3] >= oridim[3]) {
                    idx_origin[4]++;
                    idx_origin[3] = 0;
                }
            }
        }
        int offset_base_src = idx_origin[4] * oldst[4] + idx_origin[3] * oldst[3] + idx_origin[2] * oldst[2] + idx_origin[1] * oldst[1];
        HBM2VMem(input0_hbm + offset_base_src / 32, input0_vmem, cnt * dim0_128);

        for (idx[4] = idx_origin[4]; idx[4] < oridim[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim[2]; ++idx[2]) {
                    idx_origin[2] = 0;
                    for (idx[1] = idx_origin[1]; idx[1] < oridim[1]; ++idx[1]) {
                        idx_origin[1] = 0;

                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];
                        dlc_dma(input0_vmem + i * dim0_128 / 32, VMEM, output_hbm + offset_base_dst / 32, HBM, dim0_128, 128, 128, 128, 7);
                        i++;
                    }
                }
            }
        }
        Vmem2HBM(input0_vmem, output_hbm, 0);
    } else {
        int idx[5];
        int i = cnt - 1;
        
        HBM2VMem(input0_hbm, input0_vmem, cnt * dim0_128);

        for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
            idx_origin[4] = oridim[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                idx_origin[3] = oridim[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                    idx_origin[2] = oridim[2] - 1;
                    for (idx[1] = idx_origin[1]; idx[1] >= 0; --idx[1]) {
                        idx_origin[1] = oridim[1] - 1;

                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];
                        dlc_dma(input0_vmem + i * dim0_128 / 32, VMEM, output_hbm + offset_base_dst / 32, HBM, dim0_128, 128, 128, 128, 7);
                        i--;
                    }
                }
            }
        }
        
        Vmem2HBM(input0_vmem, output_hbm, 0);
    }
}

// [O, pR] => [O', pR]
// perm[4] == 4
inline void permute_keeplo1_big(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize,
                                 int *oridim, int *perm, int *idx_origin, int BLACK) {
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};

    if (BLACK) {
        int idx[5];
        
        idx_origin[1]++;
        if (idx_origin[1] >= oridim[1]) {
            idx_origin[2]++;
            idx_origin[1] = 0;
            if (idx_origin[2] >= oridim[2]) {
                idx_origin[3]++;
                idx_origin[2] = 0;
                if (idx_origin[3] >= oridim[3]) {
                    idx_origin[4]++;
                    idx_origin[3] = 0;
                }
            }
        }

        for (idx[4] = idx_origin[4]; idx[4] < oridim[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim[2]; ++idx[2]) {
                    idx_origin[2] = 0;
                    for (idx[1] = idx_origin[1]; idx[1] < oridim[1]; ++idx[1]) {
                        idx_origin[1] = 0;

                        int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];
                        int hbmlen = oridim[0];

                        int diff = ((int)(input0_hbm + offset_base_src / 32) - (int)(output_hbm + offset_base_dst / 32));
                        if (diff - diff / 128 * 128 == 0) {
                            dlc_dma(input0_hbm + offset_base_src / 32, HBM,
                                    output_hbm + offset_base_dst / 32, HBM, hbmlen, 128, 128, 128, 7);
                        } else {
                            for (int len = 0; len < hbmlen; len += VMEMSize) {
                                int vmemsize = min(hbmlen - len, VMEMSize);

                                int sync = dlc_dma(input0_hbm + offset_base_src / 32 + len / 32, HBM,
                                                    input0_vmem, VMEM, vmemsize, 128, 128, 128, 7);
                                dlc_sync(sync);
                                int sync2 = dlc_dma(input0_vmem, VMEM,
                                                    output_hbm + offset_base_dst / 32 + len / 32, HBM,
                                                    vmemsize, 128, 128, 128, 7);
                                dlc_sync(sync2);
                            }
                        }
                    }
                }
            }
        }
        int sync = dlc_dma(input0_hbm, HBM, output_hbm, HBM, 0, 128, 128, 128, 7);
        dlc_sync(sync);
    } else {
        int idx[5];

        for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
            idx_origin[4] = oridim[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                idx_origin[3] = oridim[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                    idx_origin[2] = oridim[2] - 1;
                    for (idx[1] = idx_origin[1]; idx[1] >= 0; --idx[1]) {
                        idx_origin[1] = oridim[1] - 1;

                        int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];
                        int hbmlen = oridim[0];

                        int diff = ((int)(input0_hbm + offset_base_src / 32) - (int)(output_hbm + offset_base_dst / 32));
                        if (diff - diff / 128 * 128 == 0) {
                            dlc_dma(input0_hbm + offset_base_src / 32, HBM,
                                    output_hbm + offset_base_dst / 32, HBM, hbmlen, 128, 128, 128, 7);
                        } else {
                            for (int len = 0; len < hbmlen; len += VMEMSize) {
                                int vmemsize = min(hbmlen - len, VMEMSize);

                                int sync = dlc_dma(input0_hbm + offset_base_src / 32 + len / 32, HBM,
                                                    input0_vmem, VMEM, vmemsize, 128, 128, 128, 7);
                                dlc_sync(sync);
                                int sync2 = dlc_dma(input0_vmem, VMEM,
                                                    output_hbm + offset_base_dst / 32 + len / 32, HBM,
                                                    vmemsize, 128, 128, 128, 7);
                                dlc_sync(sync2);
                            }
                        }
                    }
                }
            }
        }
        int sync = dlc_dma(input0_hbm, HBM, output_hbm, HBM, 0, 128, 128, 128, 7);
        dlc_sync(sync);
    }
}

inline int* nextidx(int *idx, int *dim, int incidx) {
    if (incidx > 4) {
        return idx;
    }
    idx[incidx]++;
    if (idx[0] >= dim[0]) {
        idx[0] = 0;
        idx[1]++;
    }
    if (idx[1] >= dim[1]) {
        idx[1] = 0;
        idx[2]++;
    }
    if (idx[2] >= dim[2]) {
        idx[2] = 0;
        idx[3]++;
    }
    if (idx[3] >= dim[3]) {
        idx[3] = 0;
        idx[4]++;
    }
    if (idx[4] >= dim[4]) {
        idx[4] = 0;
    }
    return idx;
}

inline int* nextkidx(int *idx, int *dim, int incidx, int k) {
    idx[incidx] += k;
    idx[1] += idx[0] / dim[0];
    idx[0] %= dim[0];
    idx[2] += idx[1] / dim[1];
    idx[1] %= dim[1];
    idx[3] += idx[2] / dim[2];
    idx[2] %= dim[2];
    idx[4] += idx[3] / dim[3];
    idx[3] %= dim[3];
    idx[4] %= dim[4];
    return idx;
}

inline int* nextkridx(int *idx, int *dim, int incidx, int k, int range) {
    idx[incidx] += k;
    idx[1] += idx[0] / dim[0];
    idx[0] %= dim[0];
    if (range == 2) {
        idx[1] %= dim[1];
        return idx;
    }
    idx[2] += idx[1] / dim[1];
    idx[1] %= dim[1];
    if (range == 3) {
        idx[2] %= dim[2];
        return idx;
    }
    idx[3] += idx[2] / dim[2];
    idx[2] %= dim[2];
    if (range == 4) {
        idx[3] %= dim[3];
        return idx;
    }
    idx[4] += idx[3] / dim[3];
    idx[3] %= dim[3];
    idx[4] %= dim[4];
    return idx;
}

// p0 != 0 and p1 != 0
// K which permute to lo-dim(dim 0), need process padding
// [H, K, L, pR] =>(load) [H, L, R, pK] =>(store) [(H, L, R)', pK]
inline void load_tran_trans(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst, int rd0) {
    oridim[0] = padding128(oridim[0]);
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding128(newdim[0]);
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int oridim_new[5];
    for (int i = 0; i < 5; i++) {
        oridim_new[i] = oridim[i];
    }

    int input0_stride = 0;
    int output_stride = 1;

    if (perm[0] == 4) {
        oridim_new[4] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2] * oridim[3];
    } else if (perm[0] == 3) {
        oridim_new[3] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2];
    } else if (perm[0] == 2) {
        oridim_new[2] = 1;
        input0_stride = oridim[0] * oridim[1];
    } else if (perm[0] == 1) {
        oridim_new[1] = 1;
        input0_stride = oridim[0];
    }

    if (perm[1] == 0) output_stride = newdim[0];
    else if (perm[2] == 0) output_stride = newdim[0] * newdim[1];
    else if (perm[3] == 0) output_stride = newdim[0] * newdim[1] * newdim[2];
    else if (perm[4] == 0) output_stride = newdim[0] * newdim[1] * newdim[2] * newdim[3];

    int hbm_h = rodim[perm[0]];
    int hbm_w = rd0;

    int idx[5] = {0, 0, 0, 0, 0};
    for (idx[4] = 0; idx[4] < oridim_new[4]; ++idx[4]) {
        for (idx[3] = 0; idx[3] < oridim_new[3]; ++idx[3]) {
            for (idx[2] = 0; idx[2] < oridim_new[2]; ++idx[2]) {
                for (idx[1] = 0; idx[1] < oridim_new[1]; ++idx[1]) {
                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                    tile_trans_transfer_with_stride(src, dst, offset_base_src,
                                                    hbm_h, hbm_w, offset_base_dst, input0_stride, output_stride);
                }
            }
        }
    }
}

inline void load_tran_trans_bf16(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst, int rd0) {
    oridim[0] = padding256(oridim[0]) / 2;
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding256(newdim[0]) / 2;
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int oridim_new[5];
    for (int i = 0; i < 5; i++) {
        oridim_new[i] = oridim[i];
    }

    int input0_stride = 0;
    int output_stride = 1;

    if (perm[0] == 4) {
        oridim_new[4] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2] * oridim[3];
    } else if (perm[0] == 3) {
        oridim_new[3] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2];
    } else if (perm[0] == 2) {
        oridim_new[2] = 1;
        input0_stride = oridim[0] * oridim[1];
    } else if (perm[0] == 1) {
        oridim_new[1] = 1;
        input0_stride = oridim[0];
    }

    for (int i = 0; i < 5; ++i) {
        if (perm[i] != 0) output_stride *= newdim[i];
        else break;
    }

    int hbm_h = rodim[perm[0]];
    int hbm_w = rd0;

    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);

    int idx[5] = {0, 0, 0, 0, 0};
    for (idx[4] = 0; idx[4] < oridim_new[4]; ++idx[4]) {
        for (idx[3] = 0; idx[3] < oridim_new[3]; ++idx[3]) {
            for (idx[2] = 0; idx[2] < oridim_new[2]; ++idx[2]) {
                for (idx[1] = 0; idx[1] < oridim_new[1]; ++idx[1]) {
                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                    tile_trans_transfer_bf16(src, dst, offset_base_src, hbm_h, hbm_w, offset_base_dst, input0_stride, output_stride);

                    // return;
                }
            }
        }
    }
}

// [O, T, pR] => [O', R, pT], rd0 = d0 before padding
// perm[0] == 1 && perm[1] == 0
inline void permute_easy_do(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst, int rd0) {
    oridim[0] = padding128(oridim[0]);
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding128(newdim[0]);
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int idx[5] = {0, 0, 0, 0, 0};
    // int padding_oridim0 = (oridim[0] + 127) & 0xffffff80;
    for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
        for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
            for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                int src_hi2_off = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                int dst_hi2_off = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];
                tile_trans_transfer(src, dst, src_hi2_off, oridim[1], rd0, dst_hi2_off);
            }
        }
    }
}

inline void permute_easy_do_bf16(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst, int rd0) {
    oridim[0] = padding256(oridim[0]) / 2;
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding256(newdim[0]) / 2;
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int idx[5] = {0, 0, 0, 0, 0};

    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);

    for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
        for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
            for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                int src_hi2_off = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                int dst_hi2_off = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];

                tile_trans_transfer_bf16(src, dst, src_hi2_off, oridim[1], rd0, dst_hi2_off, oridim[0], newdim[0]);
            }
        }
    }
}

inline void permute(SIM_X86::tensor in, SIM_X86::tensor out, SIM_X86::TensorInfo *inf, int d0, int p4, int p3, int p2, int p1, int p0) {
    int d4 = inf->SpaceSize[4];
    int d3 = inf->SpaceSize[3];
    int d2 = inf->SpaceSize[2];
    int d1 = inf->SpaceSize[1];
    int pd0 = inf->SpaceSize[0] * 128;
    int oridim[5] = {pd0, d1, d2, d3, d4};
    int perm[5] = {p0, p1, p2, p3, p4};
    if (p0 == 0 && p1 == 1) {
        permute_keeplo2(oridim, perm, in, out);
    } else if (p0 == 0) {
        permute_keeplo1(oridim, perm, in, out);
    } else if (p0 == 1 && p1 == 0) {
        permute_easy_do(oridim, perm, in, out, d0);
    } else {
        load_tran_trans(oridim, perm, in, out, d0);
    }
}

inline void permuteBF16(SIM_X86::tensor in, SIM_X86::tensor out, SIM_X86::TensorInfo *inf, int d0, int p4, int p3, int p2, int p1,
                        int p0) {
    int d4 = inf->SpaceSize[4];
    int d3 = inf->SpaceSize[3];
    int d2 = inf->SpaceSize[2];
    int d1 = inf->SpaceSize[1];
    int pd0 = inf->SpaceSize[0];
    int oridim[5] = {pd0, d1, d2, d3, d4};
    int perm[5] = {p0, p1, p2, p3, p4};
    if (p0 == 0 && p1 == 1) {
        permute_keeplo2(oridim, perm, in, out);
    } else if (p0 == 0) {
        permute_keeplo1(oridim, perm, in, out);
    } else if (p0 == 1 && p1 == 0) {
        permute_easy_do(oridim, perm, in, out, d0);
    } else {
        load_tran_trans(oridim, perm, in, out, d0);
    }
}

// 支持任意大小的二维tensor转置
// 要求:vmem_width必须是128的倍数，除非vmem_width == hbm_width
//      vmem_height必须是128的倍数，除非vmem_height == hbm_height
//_hbm_height、_hbm_width可以是128的倍数，也可以不是
// 如果_hbm_width是128的倍数，则padding不会被置为0;否则会置为0
inline void Tranpose2d(SIM_X86::tensor hbm_in, SIM_X86::tensor in, SIM_X86::tensor out, SIM_X86::tensor hbm_out, int _hbm_height, int _hbm_width,
                       int vmem_height, int vmem_width) {
    int hbm_height = padding128(_hbm_height);
    int hbm_width = padding128(_hbm_width);
    for (int _h = 0; _h < _hbm_height; _h += vmem_height) {
        int h = min(_hbm_height - _h, vmem_height);
        for (int _w = 0; _w < _hbm_width; _w += vmem_width) {
            int w = min(_hbm_width - _w, vmem_width);
            load_mat_0123(hbm_in, in, 1, 1, _hbm_height, hbm_width, 0, 0, _h, _w, h, padding128(w));
            tile_trans_transfer(in, out, 0, h, w, 0);
//             __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, out, 1, 1));
            store_mat_0123(out, hbm_out, 1, 1, _hbm_width, hbm_height, 0, 0, _w, _h, w, padding128(h));
        }
    }
}

inline void Tranpose(SIM_X86::tensor hbm_in, SIM_X86::tensor in, SIM_X86::tensor out, SIM_X86::tensor hbm_out, int *HbmSize, int *VmemSize) {
    int MatInSize = padding128(HbmSize[0]) * HbmSize[1];
    int MatOutSize = padding128(HbmSize[1]) * HbmSize[0];
    int n = HbmSize[4] * HbmSize[3] * HbmSize[2];
    for (int i = 0; i < n; i++) {
        Tranpose2d(hbm_in + i * MatInSize / 32, in, out, hbm_out + i * MatOutSize / 32, HbmSize[1],
                   HbmSize[0], VmemSize[1], VmemSize[0]);
    }
}

// bf16版本
// 要求:vmem_width必须是256的倍数，除非vmem_width == hbm_width
//      vmem_height必须是256的倍数，除非vmem_height == hbm_height
inline void Tranpose2dBF16(SIM_X86::tensor hbm_in, SIM_X86::tensor in, SIM_X86::tensor out, SIM_X86::tensor hbm_out, int _hbm_height,
                           int _hbm_width, int vmem_height, int vmem_width) {
    int hbm_height = padding256(_hbm_height) / 2;
    int hbm_width  = padding256(_hbm_width) / 2;
    for (int _h = 0; _h < _hbm_height; _h += vmem_height) {
        int h = min(_hbm_height - _h, vmem_height);
        for (int _w = 0; _w < _hbm_width; _w += vmem_width * 2) {
            int w = min(_hbm_width - _w, vmem_width * 2);
            load_mat_0123(hbm_in, in, 1, 1, _hbm_height, hbm_width, 0, 0, _h, _w / 2, h, padding256(w) / 2);
            tile_trans_transfer_bf16(in, out, 0, h, w, 0, padding256(w) / 2, padding256(h) / 2);
//             __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, out, 1, 1));
            store_mat_0123(out, hbm_out, 1, 1, _hbm_width, hbm_height, 0, 0, _w, _h / 2, w,
                           padding256(h) / 2);
        }
    }
}

inline void TranposeBF16(SIM_X86::tensor hbm_in, SIM_X86::tensor in, SIM_X86::tensor out, SIM_X86::tensor hbm_out, int *HbmSize, int *VmemSize) {
    int MatInSize = padding256(HbmSize[0]) * HbmSize[1];
    int MatOutSize = padding256(HbmSize[1]) * HbmSize[0];
    int n = HbmSize[4] * HbmSize[3] * HbmSize[2];
    for (int i = 0; i < n; i++) {
        Tranpose2dBF16(hbm_in + i * MatInSize / 64, in, out, hbm_out + i * MatOutSize / 64, HbmSize[1],
                       HbmSize[0], VmemSize[1], VmemSize[0]);
    }
}

inline int CheckVmemWH(int d0, int d1, int VMEMSize) {
    int d0_128 = ALIGN128(d0);
    int d1_128 = ALIGN128(d1);
    return d0_128 * d1 + d1_128 * d0 > VMEMSize;
}

inline void CalVmemWH(int* vmem_w, int* vmem_h, int VMEMSize) {
    int d0 = *vmem_w;
    int d1 = *vmem_h;

    if (!CheckVmemWH(d0, d1, VMEMSize)) {
        *vmem_w = ALIGN128(d0);
        return;
    }

    if (d1 < 128) {
        int l = 1, r = d0;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWH(m, d1, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_w = ALIGN128(l);
    } else if (d0 < 128) {
        int l = 1, r = d1;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWH(d0, m, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_w = ALIGN128(d0);
        *vmem_h = l & 0xffffff80;
    } else {
        int h = 128;
        int l = 1, r = d0;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWH(m, h, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_w = ALIGN128(l);
        int w = *vmem_w;
        l = 128, r = d1;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWH(w, m, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_h = l & 0xffffff80;
    }
}

inline int CheckVmemWHBf16(int d0, int d1, int VMEMSize) {
    int d0_128 = ALIGN256(d0) / 2;
    int d1_128 = ALIGN256(d1) / 2;
    return d0_128 * d1 + d1_128 * d0 > VMEMSize;
}

inline void CalVmemWHBf16(int* vmem_w, int* vmem_h, int VMEMSize) {
    int d0 = *vmem_w;
    int d1 = *vmem_h;

    if (!CheckVmemWHBf16(d0, d1, VMEMSize)) {
        *vmem_w = ALIGN256(d0) / 2;
        return;
    }

    if (d1 <= 256) {
        int l = 1, r = d0;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWHBf16(m, d1, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_w = ALIGN256(l) / 2;
    } else if (d0 <= 256) {
        int l = 1, r = d1;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWHBf16(d0, m, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_w = ALIGN256(d0) / 2;
        *vmem_h = l & 0xffffff00;
    } else {
        int h = 256;
        int l = 1, r = d0;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWHBf16(m, h, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_w = ALIGN256(l) / 2;
        int w = ALIGN256(l);
        l = h, r = d1;
        while (l < r) {
            int m = (l + r) / 2 + 1;
            if (CheckVmemWHBf16(w, m, VMEMSize)) r = m - 1;
            else l = m;
        }
        *vmem_h = l & 0xffffff00;
    }
}

// [O, T, pR] => [O', R, pT], rd0 = d0 before padding
// perm[0] == 1 && perm[1] == 0
inline void permute_easy_do_big(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem,
                                    int *oridim, int *perm, int rd0, int VMEMSize, int *idx_origin, int BLACK) {

    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding128(newdim[0]);
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int idx[5];

    int vmem_w = rd0;
    int vmem_h = oridim[1];
    CalVmemWH(&vmem_w, &vmem_h, VMEMSize);

    SIM_X86::tensor output_vmem;
    output_vmem = tensor_slice(input0_vmem, vmem_h * vmem_w / 32);

    if (BLACK) {
        idx_origin[2]++;
        idx_origin[1] = 0;
        if (idx_origin[2] >= oridim[2]) {
            idx_origin[3]++;
            idx_origin[2] = 0;
            if (idx_origin[3] >= oridim[3]) {
                idx_origin[4]++;
                idx_origin[3] = 0;
            }
        }

        for (idx[4] = idx_origin[4]; idx[4] < oridim[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim[2]; ++idx[2]) {
                    idx_origin[2] = 0;

                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];
                    // int hbm_len = oridim[0] * oridim[1];

                    Tranpose2d(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32, oridim[1], rd0, vmem_h, vmem_w);
                }
            }
        }
    } else {
        for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
            idx_origin[4] = oridim[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                idx_origin[3] = oridim[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                    idx_origin[2] = oridim[2] - 1;

                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];
                    // int hbm_len = oridim[0] * oridim[1];

                    Tranpose2d(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32, oridim[1], rd0, vmem_h, vmem_w);
                }
            }
        }
    }
}

inline void permute_easy_do_big_bf16(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem,
                                    int *oridim, int *perm, int rd0, int VMEMSize, int *idx_origin, int BLACK) {
    oridim[0] = padding256(oridim[0]) / 2;
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding256(newdim[0]) / 2;
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int idx[5];

    int vmem_w = rd0;
    int vmem_h = oridim[1];
    CalVmemWHBf16(&vmem_w, &vmem_h, VMEMSize);

    SIM_X86::tensor output_vmem;
    output_vmem = tensor_slice(input0_vmem, vmem_w * vmem_h / 32);

    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);

    if (BLACK) {
        idx_origin[2]++;
        if (idx_origin[2] >= oridim[2]) {
            idx_origin[3]++;
            idx_origin[2] = 0;
            if (idx_origin[3] >= oridim[3]) {
                idx_origin[4]++;
                idx_origin[3] = 0;
            }
        }

        for (idx[4] = idx_origin[4]; idx[4] < oridim[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim[2]; ++idx[2]) {
                    idx_origin[2] = 0;

                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];

                    Tranpose2dBF16(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32, oridim[1], rd0, vmem_h, vmem_w);
                }
            }
        }
    } else {
        for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
            idx_origin[4] = oridim[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                idx_origin[3] = oridim[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                    idx_origin[2] = oridim[2] - 1;

                    int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                    int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];

                    Tranpose2dBF16(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32, oridim[1], rd0, vmem_h, vmem_w);
                }
            }
        }
    }
}

inline void Tranpose2dWithOffset(SIM_X86::tensor input0_hbm, SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem, SIM_X86::tensor output_hbm,
                                 int input0_stride, int output_stride,
                                 int _hbm_height, int _hbm_width, int vmem_height, int vmem_width) {
    // int hbm_height = padding128(_hbm_height);
    // int hbm_width = padding128(_hbm_width);

    for (int _h = 0; _h < _hbm_height; _h += vmem_height) {
        int h = min(_hbm_height - _h, vmem_height);
        for (int _w = 0; _w < _hbm_width; _w += vmem_width) {
            int w = min(_hbm_width - _w, vmem_width);

            if (1) {
                const int offset = _h * input0_stride + _w;
                for (int i = 0; i < padding128(w); i += 128) {
                    int sync = dlc_dma(tensor_slice(input0_hbm, offset / 32 + i / 32), D_HBM, tensor_slice(input0_vmem, i / 32), D_VMEM,
                                        h * 128, input0_stride, padding128(w), 128, 7);
                    dlc_sync(sync);
                }
            }

            tile_trans_transfer(input0_vmem, output_vmem, 0, h, w, 0);
//             __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output_vmem, 1, 1));

            if (1) {
                const int offset = _w * output_stride + _h;
                for (int i = 0; i < padding128(h); i += 128) {
                    int sync = dlc_dma(tensor_slice(output_vmem, i / 32), D_VMEM, tensor_slice(output_hbm, offset / 32 + i / 32), D_HBM,
                                        w * 128, padding128(h), output_stride, 128, 7);
                    dlc_sync(sync);
                }
            }
        }
    }
}

inline void Tranpose2dWithOffset_bf16(SIM_X86::tensor input0_hbm, SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem, SIM_X86::tensor output_hbm,
                                        int input0_stride, int output_stride,
                                        int _hbm_height, int _hbm_width, int vmem_height, int vmem_width) {
    for (int _h = 0; _h < _hbm_height; _h += vmem_height) {
        int h = min(_hbm_height - _h, vmem_height);
        for (int _w = 0; _w < _hbm_width; _w += vmem_width * 2) {
            int w = min(_hbm_width - _w, vmem_width * 2);

            if (1) {
                const int offset = _h * input0_stride + _w / 2;
                for (int i = 0; i < padding256(w) / 2; i += 128) {
                    int sync = dlc_dma(tensor_slice(input0_hbm, offset / 32 + i / 32), D_HBM, tensor_slice(input0_vmem, i / 32), D_VMEM,
                                        h * 128, input0_stride, padding256(w) / 2, 128, 7);
                    dlc_sync(sync);
                }
            }

            tile_trans_transfer_bf16(input0_vmem, output_vmem, 0, h, w, 0, padding256(w) / 2, padding256(h) / 2);
//             __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output_vmem, 1, 1));

            if (1) {
                const int offset = _w * output_stride + _h / 2;
                for (int i = 0; i < padding256(h) / 2; i += 128) {
                    int sync = dlc_dma(tensor_slice(output_vmem, i / 32), D_VMEM, tensor_slice(output_hbm, offset / 32 + i / 32), D_HBM,
                                        w * 128, padding256(h) / 2, output_stride, 128, 7);
                    dlc_sync(sync);
                }
            }
        }
    }
}

inline void permute_full(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize,
                         int *oridim, int *perm, int rd0, int *idx_origin, int BLACK) {
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding128(newdim[0]);
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int oridim_new[5];
    for (int i = 0; i < 5; ++i) oridim_new[i] = oridim[i];

    int input0_stride = 0;
    int output_stride = 1;

    if (perm[0] == 4) {
        oridim_new[4] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2] * oridim[3];
    } else if (perm[0] == 3) {
        oridim_new[3] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2];
    } else if (perm[0] == 2) {
        oridim_new[2] = 1;
        input0_stride = oridim[0] * oridim[1];
    } else if (perm[0] == 1) {
        oridim_new[1] = 1;
        input0_stride = oridim[0];
    }

    for (int i = 0; i < 5; ++i) {
        if (perm[i] != 0) output_stride *= newdim[i];
        else break;
    }

    int hbm_h = rodim[perm[0]];
    int hbm_w = rd0;
    int vmem_w = rd0;
    int vmem_h = oridim[perm[0]];
    CalVmemWH(&vmem_w, &vmem_h, VMEMSize);

    SIM_X86::tensor output_vmem;
    output_vmem = input0_vmem + vmem_h * vmem_w / 32;

    if (BLACK) {
        idx_origin[1]++;
        if (idx_origin[1] >= oridim_new[1]) {
            idx_origin[2]++;
            idx_origin[1] = 0;
            if (idx_origin[2] >= oridim_new[2]) {
                idx_origin[3]++;
                idx_origin[2] = 0;
                if (idx_origin[3] >= oridim_new[3]) {
                    idx_origin[4]++;
                    idx_origin[3] = 0;
                }
            }
        }

        int idx[5] = {0, 0, 0, 0, 0};
        for (idx[4] = idx_origin[4]; idx[4] < oridim_new[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim_new[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim_new[2]; ++idx[2]) {
                    idx_origin[2] = 0;
                    for (idx[1] = idx_origin[1]; idx[1] < oridim_new[1]; ++idx[1]) {
                        idx_origin[1] = 0;

                        int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                        Tranpose2dWithOffset(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32,
                                            input0_stride, output_stride, hbm_h, hbm_w, vmem_h, vmem_w);
                    }
                }
            }
        }
    } else {
        int idx[5] = {0, 0, 0, 0, 0};
        for (idx[4] = idx_origin[4]; idx[4] >=0; --idx[4]) {
            idx_origin[4] = oridim_new[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >=0; --idx[3]) {
                idx_origin[3] = oridim_new[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >=0; --idx[2]) {
                    idx_origin[2] = oridim_new[2] - 1;
                    for (idx[1] = idx_origin[1]; idx[1] >=0; --idx[1]) {
                        idx_origin[1] = oridim_new[1] - 1;

                        int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                        Tranpose2dWithOffset(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32,
                                            input0_stride, output_stride, hbm_h, hbm_w, vmem_h, vmem_w);
                    }
                }
            }
        }
    }
}

inline void permute_full_bf16(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize,
                         int *oridim, int *perm, int rd0, int *idx_origin, int BLACK) {
    oridim[0] = padding256(oridim[0]) / 2;
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding256(newdim[0]) / 2;
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int oridim_new[5];
    for (int i = 0; i < 5; ++i) oridim_new[i] = oridim[i];

    int input0_stride = 0;
    int output_stride = 1;

    if (perm[0] == 4) {
        oridim_new[4] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2] * oridim[3];
    } else if (perm[0] == 3) {
        oridim_new[3] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2];
    } else if (perm[0] == 2) {
        oridim_new[2] = 1;
        input0_stride = oridim[0] * oridim[1];
    } else if (perm[0] == 1) {
        oridim_new[1] = 1;
        input0_stride = oridim[0];
    }

    for (int i = 0; i < 5; ++i) {
        if (perm[i] != 0) output_stride *= newdim[i];
        else break;
    }

    int hbm_h = rodim[perm[0]];
    int hbm_w = rd0;

    int vmem_w = rd0;
    int vmem_h = oridim[perm[0]];
    CalVmemWHBf16(&vmem_w, &vmem_h, VMEMSize);

    SIM_X86::tensor output_vmem;
    output_vmem = tensor_slice(input0_vmem, vmem_w * vmem_h / 32);

    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);

    if (BLACK) {
        idx_origin[1]++;
        if (idx_origin[1] >= oridim_new[1]) {
            idx_origin[2]++;
            idx_origin[1] = 0;
            if (idx_origin[2] >= oridim_new[2]) {
                idx_origin[3]++;
                idx_origin[2] = 0;
                if (idx_origin[3] >= oridim_new[3]) {
                    idx_origin[4]++;
                    idx_origin[3] = 0;
                }
            }
        }

        int idx[5] = {0, 0, 0, 0, 0};
        for (idx[4] = idx_origin[4]; idx[4] < oridim_new[4]; ++idx[4]) {
            idx_origin[4] = 0;
            for (idx[3] = idx_origin[3]; idx[3] < oridim_new[3]; ++idx[3]) {
                idx_origin[3] = 0;
                for (idx[2] = idx_origin[2]; idx[2] < oridim_new[2]; ++idx[2]) {
                    idx_origin[2] = 0;
                    for (idx[1] = idx_origin[1]; idx[1] < oridim_new[1]; ++idx[1]) {
                        idx_origin[1] = 0;

                        int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                        Tranpose2dWithOffset_bf16(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32,
                                            input0_stride, output_stride, hbm_h, hbm_w, vmem_h, vmem_w);
                    }
                }
            }
        }
    } else {
        int idx[5] = {0, 0, 0, 0, 0};
        for (idx[4] = idx_origin[4]; idx[4] >=0; --idx[4]) {
            idx_origin[4] = oridim_new[4] - 1;
            for (idx[3] = idx_origin[3]; idx[3] >=0; --idx[3]) {
                idx_origin[3] = oridim_new[3] - 1;
                for (idx[2] = idx_origin[2]; idx[2] >=0; --idx[2]) {
                    idx_origin[2] = oridim_new[2] - 1;
                    for (idx[1] = idx_origin[1]; idx[1] >=0; --idx[1]) {
                        idx_origin[1] = oridim_new[1] - 1;

                        int offset_base_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                        int offset_base_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                        Tranpose2dWithOffset_bf16(input0_hbm + offset_base_src / 32, input0_vmem, output_vmem, output_hbm + offset_base_dst / 32,
                                            input0_stride, output_stride, hbm_h, hbm_w, vmem_h, vmem_w);
                    }
                }
            }
        }
    }
}

inline int* GetHalfIdx(int d0, int d1, int d2, int d3, int d4, int *idx) {
    int len = d4 * d3 * d2 * d1 * d0 / 2;
    if ((d4 & 1) == 0) {
        idx[4] = d4 / 2 - 1;
        idx[3] = d3 - 1;
        idx[2] = d2 - 1;
        idx[1] = d1 - 1;
        idx[0] = d0 - 1;
    } else {
        if (d3 * d2 * d1 * d0 > 1) {
            idx[4] = len / (d3 * d2 * d1 * d0);
            len -= idx[4] * (d3 * d2 * d1 * d0);
            if ((d3 & 1) == 0) {
                idx[3] = d3 / 2 - 1;
                idx[2] = d2 - 1;
                idx[1] = d1 - 1;
                idx[0] = d0 - 1;
            } else {
                if (d2 * d1 * d0 > 1) {
                    idx[3] = len / (d2 * d1 * d0);
                    len -= idx[3] * (d2 * d1 * d0);
                    if ((d2 & 1) == 0) {
                        idx[2] = d2 / 2 - 1;
                        idx[1] = d1 - 1;
                        idx[0] = d0 - 1;
                    } else {
                        if (d1 * d0 > 1) {
                            idx[2] = len / (d1 * d0);
                            len -= idx[2] * (d1 * d0);
                            if ((d1 & 1) == 0) {
                                idx[1] = d1 / 2 - 1;
                                idx[0] = d0 - 1;
                            } else {
                                if (d0 > 1) {
                                    idx[1] = len / d0;
                                    len -= idx[1] * d0;
                                    idx[0] = len - 1;
                                } else {
                                    idx[1] = len - 1;
                                }
                            }
                        } else {
                            idx[2] = len - 1;
                        }
                    }
                } else {
                    idx[3] = len - 1;
                }
            }
        } else {
            idx[4] = len - 1;
        }
    }

    return idx;
}

inline void Permute4D_DivideDim(int *DMA_COUNT_MIN, int *ROW, int *dim, int *perm, int divide,
                                int *H, int *W, int* ndim, int *nsrc_stride, int *ndst_stride,
                                int *src_stride, int *dst_stride, int *VERTICAL_FOR_COUNT, const int BF16) {
    int idx = 4;
    int group_length = 0;
    int group_size[5] = {0};
    int group_size_sum = dim[perm[4]];
    int group_id[5] = {5};
    int group_id_length = 0;
    int max_group_size = 0;
    int max_group_size_id = 0;

    while (idx > divide) {
        if (perm[idx] - 1 == perm[idx - 1]) {
            group_size_sum *= dim[perm[idx - 1]];
            group_id[perm[idx - 1]] = group_id_length;
            idx --;
        } else {
            if (max_group_size < group_size_sum) {
                max_group_size = group_size_sum;
                max_group_size_id = group_id_length;
            }
            group_size[4 - group_length] = group_size_sum;
            group_size_sum = dim[perm[idx - 1]];
            group_length ++;
            idx --;
            group_id_length ++;
            group_id[perm[idx]] = group_id_length;
        }
    }
    group_size[4 - group_length] = group_size_sum;
    group_length ++;
    if (max_group_size < group_size_sum) {
        max_group_size = group_size_sum;
        max_group_size_id = group_id_length;
    }

    int dim0 = (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0]));
    for (int i = 1; i < divide; ++i) dim0 *= dim[i];

    int dma_count = dim0 / 128;
    for (int i = divide; i < 5; ++i) {
        if (group_id[i] != max_group_size_id) dma_count *= dim[i];
    }

    if (dma_count >= *DMA_COUNT_MIN) return;

    *DMA_COUNT_MIN = dma_count;
    *ROW = 0;
    *H = max_group_size;
    *W = dim0;

    int max_group_size_member_count = 0;
    for (int i = divide; i < 5; ++i) max_group_size_member_count += (group_id[i] == max_group_size_id);
    *VERTICAL_FOR_COUNT = 5 - divide - max_group_size_member_count;

    int dim_stride[5] = {1,
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])),
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1],
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2],
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2] * dim[3]};
    int perm_dim[5] = {(BF16 ? ALIGN256(dim[perm[0]]) / 2 : ALIGN128(dim[perm[0]])), dim[perm[1]], dim[perm[2]], dim[perm[3]], dim[perm[4]]};
    int perm_group_id[5] = {0, group_id[perm[1]], group_id[perm[2]], group_id[perm[3]], group_id[perm[4]]};
    int perm_dim_stride[5] = {1,
                            perm_dim[0],
                            perm_dim[0] * perm_dim[1],
                            perm_dim[0] * perm_dim[1] * perm_dim[2],
                            perm_dim[0] * perm_dim[1] * perm_dim[2] * perm_dim[3]};

    *src_stride = dim0;
    for (int i = divide; i < 5; ++i) {
        if (group_id[i] != max_group_size_id) *src_stride *= dim[i];
        else break;
    }
    *dst_stride = dim0;
    for (int i = divide; i < 5; ++i) {
        if (perm_group_id[i] != max_group_size_id) *dst_stride *= perm_dim[i];
        else break;
    }

    int rperm[5];
    for (int i = 0; i < 5; ++i) rperm[perm[i]] = i;

    int stride_len = 0;
    for (int i = 4; i > 0; --i) {
        if (group_id[i] != max_group_size_id) {
            nsrc_stride[4 - stride_len] = dim_stride[i];
            ndst_stride[4 - stride_len] = perm_dim_stride[rperm[i]];
            ndim[4 -stride_len] = dim[i];
            stride_len ++;
        }
    }
}

inline void Permute5D_HBM2HBM_2XYS(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize, int HbmLen) {
    if (get_device_id()) {
        int offset = HbmLen / 128 / 2 * 128;
        HbmLen = HbmLen - offset;
        input0_hbm = tensor_slice(input0_hbm, offset / 32);
        output_hbm = tensor_slice(output_hbm, offset / 32);
    } else {
        HbmLen = HbmLen / 128 / 2 * 128;
    }

    int diff_addr = ((int)(input0_hbm) - (int)(output_hbm)) * 32 * 4;

    if (diff_addr % 2048 == 0) {
        int handle = dlc_dma(input0_hbm, D_HBM, output_hbm, D_HBM, HbmLen, 128, 128, 128, 7);
        dlc_sync(handle);
    } else {
        for (int len = 0; len < HbmLen; len += VMEMSize) {
            int VMEMsize = min(HbmLen - len, VMEMSize);

            int handle1 = dlc_dma(tensor_slice(input0_hbm, len / 32), D_HBM,
                                input0_vmem, D_VMEM, VMEMsize, 128, 128, 128, 7);
            dlc_sync(handle1);

            int handle2 = dlc_dma(input0_vmem, D_VMEM,
                                tensor_slice(output_hbm, len / 32), D_HBM,
                                VMEMsize, 128, 128, 128, 7);
            dlc_sync(handle2);
        }
    }
}

inline void Permute4D_HBM2HBM(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize, 
                              int HbmLen, int src_stride, int dst_stride) {
    int diff_addr = ((int)(input0_hbm) - (int)(output_hbm)) * 32 * 4;
    int diff_stride = (src_stride - dst_stride);

    if (diff_addr % 2048 == 0 && diff_stride % 512 == 0) {
        int handle = dlc_dma(input0_hbm, D_HBM, output_hbm, D_HBM, HbmLen, src_stride, dst_stride, 128, 7);
        dlc_sync(handle);
    } else {
        for (int len = 0; len < HbmLen; len += VMEMSize) {
            int VMEMsize = min(HbmLen - len, VMEMSize);

            int handle1 = dlc_dma(tensor_slice(input0_hbm, len / 32), D_HBM,
                                input0_vmem, D_VMEM, VMEMsize, src_stride, 128, 128, 7);
            dlc_sync(handle1);

            int handle2 = dlc_dma(input0_vmem, D_VMEM,
                                tensor_slice(output_hbm, len / 32), D_HBM,
                                VMEMsize, 128, dst_stride, 128, 7);
            dlc_sync(handle2);
        }
    }
}

inline void Permute4D_HBM_ROW_2XYS(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize,
                          int *dim, int *perm, int ROW_DIM0_POS, const int BF16) {
    const int XYS1 = get_device_id();

    int dim_stride[5] = {1,
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])),
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1],
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2],
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2] * dim[3]};
    int perm_dim[5] = {BF16 ? ALIGN256(dim[perm[0]]) / 2 : ALIGN128(dim[perm[0]]), dim[perm[1]], dim[perm[2]], dim[perm[3]], dim[perm[4]]};
    int perm_dim_stride[5] = {1,
                            perm_dim[0],
                            perm_dim[0] * perm_dim[1],
                            perm_dim[0] * perm_dim[1] * perm_dim[2],
                            perm_dim[0] * perm_dim[1] * perm_dim[2] * perm_dim[3]};

    if (ROW_DIM0_POS == 2) {
        int HbmLen = (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2];
        int idx_origin[5] = {0, 0, 0, 0, 0};
        GetHalfIdx(1, 1, 1, dim[3], dim[4], idx_origin);
        int idx[5];

        if (XYS1) {
            idx_origin[3]++;
            if (idx_origin[3] >= dim[3]) {
                idx_origin[4]++;
                idx_origin[3] = 0;
            }

            for (idx[4] = idx_origin[4]; idx[4] < dim[4]; ++idx[4]) {
                for (idx[3] = idx_origin[3]; idx[3] < dim[3]; ++idx[3]) {
                    idx_origin[3] = 0;

                    int src_offset = idx[4] * dim_stride[4] + idx[3] * dim_stride[3];
                    int dst_offset = idx[perm[4]] * perm_dim_stride[4] + idx[perm[3]] * perm_dim_stride[3];

                    Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32),
                                        tensor_slice(output_hbm, dst_offset / 32),
                                        input0_vmem, VMEMSize, HbmLen, 128, 128);
                }
            }
        } else {
            for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
                for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                    idx_origin[3] = dim[3] - 1;

                    int src_offset = idx[4] * dim_stride[4] + idx[3] * dim_stride[3];
                    int dst_offset = idx[perm[4]] * perm_dim_stride[4] + idx[perm[3]] * perm_dim_stride[3];

                    Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32),
                                        tensor_slice(output_hbm, dst_offset / 32),
                                        input0_vmem, VMEMSize, HbmLen, 128, 128);
                }
            }
        }
    } else if (ROW_DIM0_POS == 1) {
        int HbmLen = (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1];
        int idx_origin[5] = {0, 0, 0, 0, 0};
        GetHalfIdx(1, 1, dim[2], dim[3], dim[4], idx_origin);
        int idx[5];

        if (XYS1) {
            idx_origin[2]++;
            if (idx_origin[2] >= dim[2]) {
                idx_origin[3]++;
                idx_origin[2] = 0;
                if (idx_origin[3] >= dim[3]) {
                    idx_origin[4]++;
                    idx_origin[3] = 0;
                }
            }

            for (idx[4] = idx_origin[4]; idx[4] < dim[4]; ++idx[4]) {
                for (idx[3] = idx_origin[3]; idx[3] < dim[3]; ++idx[3]) {
                    idx_origin[3] = 0;
                    for (idx[2] = idx_origin[2]; idx[2] < dim[2]; ++idx[2]) {
                        idx_origin[2] = 0;

                        int src_offset = idx[4] * dim_stride[4] + idx[3] * dim_stride[3] + idx[2] * dim_stride[2];
                        int dst_offset = idx[perm[4]] * perm_dim_stride[4] + idx[perm[3]] * perm_dim_stride[3] + idx[perm[2]] * perm_dim_stride[2];

                        Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32),
                                            tensor_slice(output_hbm, dst_offset / 32),
                                            input0_vmem, VMEMSize, HbmLen, 128, 128);
                    }
                }
            }
        } else {
            for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
                for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                    idx_origin[3] = dim[3] - 1;
                    for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                        idx_origin[2] = dim[2] - 1;

                        int src_offset = idx[4] * dim_stride[4] + idx[3] * dim_stride[3] + idx[2] * dim_stride[2];
                        int dst_offset = idx[perm[4]] * perm_dim_stride[4] + idx[perm[3]] * perm_dim_stride[3] + idx[perm[2]] * perm_dim_stride[2];

                        Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32),
                                            tensor_slice(output_hbm, dst_offset / 32),
                                            input0_vmem, VMEMSize, HbmLen, 128, 128);
                    }
                }
            }
        }
    } else if (ROW_DIM0_POS == 0) {
        int HbmLen = (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0]));
        int idx_origin[5] = {0, 0, 0, 0, 0};
        GetHalfIdx(1, dim[1], dim[2], dim[3], dim[4], idx_origin);
        int idx[5];

        if (XYS1) {
            idx_origin[1]++;
            if (idx_origin[1] >= dim[1]) {
                idx_origin[2]++;
                idx_origin[1] = 0;
                if (idx_origin[2] >= dim[2]) {
                    idx_origin[3]++;
                    idx_origin[2] = 0;
                    if (idx_origin[3] >= dim[3]) {
                        idx_origin[4]++;
                        idx_origin[3] = 0;
                    }
                }
            }

            for (idx[4] = idx_origin[4]; idx[4] < dim[4]; ++idx[4]) {
                for (idx[3] = idx_origin[3]; idx[3] < dim[3]; ++idx[3]) {
                    idx_origin[3] = 0;
                    for (idx[2] = idx_origin[2]; idx[2] < dim[2]; ++idx[2]) {
                        idx_origin[2] = 0;
                        for (idx[1] = idx_origin[1]; idx[1] < dim[1]; ++idx[1]) {
                            idx_origin[1] = 0;

                            int src_offset = idx[4] * dim_stride[4] +
                                                idx[3] * dim_stride[3] +
                                                idx[2] * dim_stride[2] +
                                                idx[1] * dim_stride[1];
                            int dst_offset = idx[perm[4]] * perm_dim_stride[4] +
                                                idx[perm[3]] * perm_dim_stride[3] +
                                                idx[perm[2]] * perm_dim_stride[2] +
                                                idx[perm[1]] * perm_dim_stride[1];

                            Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32),
                                                tensor_slice(output_hbm, dst_offset / 32),
                                                input0_vmem, VMEMSize, HbmLen, 128, 128);
                        }
                    }
                }
            }
        } else {
            for (idx[4] = idx_origin[4]; idx[4] >= 0; --idx[4]) {
                for (idx[3] = idx_origin[3]; idx[3] >= 0; --idx[3]) {
                    idx_origin[3] = dim[3] - 1;
                    for (idx[2] = idx_origin[2]; idx[2] >= 0; --idx[2]) {
                        idx_origin[2] = dim[2] - 1;
                        for (idx[1] = idx_origin[1]; idx[1] >= 0; --idx[1]) {
                            idx_origin[1] = dim[1] - 1;

                            int src_offset = idx[4] * dim_stride[4] +
                                                idx[3] * dim_stride[3] +
                                                idx[2] * dim_stride[2] +
                                                idx[1] * dim_stride[1];
                            int dst_offset = idx[perm[4]] * perm_dim_stride[4] +
                                                idx[perm[3]] * perm_dim_stride[3] +
                                                idx[perm[2]] * perm_dim_stride[2] +
                                                idx[perm[1]] * perm_dim_stride[1];

                            Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32),
                                                tensor_slice(output_hbm, dst_offset / 32),
                                                input0_vmem, VMEMSize, HbmLen, 128, 128);
                        }
                    }
                }
            }
        }
    }
    
    sync_device();
}

inline void Permute4D_HBM_VERTICAL_2XYS(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize,
                          int* ndim, int* nsrc_stride, int* ndst_stride,
                          int H, int W, int VERTICAL_FOR_COUNT, int src_stride, int dst_stride) {
    int offset = W / 128 / 2 * 128;
    int W_start = 0;
    int W_end = 0;

    if (get_device_id()) {
        W_start = offset;
        W_end = W;
    } else {
        W_start = 0;
        W_end = offset;
    }

    if (VERTICAL_FOR_COUNT == 1) {
        for (int dim4 = 0; dim4 < ndim[4]; ++dim4) {
            int src_offset = dim4 * nsrc_stride[4];
            int dst_offset = dim4 * ndst_stride[4];

            for (int step = W_start; step < W_end; step += 128) {
                for (int len = 0; len < H * 128; len += VMEMSize) {
                    int VMEMsize = min(H * 128 - len, VMEMSize);

                    Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32 + step / 32 + len / 128 * src_stride / 32),
                                        tensor_slice(output_hbm, dst_offset / 32 + step / 32 + len / 128 * dst_stride / 32),
                                        input0_vmem, VMEMSize, VMEMsize, src_stride, dst_stride);
                }
            }
        }
    } else if (VERTICAL_FOR_COUNT == 2) {
        for (int dim4 = 0; dim4 < ndim[4]; ++dim4) {
            for (int dim3 = 0; dim3 < ndim[3]; ++dim3) {
                int src_offset = dim4 * nsrc_stride[4] + dim3 * nsrc_stride[3];
                int dst_offset = dim4 * ndst_stride[4] + dim3 * ndst_stride[3];

                for (int step = W_start; step < W_end; step += 128) {
                    for (int len = 0; len < H * 128; len += VMEMSize) {
                        int VMEMsize = min(H * 128 - len, VMEMSize);

                        Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32 + step / 32 + len / 128 * src_stride / 32),
                                            tensor_slice(output_hbm, dst_offset / 32 + step / 32 + len / 128 * dst_stride / 32),
                                            input0_vmem, VMEMSize, VMEMsize, src_stride, dst_stride);
                    }
                }
            }
        }
    } else if (VERTICAL_FOR_COUNT == 3) {
        for (int dim4 = 0; dim4 < ndim[4]; ++dim4) {
            for (int dim3 = 0; dim3 < ndim[3]; ++dim3) {
                for (int dim2 = 0; dim2 < ndim[2]; ++dim2) {
                    int src_offset = dim4 * nsrc_stride[4] + dim3 * nsrc_stride[3] + dim2 * nsrc_stride[2];
                    int dst_offset = dim4 * ndst_stride[4] + dim3 * ndst_stride[3] + dim2 * ndst_stride[2];

                    for (int step = W_start; step < W_end; step += 128) {
                        for (int len = 0; len < H * 128; len += VMEMSize) {
                            int VMEMsize = min(H * 128 - len, VMEMSize);

                            Permute4D_HBM2HBM(tensor_slice(input0_hbm, src_offset / 32 + step / 32 + len / 128 * src_stride / 32),
                                                tensor_slice(output_hbm, dst_offset / 32 + step / 32 + len / 128 * dst_stride / 32),
                                                input0_vmem, VMEMSize, VMEMsize, src_stride, dst_stride);
                        }
                    }
                }
            }
        }
    }
}

inline void Permute4D_VMEM2VMEM(SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem, int DmaLength, int src_stride_128, int dst_stride_128) {
    /* if load store is quicker than dma*/
    if (DmaLength / 1024 * 5 < (225 + 89 + 166 + DmaLength / 128)) {
        int len1024 = DmaLength / 1024 * 1024;
        int vs = 0;
        for (; vs < len1024; vs += 1024) {
            store8_128_stride_stmk(vs / 128 * dst_stride_128 / 32, dst_stride_128 / 128, output_vmem,
                                 load8_128_stride_ldmk(vs / 128 * src_stride_128 / 32, src_stride_128 / 128, input0_vmem, 255), 255);
        }
        if (vs < DmaLength) {
            int ldst_msk = pre_exp2((DmaLength - vs) / 128);

            store8_128_stride_stmk(vs / 128 * dst_stride_128 / 32, dst_stride_128 / 128, output_vmem,
                                 load8_128_stride_ldmk(vs / 128 * src_stride_128 / 32, src_stride_128 / 128, input0_vmem, ldst_msk), ldst_msk);
        }
    } else {
        int handle = dlc_dma(input0_vmem, D_VMEM, output_vmem, D_VMEM, DmaLength, src_stride_128,
                             dst_stride_128, 128, 7);
        dlc_sync(handle);
    }
}

inline void Permute4D_VMEM_ROW(SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem,
                               int *dim, int *perm, int ROW_DIM0_POS, const int BF16) {
    int dim_stride[5] = {1,
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])),
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1],
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2],
                       (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2] * dim[3]};
    int perm_dim[5] = {BF16 ? ALIGN256(dim[perm[0]]) / 2 : ALIGN128(dim[perm[0]]), dim[perm[1]], dim[perm[2]], dim[perm[3]], dim[perm[4]]};
    int perm_dim_stride[5] = {1,
                            perm_dim[0],
                            perm_dim[0] * perm_dim[1],
                            perm_dim[0] * perm_dim[1] * perm_dim[2],
                            perm_dim[0] * perm_dim[1] * perm_dim[2] * perm_dim[3]};

    if (ROW_DIM0_POS == 2) {
        int DmaLength = (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1] * dim[2];
        int idx[5];

        for (idx[4] = 0; idx[4] < dim[4]; ++idx[4]) {
            for (idx[3] = 0; idx[3] < dim[3]; ++idx[3]) {

                int src_offset = idx[4] * dim_stride[4] + idx[3] * dim_stride[3];
                int dst_offset = idx[perm[4]] * perm_dim_stride[4] + idx[perm[3]] * perm_dim_stride[3];

                Permute4D_VMEM2VMEM(tensor_slice(input0_vmem, src_offset / 32),
                                    tensor_slice(output_vmem, dst_offset / 32),
                                    DmaLength, 128, 128);
            }
        }
    } else if (ROW_DIM0_POS == 1) {
        int DmaLength = (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0])) * dim[1];
        int idx[5];

        for (idx[4] = 0; idx[4] < dim[4]; ++idx[4]) {
            for (idx[3] = 0; idx[3] < dim[3]; ++idx[3]) {
                for (idx[2] = 0; idx[2] < dim[2]; ++idx[2]) {

                    int src_offset = idx[4] * dim_stride[4] + idx[3] * dim_stride[3] + idx[2] * dim_stride[2];
                    int dst_offset = idx[perm[4]] * perm_dim_stride[4] + idx[perm[3]] * perm_dim_stride[3] + idx[perm[2]] * perm_dim_stride[2];

                    Permute4D_VMEM2VMEM(tensor_slice(input0_vmem, src_offset / 32),
                                        tensor_slice(output_vmem, dst_offset / 32),
                                        DmaLength, 128, 128);
                }
            }
        }
    } else if (ROW_DIM0_POS == 0) {
        int DmaLength = (BF16 ? ALIGN256(dim[0]) / 2 : ALIGN128(dim[0]));
        int idx[5];

        for (idx[4] = 0; idx[4] < dim[4]; ++idx[4]) {
            for (idx[3] = 0; idx[3] < dim[3]; ++idx[3]) {
                for (idx[2] = 0; idx[2] < dim[2]; ++idx[2]) {
                    for (idx[1] = 0; idx[1] < dim[1]; ++idx[1]) {

                        int src_offset = idx[4] * dim_stride[4] +
                                            idx[3] * dim_stride[3] +
                                            idx[2] * dim_stride[2] +
                                            idx[1] * dim_stride[1];
                        int dst_offset = idx[perm[4]] * perm_dim_stride[4] +
                                            idx[perm[3]] * perm_dim_stride[3] +
                                            idx[perm[2]] * perm_dim_stride[2] +
                                            idx[perm[1]] * perm_dim_stride[1];

                        Permute4D_VMEM2VMEM(tensor_slice(input0_vmem, src_offset / 32),
                                            tensor_slice(output_vmem, dst_offset / 32),
                                            DmaLength, 128, 128);
                    }
                }
            }
        }
    }
}

inline void Permute4D_VMEM_VERTICAL(SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem,
                                    int* ndim, int* nsrc_stride, int* ndst_stride,
                                    int H, int W, int VERTICAL_FOR_COUNT, int src_stride, int dst_stride) {
    if (VERTICAL_FOR_COUNT == 1) {
        for (int dim4 = 0; dim4 < ndim[4]; ++dim4) {
            int src_offset = dim4 * nsrc_stride[4];
            int dst_offset = dim4 * ndst_stride[4];

            for (int step = 0; step < W; step += 128) {
                Permute4D_VMEM2VMEM(tensor_slice(input0_vmem, src_offset / 32 + step / 32),
                                    tensor_slice(output_vmem, dst_offset / 32 + step / 32),
                                    H * 128, src_stride, dst_stride);
            }
        }
    } else if (VERTICAL_FOR_COUNT == 2) {
        for (int dim4 = 0; dim4 < ndim[4]; ++dim4) {
            for (int dim3 = 0; dim3 < ndim[3]; ++dim3) {
                int src_offset = dim4 * nsrc_stride[4] + dim3 * nsrc_stride[3];
                int dst_offset = dim4 * ndst_stride[4] + dim3 * ndst_stride[3];

                for (int step = 0; step < W; step += 128) {
                    Permute4D_VMEM2VMEM(tensor_slice(input0_vmem, src_offset / 32 + step / 32),
                                        tensor_slice(output_vmem, dst_offset / 32 + step / 32),
                                        H * 128, src_stride, dst_stride);
                }
            }
        }
    } else if (VERTICAL_FOR_COUNT == 3) {
        for (int dim4 = 0; dim4 < ndim[4]; ++dim4) {
            for (int dim3 = 0; dim3 < ndim[3]; ++dim3) {
                for (int dim2 = 0; dim2 < ndim[2]; ++dim2) {
                    int src_offset = dim4 * nsrc_stride[4] + dim3 * nsrc_stride[3] + dim2 * nsrc_stride[2];
                    int dst_offset = dim4 * ndst_stride[4] + dim3 * ndst_stride[3] + dim2 * ndst_stride[2];

                    for (int step = 0; step < W; step += 128) {
                        Permute4D_VMEM2VMEM(tensor_slice(input0_vmem, src_offset / 32 + step / 32),
                                            tensor_slice(output_vmem, dst_offset / 32 + step / 32),
                                            H * 128, src_stride, dst_stride);
                    }
                }
            }
        }
    }
}

inline void _permute_hbm(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize, int *shape, int *perm) {
    int dim[5] = {shape[0], shape[1], shape[2], shape[3], shape[4]};
    int oridim[5] = {ALIGN128(shape[0]), shape[1], shape[2], shape[3], shape[4]};

    int dim0_128 = ALIGN128(shape[0]);
    int dim1 = dim[1] * dim[2] * dim[3] * dim[4];

    int d4 = shape[4];
    int d3 = shape[3];
    int d2 = shape[2];
    int d1 = shape[1];
    int d0 = shape[0];

    int perm_5d = (perm[0] ^ 0) + (perm[1] ^ 1) + (perm[2] ^ 2) + (perm[3] ^ 3) + (perm[4] ^ 4);
    if (perm_5d == 0) {
        Permute5D_HBM2HBM_2XYS(input0_hbm, output_hbm, input0_vmem, VMEMSize, dim0_128 * dim1);
    } else if (perm[0] == 0) {
        int row_stride[5] = {dim[1] * dim[2] * dim[3] * dim[4], dim[2] * dim[3] * dim[4], dim[3] * dim[4], dim[4], 1};

        int DMA_COUNT_MIN = row_stride[0];
        int ROW = 1; // row = 1, vertical = 0
        int ROW_DIM0_POS = 0;

        int H = 0;
        int W = 0;
        int ndim[5];
        int nsrc_stride[5];
        int ndst_stride[5];
        int src_stride = 1;
        int dst_stride = 1;
        int VERTICAL_FOR_COUNT;

        Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 1,  &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 0);

        if (perm[1] == 1) {
            if (row_stride[1] < DMA_COUNT_MIN) {
                DMA_COUNT_MIN = row_stride[1];
                ROW = 1;
                ROW_DIM0_POS = 1;
            }
            Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 2,  &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 0);

            if (perm[2] == 2) {
                if (row_stride[2] < DMA_COUNT_MIN) {
                    DMA_COUNT_MIN = row_stride[2];
                    ROW = 1;
                    ROW_DIM0_POS = 2;
                }
                Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 3, &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 0);
            }
        }

        if (ROW) {
            Permute4D_HBM_ROW_2XYS(input0_hbm, output_hbm, input0_vmem, VMEMSize, dim, perm, ROW_DIM0_POS, 0);
        } else {
            Permute4D_HBM_VERTICAL_2XYS(input0_hbm, output_hbm, input0_vmem, VMEMSize, ndim, 
                                nsrc_stride, ndst_stride, H, W, VERTICAL_FOR_COUNT, src_stride, dst_stride);
        }
    } else if (perm[0] == 1 && perm[1] == 0) {
        int idx[5] = {0, 0, 0, 0, 0};
        GetHalfIdx(1, 1, d2, d3, d4, idx);

        int cnt = d2 * d3 * d4 / 2;
        if (get_device_id()) {
            permute_easy_do_big(input0_hbm, output_hbm, input0_vmem, oridim, perm, d0, VMEMSize, idx, 1);
        } else if (cnt) { // if len > 0 then do
            permute_easy_do_big(input0_hbm, output_hbm, input0_vmem, oridim, perm, d0, VMEMSize, idx, 0);
        }

        sync_device();
    } else {
        if (perm[0] == 4) d4 = 1;
        else if (perm[0] == 3) d3 = 1;
        else if (perm[0] == 2) d2 = 1;
        else if (perm[0] == 1) d1 = 1;

        int idx[5] = {0, 0, 0, 0, 0};
        GetHalfIdx(1, d1, d2, d3, d4, idx);

        int cnt = d1 * d2 * d3 * d4 / 2;
        if (get_device_id()) {
            permute_full(input0_hbm, output_hbm, input0_vmem, VMEMSize, oridim, perm, d0, idx, 1);
        } else if (cnt) { // if len > 0 then do
            permute_full(input0_hbm, output_hbm, input0_vmem, VMEMSize, oridim, perm, d0, idx, 0);
        }

        sync_device();
    }
}

inline void _permute_hbm_bf16(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSize, int *shape, int *perm) {
    int dim[5] = {shape[0], shape[1], shape[2], shape[3], shape[4]};
    int oridim[5] = {ALIGN128(shape[0]), shape[1], shape[2], shape[3], shape[4]};

    int dim0_128 = ALIGN256(shape[0]) / 2;
    int dim1 = dim[1] * dim[2] * dim[3] * dim[4];

    int d4 = shape[4];
    int d3 = shape[3];
    int d2 = shape[2];
    int d1 = shape[1];
    int d0 = shape[0];

    int perm_5d = (perm[0] ^ 0) + (perm[1] ^ 1) + (perm[2] ^ 2) + (perm[3] ^ 3) + (perm[4] ^ 4);
    if (perm_5d == 0) {
        Permute5D_HBM2HBM_2XYS(input0_hbm, output_hbm, input0_vmem, VMEMSize, dim0_128 * dim1);
    } else if (perm[0] == 0) {
        int row_stride[5] = {dim[1] * dim[2] * dim[3] * dim[4], dim[2] * dim[3] * dim[4], dim[3] * dim[4], dim[4], 1};

        int DMA_COUNT_MIN = row_stride[0];
        int ROW = 1; // row = 1, vertical = 0
        int ROW_DIM0_POS = 0;

        int H = 0;
        int W = 0;
        int ndim[5];
        int nsrc_stride[5];
        int ndst_stride[5];
        int src_stride = 1;
        int dst_stride = 1;
        int VERTICAL_FOR_COUNT;

        Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 1,  &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 1);

        if (perm[1] == 1) {
            if (row_stride[1] < DMA_COUNT_MIN) {
                DMA_COUNT_MIN = row_stride[1];
                ROW = 1;
                ROW_DIM0_POS = 1;
            }
            Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 2,  &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 1);

            if (perm[2] == 2) {
                if (row_stride[2] < DMA_COUNT_MIN) {
                    DMA_COUNT_MIN = row_stride[2];
                    ROW = 1;
                    ROW_DIM0_POS = 2;
                }
                Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 3, &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 1);
            }
        }

        if (ROW) {
            Permute4D_HBM_ROW_2XYS(input0_hbm, output_hbm, input0_vmem, VMEMSize, dim, perm, ROW_DIM0_POS, 1);
        } else {
            Permute4D_HBM_VERTICAL_2XYS(input0_hbm, output_hbm, input0_vmem, VMEMSize, ndim, 
                                nsrc_stride, ndst_stride, H, W, VERTICAL_FOR_COUNT, src_stride, dst_stride);
        }
    } else if (perm[0] == 1 && perm[1] == 0) {
        int idx[5] = {0, 0, 0, 0, 0};
        GetHalfIdx(1, 1, d2, d3, d4, idx);

        int cnt = d2 * d3 * d4 / 2;
        if (get_device_id()) {
            permute_easy_do_big_bf16(input0_hbm, output_hbm, input0_vmem, oridim, perm, d0, VMEMSize, idx, 1);
        } else if (cnt) { // if len > 0 then do
            permute_easy_do_big_bf16(input0_hbm, output_hbm, input0_vmem, oridim, perm, d0, VMEMSize, idx, 0);
        }

        sync_device();
    } else {
        if (perm[0] == 4) d4 = 1;
        else if (perm[0] == 3) d3 = 1;
        else if (perm[0] == 2) d2 = 1;
        else if (perm[0] == 1) d1 = 1;
        int idx[5] = {0, 0, 0, 0, 0};
        GetHalfIdx(1, d1, d2, d3, d4, idx);

        int cnt = d1 * d2 * d3 * d4 / 2;
        if (get_device_id()) {
            permute_full_bf16(input0_hbm, output_hbm, input0_vmem, VMEMSize, oridim, perm, d0, idx, 1);
        } else if (cnt) { // if len > 0 then do
            permute_full_bf16(input0_hbm, output_hbm, input0_vmem, VMEMSize, oridim, perm, d0, idx, 0);
        }

        sync_device();
    }
}

inline void _permute(SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem, unsigned *inf, int d0, int p4, int p3, int p2, int p1, int p0) {
    int d4 = inf[4];
    int d3 = inf[3];
    int d2 = inf[2];
    int d1 = inf[1];
    int pd0 = padding128(inf[0]);
    int oridim[5] = {pd0, d1, d2, d3, d4};
    int perm[5] = {p0, p1, p2, p3, p4};

    int dim[5] = {inf[0], d1, d2, d3, d4};
    int dim0_128 = ALIGN128(inf[0]);
    int dim1 = d1 * d2 * d3 * d4;

    int perm_5d = (perm[0] ^ 0) + (perm[1] ^ 1) + (perm[2] ^ 2) + (perm[3] ^ 3) + (perm[4] ^ 4);
    if (perm_5d == 0) {
        Permute4D_VMEM2VMEM(input0_vmem, output_vmem, dim0_128 * dim1, 128, 128);
    } else if (perm[0] == 0) {
        int row_stride[5] = {dim[1] * dim[2] * dim[3] * dim[4], dim[2] * dim[3] * dim[4], dim[3] * dim[4], dim[4], 1};

        int DMA_COUNT_MIN = row_stride[0];
        int ROW = 1; // row = 1, vertical = 0
        int ROW_DIM0_POS = 0;

        int H = 0;
        int W = 0;
        int ndim[5];
        int nsrc_stride[5];
        int ndst_stride[5];
        int src_stride = 1;
        int dst_stride = 1;
        int VERTICAL_FOR_COUNT;

        Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 1,  &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 0);

        if (perm[1] == 1) {
            if (row_stride[1] < DMA_COUNT_MIN) {
                DMA_COUNT_MIN = row_stride[1];
                ROW = 1;
                ROW_DIM0_POS = 1;
            }
            Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 2,  &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 0);

            if (perm[2] == 2) {
                if (row_stride[2] < DMA_COUNT_MIN) {
                    DMA_COUNT_MIN = row_stride[2];
                    ROW = 1;
                    ROW_DIM0_POS = 2;
                }
                Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 3, &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 0);
            }
        }

        if (ROW) {
            Permute4D_VMEM_ROW(input0_vmem, output_vmem, dim, perm, ROW_DIM0_POS, 0);
        } else {
            Permute4D_VMEM_VERTICAL(input0_vmem, output_vmem, ndim, nsrc_stride, ndst_stride,
                                    H, W, VERTICAL_FOR_COUNT, src_stride, dst_stride);
        }
    } else if (perm[0] == 1 && perm[1] == 0) {
        permute_easy_do(oridim, perm, input0_vmem, output_vmem, d0);
    } else {
        load_tran_trans(oridim, perm, input0_vmem, output_vmem, d0);
    }
}

inline void _permute_bf16(SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem, unsigned *inf, int d0, int p4, int p3, int p2, int p1, int p0) {
    int d4 = inf[4];
    int d3 = inf[3];
    int d2 = inf[2];
    int d1 = inf[1];
    int pd0 = inf[0];
    int oridim[5] = {pd0, d1, d2, d3, d4};
    int perm[5] = {p0, p1, p2, p3, p4};

    int dim[5] = {inf[0], d1, d2, d3, d4};
    int dim0_128 = ALIGN256(inf[0]) / 2;
    int dim1 = d1 * d2 * d3 * d4;

    int perm_5d = (perm[0] ^ 0) + (perm[1] ^ 1) + (perm[2] ^ 2) + (perm[3] ^ 3) + (perm[4] ^ 4);
    if (perm_5d == 0) {
        Permute4D_VMEM2VMEM(input0_vmem, output_vmem, dim0_128 * dim1, 128, 128);
    } else if (perm[0] == 0) {
        int row_stride[5] = {dim[1] * dim[2] * dim[3] * dim[4], dim[2] * dim[3] * dim[4], dim[3] * dim[4], dim[4], 1};

        int DMA_COUNT_MIN = row_stride[0];
        int ROW = 1; // row = 1, vertical = 0
        int ROW_DIM0_POS = 0;

        int H = 0;
        int W = 0;
        int ndim[5];
        int nsrc_stride[5];
        int ndst_stride[5];
        int src_stride = 1;
        int dst_stride = 1;
        int VERTICAL_FOR_COUNT;

        Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 1, &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 1);

        if (perm[1] == 1) {
            if (row_stride[1] < DMA_COUNT_MIN) {
                DMA_COUNT_MIN = row_stride[1];
                ROW = 1;
                ROW_DIM0_POS = 1;
            }
            Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 2, &H, &W, ndim, nsrc_stride, ndst_stride, &src_stride, &dst_stride, &VERTICAL_FOR_COUNT, 1);

            if (perm[2] == 2) {
                if (row_stride[2] < DMA_COUNT_MIN) {
                    DMA_COUNT_MIN = row_stride[2];
                    ROW = 1;
                    ROW_DIM0_POS = 2;
                }
                Permute4D_DivideDim(&DMA_COUNT_MIN, &ROW, dim, perm, 3, &H, &W, ndim, 
                                    nsrc_stride, ndst_stride, &src_stride, &dst_stride,
                                    &VERTICAL_FOR_COUNT, 1);
            }
        }

        if (ROW) {
            Permute4D_VMEM_ROW(input0_vmem, output_vmem, dim, perm, ROW_DIM0_POS, 1);
        } else {
            Permute4D_VMEM_VERTICAL(input0_vmem, output_vmem, ndim, nsrc_stride, ndst_stride,
                                    H, W, VERTICAL_FOR_COUNT, src_stride, dst_stride);
        }
    } else if (perm[0] == 1 && perm[1] == 0) {
        permute_easy_do_bf16(oridim, perm, input0_vmem, output_vmem, d0);
    } else {
        load_tran_trans_bf16(oridim, perm, input0_vmem, output_vmem, d0);
    }
}

inline void permute_keep_2d_bf16_to_f32(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst) {
    int d0 = oridim[0];
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};

    oridim[0] = padding256(oridim[0]) / 2;
    int oridim_offset[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    
    newdim[0] = padding128(newdim[0]);
    int newdim_offset[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};

    int idx[5] = {0, 0, 0, 0, 0};
    
    if (padding128(d0) % 256 == 0) {
        for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
            for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
                for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                    int offset_src = idx[4] * oridim_offset[4] + idx[3] * oridim_offset[3] + idx[2] * oridim_offset[2];
                    int offset_dst = idx[perm[4]] * newdim_offset[4] + idx[perm[3]] * newdim_offset[3] + idx[perm[2]] * newdim_offset[2];

                    int len = oridim[0] * oridim[1];

                    int ldst_msk = pre_exp2((len - len / 1024 * 1024) / 128);

                    int i = 0;
                    for (; i < len / 1024 * 1024; i += 1024) {
                        float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                        float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                        float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                        store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                        store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 255);
                    }
                    if (ldst_msk) {
                        float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, src + offset_src / 32, 1, ldst_msk);
                        float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                        float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                        store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, ldst_msk);
                        store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, ldst_msk);
                    }
                }
            }
        }
    } else {
        if (oridim[0] % 1024 == 0) {
            for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
                for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
                    for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                        for (idx[1] = 0; idx[1]  < oridim[1]; idx[1]++) {
                            int offset_src = idx[4] * oridim_offset[4] +
                                            idx[3] * oridim_offset[3] +
                                            idx[2] * oridim_offset[2] +
                                            idx[1] * oridim_offset[1];
                            int offset_dst = idx[perm[4]] * newdim_offset[4] +
                                            idx[perm[3]] * newdim_offset[3] +
                                            idx[perm[2]] * newdim_offset[2] +
                                            idx[perm[1]] * newdim_offset[1];

                            int len = oridim[0];

                            int i = 0;
                            for (; i < len - 1024; i += 1024) {
                                float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 255);
                            }
                            if (1) {
                                float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 127);
                            }
                        }
                    }
                }
            }
        } else {
            for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
                for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
                    for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                        for (idx[1] = 0; idx[1]  < oridim[1]; idx[1]++) {
                            int offset_src = idx[4] * oridim_offset[4] +
                                            idx[3] * oridim_offset[3] +
                                            idx[2] * oridim_offset[2] +
                                            idx[1] * oridim_offset[1];
                            int offset_dst = idx[perm[4]] * newdim_offset[4] +
                                            idx[perm[3]] * newdim_offset[3] +
                                            idx[perm[2]] * newdim_offset[2] +
                                            idx[perm[1]] * newdim_offset[1];

                            int len = oridim[0];
                            int ldst_msk = pre_exp2((len - len / 1024 * 1024) / 128);

                            int i = 0;
                            for (; i < len / 1024 * 1024; i += 1024) {
                                float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 255);
                            }
                            if (ldst_msk) {
                                float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, src + offset_src / 32, 1, ldst_msk);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, ldst_msk);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, ldst_msk / 2);
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void permute_keep_1d_bf16_to_f32(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst) {    
    int d0 = oridim[0];
    int newdim[5] = {oridim[perm[0]], oridim[perm[1]], oridim[perm[2]], oridim[perm[3]], oridim[perm[4]]};

    oridim[0] = padding256(oridim[0]) / 2;
    int oridim_offset[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    
    newdim[0] = padding128(newdim[0]);
    int newdim_offset[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};

    int idx[5] = {0, 0, 0, 0, 0};

    if (padding128(d0) % 256 == 0) {
        for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
            for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
                for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                    for (idx[1] = 0; idx[1] < oridim[1]; idx[1]++) {
                        int offset_src = idx[4] * oridim_offset[4] +
                                            idx[3] * oridim_offset[3] +
                                            idx[2] * oridim_offset[2] +
                                            idx[1] * oridim_offset[1];
                        int offset_dst = idx[perm[4]] * newdim_offset[4] +
                                            idx[perm[3]] * newdim_offset[3] +
                                            idx[perm[2]] * newdim_offset[2] +
                                            idx[perm[1]] * newdim_offset[1];

                        int len = oridim[0];

                        int ldst_msk = pre_exp2((len - len / 1024 * 1024) / 128);

                        int i = 0;
                        for (; i < len / 1024 * 1024; i += 1024) {
                            float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                            float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                            float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                            store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                            store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 255);
                        }
                        if (ldst_msk) {
                            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, src + offset_src / 32, 1, ldst_msk);
                            float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                            float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                            store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, ldst_msk);
                            store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, ldst_msk);
                        }
                    }
                }
            }
        }
    } else {
        if (oridim[0] % 1024 == 0) {
            for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
                for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
                    for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                        for (idx[1] = 0; idx[1]  < oridim[1]; idx[1]++) {
                            int offset_src = idx[4] * oridim_offset[4] +
                                            idx[3] * oridim_offset[3] +
                                            idx[2] * oridim_offset[2] +
                                            idx[1] * oridim_offset[1];
                            int offset_dst = idx[perm[4]] * newdim_offset[4] +
                                            idx[perm[3]] * newdim_offset[3] +
                                            idx[perm[2]] * newdim_offset[2] +
                                            idx[perm[1]] * newdim_offset[1];

                            int len = oridim[0];

                            int i = 0;
                            for (; i < len - 1024; i += 1024) {
                                float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 255);
                            }
                            if (1) {
                                float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 127);
                            }
                        }
                    }
                }
            }
        } else {
            for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
                for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
                    for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                        for (idx[1] = 0; idx[1]  < oridim[1]; idx[1]++) {
                            int offset_src = idx[4] * oridim_offset[4] +
                                            idx[3] * oridim_offset[3] +
                                            idx[2] * oridim_offset[2] +
                                            idx[1] * oridim_offset[1];
                            int offset_dst = idx[perm[4]] * newdim_offset[4] +
                                            idx[perm[3]] * newdim_offset[3] +
                                            idx[perm[2]] * newdim_offset[2] +
                                            idx[perm[1]] * newdim_offset[1];

                            int len = oridim[0];
                            int ldst_msk = pre_exp2((len - len / 1024 * 1024) / 128);

                            int i = 0;
                            for (; i < len / 1024 * 1024; i += 1024) {
                                float8_128 x = v_f32_ld_tnsr_b(i / 32, src + offset_src / 32);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, 255);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, 255);
                            }
                            if (ldst_msk) {
                                float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, src + offset_src / 32, 1, ldst_msk);
                                float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));
                                float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
                                store8_128_stride_stmk(i / 32 * 2, (2), dst + offset_dst / 32, x1, ldst_msk);
                                store8_128_stride_stmk(i / 32 * 2 + 128 / 32, (2), dst + offset_dst / 32, x2, ldst_msk / 2);
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void permute_2d_bf16_to_f32(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst, int rd0) {
    oridim[0] = padding256(oridim[0]) / 2;
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding128(newdim[0]);
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int idx[5] = {0, 0, 0, 0, 0};

    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);

    for (idx[4] = 0; idx[4] < oridim[4]; idx[4]++) {
        for (idx[3] = 0; idx[3] < oridim[3]; idx[3]++) {
            for (idx[2] = 0; idx[2] < oridim[2]; idx[2]++) {
                int offset_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2];
                int offset_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2];

                tile_trans_transfer_bf16_to_f32_with_stride(src, dst, offset_src, oridim[1], rd0, offset_dst, oridim[0], newdim[0]);
            }
        }
    }
}

inline void permute_5d_bf16_to_f32(int *oridim, int *perm, SIM_X86::tensor src, SIM_X86::tensor dst, int rd0) {
    oridim[0] = padding256(oridim[0]) / 2;
    int oldst[5] = {1, oridim[0], oridim[0] * oridim[1], oridim[0] * oridim[1] * oridim[2],
                  oridim[0] * oridim[1] * oridim[2] * oridim[3]};
    int rodim[5] = {rd0, oridim[1], oridim[2], oridim[3], oridim[4]};
    int newdim[5] = {rodim[perm[0]], rodim[perm[1]], rodim[perm[2]], rodim[perm[3]], rodim[perm[4]]};
    newdim[0] = padding128(newdim[0]);
    int newst[5] = {1, newdim[0], newdim[0] * newdim[1], newdim[0] * newdim[1] * newdim[2],
                  newdim[0] * newdim[1] * newdim[2] * newdim[3]};
    int oridim_new[5];
    for (int i = 0; i < 5; i++) {
        oridim_new[i] = oridim[i];
    }

    int input0_stride = 0;
    int output_stride = 1;

    if (perm[0] == 4) {
        oridim_new[4] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2] * oridim[3];
    } else if (perm[0] == 3) {
        oridim_new[3] = 1;
        input0_stride = oridim[0] * oridim[1] * oridim[2];
    } else if (perm[0] == 2) {
        oridim_new[2] = 1;
        input0_stride = oridim[0] * oridim[1];
    } else if (perm[0] == 1) {
        oridim_new[1] = 1;
        input0_stride = oridim[0];
    }

    for (int i = 0; i < 5; ++i) {
        if (perm[i] != 0) output_stride *= newdim[i];
        else break;
    }

    int hbm_h = rodim[perm[0]];
    int hbm_w = rd0;

    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);

    int idx[5] = {0, 0, 0, 0, 0};
    for (idx[4] = 0; idx[4] < oridim_new[4]; ++idx[4]) {
        for (idx[3] = 0; idx[3] < oridim_new[3]; ++idx[3]) {
            for (idx[2] = 0; idx[2] < oridim_new[2]; ++idx[2]) {
                for (idx[1] = 0; idx[1] < oridim_new[1]; ++idx[1]) {
                    int offset_src = idx[4] * oldst[4] + idx[3] * oldst[3] + idx[2] * oldst[2] + idx[1] * oldst[1];
                    int offset_dst = idx[perm[4]] * newst[4] + idx[perm[3]] * newst[3] + idx[perm[2]] * newst[2] + idx[perm[1]] * newst[1];

                    tile_trans_transfer_bf16_to_f32_with_stride(src, dst, offset_src, hbm_h, hbm_w, offset_dst, input0_stride, output_stride);
                }
            }
        }
    }
}

inline void _permute_bf16_to_f32(SIM_X86::tensor input0_vmem, SIM_X86::tensor output_vmem, unsigned *dim, int d0, int p4, int p3, int p2, int p1, int p0) {
    int perm[5] = {p0, p1, p2, p3, p4};

    int oridim[5] = {dim[0], dim[1], dim[2], dim[03], dim[4]};
    if (p0 == 0 && p1 == 1) {
        permute_keep_2d_bf16_to_f32(oridim, perm, input0_vmem, output_vmem);
    } else if (p0 == 0) {
        permute_keep_1d_bf16_to_f32(oridim, perm, input0_vmem, output_vmem);
    } else if (p0 == 1 && p1 == 0) {
        permute_2d_bf16_to_f32(oridim, perm, input0_vmem, output_vmem, d0);
    } else {
        permute_5d_bf16_to_f32(oridim, perm, input0_vmem, output_vmem, d0);
    }
}