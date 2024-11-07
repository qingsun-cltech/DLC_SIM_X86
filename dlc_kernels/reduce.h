#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"


#include "ldst.h"


#define MAKECHIPID(CHIP, XYS) (((CHIP) << 4) | ((XYS) + 2))

#define CORE_FXCHBM 1
#define CORE_XYS0 2
#define CORE_XYS1 3
#define MEM_H_X86BM 0
#define MEM_CMEM 2
#define MEM_VMEM 0
#define MEM_SMEM 1
#define MEM_IMEM 2
#define DMA_TYPE_LOCAL 0
#define DMA_TYPE_REMOTE_UNICAST 2
#define DMA_TYPE_REMOTE_MULTICAST 3

inline void sync_inc(int syncflag) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo imm_2 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = syncinc 4; }"
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

inline void sync_wait_done(int syncflag) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) Nah = wait.done 1, 0; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

inline void sync_wait_ge(int syncflag, int val) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val]; "
                 "  MISC@(pr0) Nah = wait.gte 1, 2; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 : [flag] "r"(syncflag), [val] "r"(val)
                 : "r0", "r1");
}

inline void sync_clear(int syncflag) {
    // set clear and 0
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = setsync.clear 0; }"
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

// rmtsync-target, bit [0] reserved, bit [1] boardcast, bit[2,13] chip id,
// bit[14,17], component id(xys 0010, xys 0011)
//
// chipid should be bit[0,11] is chip id, bit[12, 15] is component id
inline void rsync_inc(int chipid, int syncflag) {
    int target = (chipid << 14) | syncflag;
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo imm_2 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = rmtsyncinc 4; }"
                 :
                 : [flag] "r"(target)
                 : "r0");
}

inline void rsync_set_done(int chipid, int syncflag) {
    int target = (chipid << 14) | syncflag;
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = rmtsetsync.done 0; }"
                 :
                 : [flag] "r"(target)
                 : "r0");
}

inline void rsync_clear(int chipid, int syncflag) {
    int target = (chipid << 14) | syncflag;
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = rmtsetsync.clear 0; }"
                 :
                 : [flag] "r"(target)
                 : "r0");
}

inline int MakeDmaHeader(int trace_en, int dst_opcode, int dst_coreid, int dst_memid, int src_opcode,
                         int src_codeid, int src_memid, int dmatype, int dst_id) {
    return (dst_id >> 4 /*dst_id contain xysid, remove it*/) | (dmatype << 14) | (src_memid << 16) |
           (src_codeid << 18) | (src_opcode << 21) | (dst_memid << 24) | (dst_coreid << 26) |
           (dst_opcode << 29) | (trace_en << 31);
}

inline int MakeDstSyncFlag(int sync0_core, int sync0_flag, int sync1_core, int sync1_flag) {
    return sync0_flag | ((sync0_core + 2) << 13) | (sync1_flag << 16) | ((sync1_core + 2) << 29);
}

inline int MakeSrcSyncFlag(int sync_core, int sync_flag) { return sync_flag | ((sync_core + 2) << 13); }

inline void rdma_xys_v2v(SIM_X86::tensor sAddr, int sFlag, int dChip, SIM_X86::tensor dAddr, int dFlag, int len) {
    int smemCfg[8];
    smemCfg[0] = MakeDmaHeader(0, 0, CORE_XYS0, MEM_VMEM, 0, CORE_XYS1, MEM_VMEM, DMA_TYPE_LOCAL,
                               0); // header
    smemCfg[1] = sFlag;                // src sync flag
    smemCfg[2] = dFlag;                // dst sync flag
    smemCfg[3] = len / 128;            // length
    smemCfg[4] = (int)sAddr / 4;       // src_addr
    smemCfg[5] = (int)dAddr / 4;       // dst_addr
    smemCfg[6] = 1;                    // src_stride
    smemCfg[7] = 1;                    // dst_stride
    asm volatile("{ S1@(pr0) Nah = dma [smem:%[cfg]] }" ::[cfg] "r"(&smemCfg) :);
}




typedef float8_128 (*reduceBinaryOp_t)(float8_128, float8_128);
typedef float8_128 (*reduceUnaryOp_t)(float8_128);

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

/*
inline float128_128 loadh_k_old(SIM_X86::tensor t, int st, int h, int w, float fill) {
    float8_128 data[16];
    int top = 0;
    for (int i = 0; i < h; i += 8) {
        int cur_h = min(h - i, 8);
        int ldmk = (1 << cur_h) - 1;
        data[top++] = loadmin8_k(tensor_slice(t, i * st * 128 / 32), st, cur_h, ldmk, w, fill);
    }
    top = (h + 7) / 8;
    for (; top < 16; top++) {
        data[top] = fill;
    }
    return v_concat_16(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                       data[9], data[10], data[11], data[12], data[13], data[14], data[15]);
}
*/

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

inline void store128_128_ex(SIM_X86::tensor t, int h, int w, int ih, int iw, float128_128 v) {
    int pw = (w + 127) & 0xffffff80;
//     int cur_w = min(w - iw * 128, 128);
    int cur_h = min(h - ih * 128, 128);
    int kS = (cur_h + 7) / 8;
#define CASE_ITEM(x)                                                                                         \
    {                                                                                                        \
        int i = (x);                                                                                         \
        int cur_sth = min(cur_h - i * 8, 8);                                                                 \
        store8_128_stride_stmk((ih + i * 8) * pw / 32, pw / 128, t, sub_vector(v, x), (1 << cur_sth) - 1);   \
    }
    SWITCH_CASES_REV(kS)
#undef CASE_ITEM
}

/*
inline void store128_128_ex_old(SIM_X86::tensor t, int h, int w, int ih, int iw, float128_128 v) {
    int pw = (w + 127) & 0xffffff80;
    int cur_w = min(w - iw * 128, 128);
    int cur_h = min(h - ih * 128, 128);
    int kS = (cur_h + 7) / 8;
    float8_128 d[16];
    d[0] = sub_vector(v, 0);
    d[1] = sub_vector(v, 1);
    d[2] = sub_vector(v, 2);
    d[3] = sub_vector(v, 3);
    d[4] = sub_vector(v, 4);
    d[5] = sub_vector(v, 5);
    d[6] = sub_vector(v, 6);
    d[7] = sub_vector(v, 7);
    d[8] = sub_vector(v, 8);
    d[9] = sub_vector(v, 9);
    d[10] = sub_vector(v, 10);
    d[11] = sub_vector(v, 11);
    d[12] = sub_vector(v, 12);
    d[13] = sub_vector(v, 13);
    d[14] = sub_vector(v, 14);
    d[15] = sub_vector(v, 15);
    for (int i = 0; i < kS; i++) {
        int cur_sth = min(cur_h - i * 8, 8);
        store8_128_stride_stmk((ih + i * 8) * pw / 32, pw / 128, t, d[i], (1 << cur_sth) - 1);
    }
}
*/

inline float128_128 reduceEach128_128(reduceBinaryOp_t e_fn, float128_128 s, float128_128 a) {
    float8_128 r0 = e_fn(sub_vector(s, 0), sub_vector(a, 0));
    float8_128 r1 = e_fn(sub_vector(s, 1), sub_vector(a, 1));
    float8_128 r2 = e_fn(sub_vector(s, 2), sub_vector(a, 2));
    float8_128 r3 = e_fn(sub_vector(s, 3), sub_vector(a, 3));
    float8_128 r4 = e_fn(sub_vector(s, 4), sub_vector(a, 4));
    float8_128 r5 = e_fn(sub_vector(s, 5), sub_vector(a, 5));
    float8_128 r6 = e_fn(sub_vector(s, 6), sub_vector(a, 6));
    float8_128 r7 = e_fn(sub_vector(s, 7), sub_vector(a, 7));
    float8_128 r8 = e_fn(sub_vector(s, 8), sub_vector(a, 8));
    float8_128 r9 = e_fn(sub_vector(s, 9), sub_vector(a, 9));
    float8_128 r10 = e_fn(sub_vector(s, 10), sub_vector(a, 10));
    float8_128 r11 = e_fn(sub_vector(s, 11), sub_vector(a, 11));
    float8_128 r12 = e_fn(sub_vector(s, 12), sub_vector(a, 12));
    float8_128 r13 = e_fn(sub_vector(s, 13), sub_vector(a, 13));
    float8_128 r14 = e_fn(sub_vector(s, 14), sub_vector(a, 14));
    float8_128 r15 = e_fn(sub_vector(s, 15), sub_vector(a, 15));
    return v_concat_16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
}

inline float128_128 reduceAll128_128(reduceUnaryOp_t r_fn, float128_128 s) {
    float8_128 r0 = r_fn(sub_vector(s, 0));
    float8_128 r1 = r_fn(sub_vector(s, 1));
    float8_128 r2 = r_fn(sub_vector(s, 2));
    float8_128 r3 = r_fn(sub_vector(s, 3));
    float8_128 r4 = r_fn(sub_vector(s, 4));
    float8_128 r5 = r_fn(sub_vector(s, 5));
    float8_128 r6 = r_fn(sub_vector(s, 6));
    float8_128 r7 = r_fn(sub_vector(s, 7));
    float8_128 r8 = r_fn(sub_vector(s, 8));
    float8_128 r9 = r_fn(sub_vector(s, 9));
    float8_128 r10 = r_fn(sub_vector(s, 10));
    float8_128 r11 = r_fn(sub_vector(s, 11));
    float8_128 r12 = r_fn(sub_vector(s, 12));
    float8_128 r13 = r_fn(sub_vector(s, 13));
    float8_128 r14 = r_fn(sub_vector(s, 14));
    float8_128 r15 = r_fn(sub_vector(s, 15));
    return v_concat_16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
}

// [L, pR] => [L, 128]
inline void reduce_low(SIM_X86::tensor in, SIM_X86::tensor out, int h, int padding_w, int w, float reduceOpIdent,
                       reduceUnaryOp_t map, reduceBinaryOp_t combine, reduceUnaryOp_t reduce_combine,
                       reduceUnaryOp_t project, bool skipMap, bool skipProject) {
    for (int i = 0; i < h; i += 8) {
        int cur_h = min(h - i, 8);
        int ldstmk = (1 << cur_h) - 1;
        float8_128 vmax = v_u32_move_f(reduceOpIdent);
        for (int j = 0; j < w; j += 128) {
            int cur_w = min(w - j, 128);
            float8_128 v = load8_k(tensor_slice(in, (i * padding_w + j) / 32), padding_w / 128, ldstmk, cur_w,
                                   reduceOpIdent);
            if (skipMap) {
                vmax = combine(vmax, v);
            } else {
                vmax = combine(vmax, map(v));
            }
        }
        float8_128 rmax = reduce_combine(vmax);
        if (!skipProject) {
            rmax = project(rmax);
        }
        v_f32_st_tnsr_st_msk(i * 128 / 32, out, 1, ldstmk, rmax);
    }
}

// [K, pR] => [1, 128]
inline void reduce_all(SIM_X86::tensor in, SIM_X86::tensor out, int K, int R, float reduceOpIdent, reduceUnaryOp_t map,
                       reduceBinaryOp_t combine, reduceUnaryOp_t reduce_combine, reduceUnaryOp_t project) {
    int padding_R = (R + 127) & 0xffffff80;
//     int padding_K = (K + 127) & 0xffffff80;
//     int8_128 c = v_u32_and(get_core_id(), v_u32_move_i(127));
    float128_128 mx = expand(v_u32_move_f(reduceOpIdent));
    for (int k = 0; k < K; k += 128) {
        for (int r = 0; r < R; r += 128) {
            int cur_h = min(K - k, 128);
            int cur_w = min(R - r, 128);
            float128_128 v = loadh_k(tensor_slice(in, (k * padding_R + r) / 32), padding_R / 128, cur_h,
                                     cur_w, reduceOpIdent);
            mx = reduceEach128_128(combine, reduceAll128_128(map, v), mx);
        }
    }
    mx = reduceAll128_128(reduce_combine, mx);
    float128_128 pR = m_transpose_128_128_nws(mx, 0);
    float8_128 dR = sub_vector(pR, 0);
    dR = project(reduce_combine(dR));
    v_f32_st_tnsr_st_msk(0, out, 1, 1, dR);
}

// [K, pR] => [1, 128]
inline float8_128 reduce_all_v(SIM_X86::tensor in, int K, int R, float reduceOpIdent, reduceUnaryOp_t map,
                               reduceBinaryOp_t combine, reduceUnaryOp_t reduce_combine,
                               reduceUnaryOp_t project) {
    int padding_R = (R + 127) & 0xffffff80;
//     int padding_K = (K + 127) & 0xffffff80;
//     int8_128 c = v_u32_and(get_core_id(), v_u32_move_i(127));
    float128_128 mx = expand(v_u32_move_f(reduceOpIdent));
    for (int k = 0; k < K; k += 128) {
        for (int r = 0; r < R; r += 128) {
            int cur_h = min(K - k, 128);
            int cur_w = min(R - r, 128);
            float128_128 v = loadh_k(tensor_slice(in, (k * padding_R + r) / 32), padding_R / 128, cur_h,
                                     cur_w, reduceOpIdent);
            mx = reduceEach128_128(combine, reduceAll128_128(map, v), mx);
        }
    }
    mx = reduceAll128_128(reduce_combine, mx);
    float128_128 pR = m_transpose_128_128_nws(mx, 0);
    float8_128 dR = sub_vector(pR, 0);
    dR = reduce_combine(dR);
    return dR;
}

// [L, k, P, pR] => [L, P, pR]
inline void reduce_mid(SIM_X86::tensor in, SIM_X86::tensor out, int L, int K, int P, int R, float reduceOpIdent,
                       reduceUnaryOp_t map, reduceBinaryOp_t combine, reduceUnaryOp_t reduce_combine,
                       reduceUnaryOp_t project, bool skipMap, bool skipProject) {
    int padding_R = (R + 127) & 0xffffff80;
    int padding_PR = P * padding_R;
    int padding_K = (K + 127) & 0xffffff80;
    int8_128 c = v_u32_and(get_core_id(), v_u32_move_i(127));
    for (int l = 0; l < L; l++) {
        for (int p = 0; p < P; p++) {
            for (int r = 0; r < R; r += 128) {
                float128_128 mx = expand(v_u32_move_f(reduceOpIdent));
                for (int k = 0; k < K; k += 128) {
                    float128_128 v =
                        loadh_k_T(tensor_slice(in, K * padding_PR * l / 32), padding_K, K, padding_PR,
                                  (P - 1) * padding_R + R, reduceOpIdent, k / 128, (p * padding_R + r) / 128);
                    if (skipMap) {
                        mx = reduceEach128_128(combine, v, mx);
                    } else {
                        mx = reduceEach128_128(combine, reduceAll128_128(map, v), mx);
                    }
                }
                if (skipProject) {
                    mx = reduceAll128_128(reduce_combine, mx);
                } else {
                    mx = reduceAll128_128(project, reduceAll128_128(reduce_combine, mx));
                }
                float128_128 pR = m_transpose_128_128_nws(mx, 0);
                int cur_r = min(R - r, 128);
                bool8_128 m = v_s32_cmp(LS, c, v_u32_move_i(cur_r));
                // not use v_st_vmsk for pervert ecc fail
                float8_128 res = v_f32_sel(m, v_u32_move_f(0.0f), sub_vector(pR, 0));
                v_f32_st_tnsr_st_msk((l * padding_PR + p * padding_R + r) / 32, out, 1, 1, res);
            }
        }
    }
}

// (T, [E], E1024 -> T1024, (T1024, T1024) -> T1024, T1024 -> T,     T1024 -> R1024) -> R
//          map,            combine,                 reduce_combine, project
inline void reduce(SIM_X86::tensor in, SIM_X86::tensor out, int d4, int d3, int d2, int d1, int d0, int reduceDim,
                   float reduceOpIdent, reduceUnaryOp_t map, reduceBinaryOp_t combine,
                   reduceUnaryOp_t reduce_combine, reduceUnaryOp_t project) {
    int pd0 = (d0 + 127) & 0xffffff80;
    if (reduceDim == 4) {
        reduce_low(in, out, d4 * d3 * d2 * d1, pd0, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
    } else if (reduceDim == 3) {
        reduce_mid(in, out, d4 * d3 * d2, d1, 1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
    } else if (reduceDim == 2) {
        reduce_mid(in, out, d4 * d3, d2, d1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
    } else if (reduceDim == 1) {
        reduce_mid(in, out, d4, d3, d2 * d1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
    } else if (reduceDim == 0) {
        reduce_mid(in, out, 1, d4, d3 * d2 * d1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
    } else if (reduceDim == -1) {
        reduce_all(in, out, d4 * d3 * d2 * d1, d0, reduceOpIdent, map, combine, reduce_combine, project);
    }
}

inline void reduce_hbm(SIM_X86::tensor inhbm, SIM_X86::tensor outhbm, int d4, int d3, int d2, int d1, int d0, int reduceDim,
                       float reduceOpIdent, reduceUnaryOp_t map, reduceBinaryOp_t combine,
                       reduceUnaryOp_t reduce_combine, reduceUnaryOp_t project, SIM_X86::tensor vmem, int vmemlen) {
    int pd0 = (d0 + 127) & 0xffffff80;
    int totlen = d4 * d3 * d2 * d1 * pd0;
    if (reduceDim == 4) {
        int min_vin = pd0;
        int min_vout = 128;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3 * d2 * d1;
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_low(vin, vout, curgroup, pd0, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
//             __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 3) {
        int min_vin = d1 * pd0;
        int min_vout = pd0;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3 * d2;
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_mid(vin, vout, curgroup, d1, 1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 2) {
        int min_vin = d2 * d1 * pd0;
        int min_vout = d1 * pd0;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3;
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_mid(vin, vout, curgroup, d2, d1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 1) {
        int min_vin = d3 * d2 * d1 * pd0;
        int min_vout = d2 * d1 * pd0;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4;
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_mid(vin, vout, curgroup, d3, d2 * d1, d0, reduceOpIdent, map, combine, reduce_combine,
                       project, 0, 0);
// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 0) {
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + totlen / 32;
        int sync0 = dlc_dma(inhbm, HBM, vin, VMEM, totlen, 128, 128, 128, 7);
        dlc_sync(sync0);
        reduce_mid(vin, vout, 1, d4, d3 * d2 * d1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
//          __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
        int sync1 = dlc_dma(vout, VMEM, outhbm, HBM, d3 * d2 * d1 * pd0, 128, 128, 128, 7);
        dlc_sync(sync1);
    } else if (reduceDim == -1) {
        int min_vin = pd0;
        int min_v = min_vin;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3 * d2 * d1;
//         int vmeminlen = group * min_vin;
        SIM_X86::tensor vin = vmem;
        float8_128 res = v_u32_move_f(reduceOpIdent);
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            float8_128 v =
                reduce_all_v(vin, curgroup, d0, reduceOpIdent, map, combine, reduce_combine, project);
            res = combine(res, v);
        }
        res = project(res);
        v_f32_st_tnsr_st_msk(0, vmem, 1, 1, res);
//          __attribute__((unused))volatile float wait = vstore_wait(res);
        int sync1 = dlc_dma(vmem, VMEM, outhbm, HBM, 128, 128, 128, 128, 7);
        dlc_sync(sync1);
    }
}

// Attention: d4 is high, d0 is low,
inline void reduce_hbm_dimlist(SIM_X86::tensor inhbm, SIM_X86::tensor outhbm, int d4, int d3, int d2, int d1, int d0,
                               bool reduce_d4, bool reduce_d3, bool reduce_d2, bool reduce_d1, bool reduce_d0,
                               float reduceOpIdent, reduceUnaryOp_t map, reduceBinaryOp_t combine,
                               reduceUnaryOp_t reduce_combine, reduceUnaryOp_t project, SIM_X86::tensor vmem,
                               int vmemlen) {
    int pd0 = (d0 + 127) & 0xffffff80;
//     int totlen = d4 * d3 * d2 * d1 * pd0;
    if (reduce_d4 && reduce_d3 && reduce_d2 && reduce_d1 && reduce_d0) {
        int min_vin = pd0;
        int min_v = min_vin;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3 * d2 * d1;
//         int vmeminlen = group * min_vin;
        SIM_X86::tensor vin = vmem;
        float8_128 res = v_u32_move_f(reduceOpIdent);
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            float8_128 v =
                reduce_all_v(vin, curgroup, d0, reduceOpIdent, map, combine, reduce_combine, project);
            res = combine(res, v);
        }
        res = project(res);
        v_f32_st_tnsr_st_msk(0, vmem, 1, 1, res);
        int sync1 = dlc_dma(vmem, VMEM, outhbm, HBM, 128, 128, 128, 128, 7);
        dlc_sync(sync1);
        return;
    }

    if (reduce_d4 && reduce_d3) {
        d3 *= d4;
        d4 = 1;
        reduce_d4 = 0;
    }
    if (reduce_d3 && reduce_d2) {
        d2 *= d3;
        d3 = 1;
        reduce_d3 = 0;
    }
    if (reduce_d2 && reduce_d1) {
        d1 *= d2;
        d2 = 1;
        reduce_d2 = 0;
    }

    int topKeepDim = 5;
    bool reduce_ds[5] = {reduce_d0, reduce_d1, reduce_d2, reduce_d3, reduce_d4};
    for (int i = 4; i >= 0; i--) {
        if (reduce_ds[i]) {
            break;
        }
        topKeepDim = i;
    }

    int insize[6] = {1, pd0, pd0 * d1, pd0 * d1 * d2, pd0 * d1 * d2 * d3, pd0 * d1 * d2 * d3 * d4};
    int groupsize[6] = {pd0 * d1 * d2 * d3 * d4, d1 * d2 * d3 * d4, d2 * d3 * d4, d3 * d4, d4, 1};
    int outdim[5] = {reduce_d0 ? 128 : pd0, reduce_d1 ? 1 : d1, reduce_d2 ? 1 : d2, reduce_d3 ? 1 : d3,
                     reduce_d4 ? 1 : d4};
    int firstoutdim[5] = {topKeepDim == 1 ? 128 : pd0, topKeepDim == 2 ? 1 : d1, topKeepDim == 3 ? 1 : d2,
                          topKeepDim == 4 ? 1 : d3, topKeepDim == 5 ? 1 : d4};
    int firstoutsize[6] = {128,
                           firstoutdim[0],
                           firstoutdim[0] * firstoutdim[1],
                           firstoutdim[0] * firstoutdim[1] * firstoutdim[2],
                           firstoutdim[0] * firstoutdim[1] * firstoutdim[2] * firstoutdim[3],
                           firstoutdim[0] * firstoutdim[1] * firstoutdim[2] * firstoutdim[3] *
                               firstoutdim[4]};
    int outsize[6] = {128,
                      outdim[0],
                      outdim[0] * outdim[1],
                      outdim[0] * outdim[1] * outdim[2],
                      outdim[0] * outdim[1] * outdim[2] * outdim[3],
                      outdim[0] * outdim[1] * outdim[2] * outdim[3] * outdim[4]};
    int min_vin = insize[topKeepDim];
    int max_vout = firstoutsize[topKeepDim];
    int min_vout = outsize[topKeepDim];
    int totgroup = groupsize[topKeepDim];
    if (topKeepDim == 0) {
        int sync = dlc_dma(inhbm, HBM, outhbm, HBM, d4 * d3 * d2 * d1 * pd0, 128, 128, 128, 7);
        dlc_sync(sync);
    } else if (topKeepDim == 1) {
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_low(vin, vout, curgroup, pd0, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else {
        int min_v = min_vin + max_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * max_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        int reduceCnt = reduce_d4 + reduce_d3 + reduce_d2 + reduce_d1 + reduce_d0;
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            SIM_X86::tensor curvin = vin;
            SIM_X86::tensor curvout = vout;
            int cd4 = topKeepDim >= 5 ? d4 : 1;
            int cd3 = topKeepDim >= 4 ? d3 : 1;
            int cd2 = topKeepDim >= 3 ? d2 : 1;
            int cd1 = topKeepDim >= 2 ? d1 : 1;
            int reduceInc = 0;
            if (reduce_d4) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup, d4, d3 * d2 * d1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd4 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d3) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup * cd4, d3, d2 * d1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd3 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d2) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup * cd4 * cd3, d2, d1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd2 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d1) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup * cd4 * cd3 * cd2, d1, 1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd1 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d0) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_low(tvin, tvout, curgroup * cd4 * cd3 * cd2 * cd1, pd0, d0, reduceOpIdent, map,
                           combine, reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            int sync1 = dlc_dma(curvin, VMEM, outhbm + (k * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    }
}


// Attention: d4 is high, d0 is low,
inline void reduce_hbm_dimlist_twoxys(SIM_X86::tensor inhbm, SIM_X86::tensor outhbm, int d4, int d3, int d2, int d1, int d0,
                               bool reduce_d4, bool reduce_d3, bool reduce_d2, bool reduce_d1, bool reduce_d0,
                               float reduceOpIdent, reduceUnaryOp_t map, reduceBinaryOp_t combine,
                               reduceUnaryOp_t reduce_combine, reduceUnaryOp_t project, SIM_X86::tensor vmem,
                               int vmemlen ,int device_id) {
    int pd0 = (d0 + 127) & 0xffffff80;
//     int totlen = d4 * d3 * d2 * d1 * pd0;
    if (reduce_d4 && reduce_d3 && reduce_d2 && reduce_d1 && reduce_d0) {
        if(device_id == 0)
        {
            int min_vin = pd0;
            int min_v = min_vin;
            int group = soft_sdiv(vmemlen, min_v);
            if (group <= 0) {
                return;
            }
            int totgroup = d4 * d3 * d2 * d1;
//             int vmeminlen = group * min_vin;
            SIM_X86::tensor vin = vmem;
            float8_128 res = v_u32_move_f(reduceOpIdent);
            for (int k = 0; k < totgroup; k += group) {
                int curgroup = min(totgroup - k, group);
                int curin = curgroup * min_vin;
                int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
                dlc_sync(sync0);
                float8_128 v =
                    reduce_all_v(vin, curgroup, d0, reduceOpIdent, map, combine, reduce_combine, project);
                res = combine(res, v);
            }
            res = project(res);
            v_f32_st_tnsr_st_msk(0, vmem, 1, 1, res);

// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem, 1, 1));

            int sync1 = dlc_dma(vmem, VMEM, outhbm, HBM, 128, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
        return;
    }

    if (reduce_d4 && reduce_d3) {
        d3 *= d4;
        d4 = 1;
        reduce_d4 = 0;
    }
    if (reduce_d3 && reduce_d2) {
        d2 *= d3;
        d3 = 1;
        reduce_d3 = 0;
    }
    if (reduce_d2 && reduce_d1) {
        d1 *= d2;
        d2 = 1;
        reduce_d2 = 0;
    }

    int topKeepDim = 5;
    bool reduce_ds[5] = {reduce_d0, reduce_d1, reduce_d2, reduce_d3, reduce_d4};
    for (int i = 4; i >= 0; i--) {
        if (reduce_ds[i]) {
            break;
        }
        topKeepDim = i;
    }

    int insize[6] = {1, pd0, pd0 * d1, pd0 * d1 * d2, pd0 * d1 * d2 * d3, pd0 * d1 * d2 * d3 * d4};
    int groupsize[6] = {pd0 * d1 * d2 * d3 * d4, d1 * d2 * d3 * d4, d2 * d3 * d4, d3 * d4, d4, 1};
    int outdim[5] = {reduce_d0 ? 128 : pd0, reduce_d1 ? 1 : d1, reduce_d2 ? 1 : d2, reduce_d3 ? 1 : d3,
                     reduce_d4 ? 1 : d4};
    int firstoutdim[5] = {topKeepDim == 1 ? 128 : pd0, topKeepDim == 2 ? 1 : d1, topKeepDim == 3 ? 1 : d2,
                          topKeepDim == 4 ? 1 : d3, topKeepDim == 5 ? 1 : d4};
    int firstoutsize[6] = {128,
                           firstoutdim[0],
                           firstoutdim[0] * firstoutdim[1],
                           firstoutdim[0] * firstoutdim[1] * firstoutdim[2],
                           firstoutdim[0] * firstoutdim[1] * firstoutdim[2] * firstoutdim[3],
                           firstoutdim[0] * firstoutdim[1] * firstoutdim[2] * firstoutdim[3] *
                               firstoutdim[4]};
    int outsize[6] = {128,
                      outdim[0],
                      outdim[0] * outdim[1],
                      outdim[0] * outdim[1] * outdim[2],
                      outdim[0] * outdim[1] * outdim[2] * outdim[3],
                      outdim[0] * outdim[1] * outdim[2] * outdim[3] * outdim[4]};
    int min_vin = insize[topKeepDim];
    int max_vout = firstoutsize[topKeepDim];
    int min_vout = outsize[topKeepDim];
    int totgroup = groupsize[topKeepDim];
    int totgroup_xys = totgroup / 2;
    int totgroup_xys1 = totgroup_xys;
    if(device_id == 1){
        totgroup_xys1 = totgroup - totgroup_xys;
    }
    if (topKeepDim == 0) {
        int sync = dlc_dma(inhbm, HBM, outhbm, HBM, d4 * d3 * d2 * d1 * pd0, 128, 128, 128, 7);
        dlc_sync(sync);
    } else if (topKeepDim == 1) {
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup_xys1; k += group) {
            int curgroup = min(totgroup_xys1 - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin + device_id * totgroup_xys * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_low(vin, vout, curgroup, pd0, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);

// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));

            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout + device_id * totgroup_xys * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else {
        int min_v = min_vin + max_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * max_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        int reduceCnt = reduce_d4 + reduce_d3 + reduce_d2 + reduce_d1 + reduce_d0;
        for (int k = 0; k < totgroup_xys1; k += group) {
            int curgroup = min(totgroup_xys1 - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin + device_id * totgroup_xys * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            SIM_X86::tensor curvin = vin;
            SIM_X86::tensor curvout = vout;
            int cd4 = topKeepDim >= 5 ? d4 : 1;
            int cd3 = topKeepDim >= 4 ? d3 : 1;
            int cd2 = topKeepDim >= 3 ? d2 : 1;
            int cd1 = topKeepDim >= 2 ? d1 : 1;
            int reduceInc = 0;
            if (reduce_d4) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup, d4, d3 * d2 * d1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd4 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d3) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup * cd4, d3, d2 * d1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd3 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d2) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup * cd4 * cd3, d2, d1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd2 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d1) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_mid(tvin, tvout, curgroup * cd4 * cd3 * cd2, d1, 1, d0, reduceOpIdent, map, combine,
                           reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                cd1 = 1;
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }
            if (reduce_d0) {
                SIM_X86::tensor tvin = curvin;
                SIM_X86::tensor tvout = curvout;
                reduce_low(tvin, tvout, curgroup * cd4 * cd3 * cd2 * cd1, pd0, d0, reduceOpIdent, map,
                           combine, reduce_combine, project, reduceInc != 0, reduceInc != (reduceCnt - 1));
                curvin = tvout;
                curvout = tvin;
                reduceInc++;
            }

// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, curvin, 1, 1));

            int sync1 = dlc_dma(curvin, VMEM, outhbm + (k * min_vout + device_id * totgroup_xys * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    }
}





inline void reduce_hbm_twoxys(SIM_X86::tensor inhbm, SIM_X86::tensor outhbm, int d4, int d3, int d2, int d1, int d0, int reduceDim,
                       float reduceOpIdent, reduceUnaryOp_t map, reduceBinaryOp_t combine,
                       reduceUnaryOp_t reduce_combine, reduceUnaryOp_t project, SIM_X86::tensor vmem, int vmemlen ,int device_id) {
    int pd0 = (d0 + 127) & 0xffffff80;
    int totlen = d4 * d3 * d2 * d1 * pd0;
    if (reduceDim == 4) {
        int min_vin = pd0;
        int min_vout = 128;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3 * d2 * d1;
        int totgroup_xys = totgroup / 2;
        int totgroup_xys1 = totgroup_xys;
        if(device_id == 1){
            totgroup_xys1 = totgroup - totgroup_xys;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup_xys1; k += group) {
            int curgroup = min(totgroup_xys1 - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin + device_id * totgroup_xys * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_low(vin, vout, curgroup, pd0, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout + device_id * totgroup_xys * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 3) {
        int min_vin = d1 * pd0;
        int min_vout = pd0;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3 * d2;
        int totgroup_xys = totgroup / 2;
        int totgroup_xys1 = totgroup_xys;
        if(device_id == 1){
            totgroup_xys1 = totgroup - totgroup_xys;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup_xys1; k += group) {
            int curgroup = min(totgroup_xys1 - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin + device_id * totgroup_xys * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_mid(vin, vout, curgroup, d1, 1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout + device_id * totgroup_xys * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 2) {
        int min_vin = d2 * d1 * pd0;
        int min_vout = d1 * pd0;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3;
        int totgroup_xys = totgroup / 2;
        int totgroup_xys1 = totgroup_xys;
        if(device_id == 1){
            totgroup_xys1 = totgroup - totgroup_xys;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup_xys1; k += group) {
            int curgroup = min(totgroup_xys1 - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin + device_id * totgroup_xys * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_mid(vin, vout, curgroup, d2, d1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout +  device_id * totgroup_xys * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 1) {
        int min_vin = d3 * d2 * d1 * pd0;
        int min_vout = d2 * d1 * pd0;
        int min_v = min_vin + min_vout;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4;
        int totgroup_xys = totgroup / 2;
        int totgroup_xys1 = totgroup_xys;
        if(device_id == 1){
            totgroup_xys1 = totgroup - totgroup_xys;
        }
        int vmeminlen = group * min_vin;
//         int vmemoutlen = group * min_vout;
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + vmeminlen / 32;
        for (int k = 0; k < totgroup_xys1; k += group) {
            int curgroup = min(totgroup_xys1 - k, group);
            int curin = curgroup * min_vin;
            int curout = curgroup * min_vout;
            int sync0 = dlc_dma(inhbm + (k * min_vin + device_id * totgroup_xys * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            reduce_mid(vin, vout, curgroup, d3, d2 * d1, d0, reduceOpIdent, map, combine, reduce_combine,
                       project, 0, 0);
// __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
            int sync1 = dlc_dma(vout, VMEM, outhbm + (k * min_vout + device_id * totgroup_xys * min_vout) / 32, HBM, curout, 128, 128, 128, 7);
            dlc_sync(sync1);
        }
    } else if (reduceDim == 0) {
        SIM_X86::tensor vin = vmem;
        SIM_X86::tensor vout = vmem + totlen / 32;
        int sync0 = dlc_dma(inhbm , HBM, vin, VMEM, totlen, 128, 128, 128, 7);
        dlc_sync(sync0);
        reduce_mid(vin, vout, 1, d4, d3 * d2 * d1, d0, reduceOpIdent, map, combine, reduce_combine, project, 0, 0);
//          __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
        int sync1 = dlc_dma(vout, VMEM, outhbm, HBM, d3 * d2 * d1 * pd0, 128, 128, 128, 7);
        dlc_sync(sync1);
    } else if (reduceDim == -1) {
        int min_vin = pd0;
        int min_v = min_vin;
        int group = soft_sdiv(vmemlen, min_v);
        if (group <= 0) {
            return;
        }
        int totgroup = d4 * d3 * d2 * d1;
//         int vmeminlen = group * min_vin;
        SIM_X86::tensor vin = vmem;
        float8_128 res = v_u32_move_f(reduceOpIdent);
        for (int k = 0; k < totgroup; k += group) {
            int curgroup = min(totgroup - k, group);
            int curin = curgroup * min_vin;
            int sync0 = dlc_dma(inhbm + (k * min_vin) / 32, HBM, vin, VMEM, curin, 128, 128, 128, 7);
            dlc_sync(sync0);
            float8_128 v =
                reduce_all_v(vin, curgroup, d0, reduceOpIdent, map, combine, reduce_combine, project);
            res = combine(res, v);
        }
        res = project(res);
        v_f32_st_tnsr_st_msk(0, vmem, 1, 1, res);
//          __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem, 1, 1));
        int sync1 = dlc_dma(vmem, VMEM, outhbm, HBM, 128, 128, 128, 128, 7);
        dlc_sync(sync1);
    }
}
