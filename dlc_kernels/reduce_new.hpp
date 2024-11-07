#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"


#include "cxxbase.hpp"
#include "ldst.h"


inline int8_128 broadcast(int v) { return v_u32_move_i(v); }
inline float8_128 broadcast(float v) { return v_u32_move_f(v); }

typedef float8_128 (*reduceBinaryOp_t)(float8_128, float8_128);
typedef float8_128 (*reduceUnaryOp_t)(float8_128);

template <typename T>
concept ReduceHasMap = requires {
    typename T::AccType;
    { T::map(float8_128{}) } -> Types::same_as<typename T::AccType>;
};

template <typename T>
concept ReduceHasProject = requires {
    typename T::AccType;
    { T::project(typename T::AccType{}) } -> Types::same_as<float8_128>;
};

template <typename T>
concept ReduceHasIdent = requires {
    typename T::AccType;
    T::ident;
};

template <typename T>
concept ReduceHasCombine = requires {
    typename T::AccType;
    { T::combine(typename T::AccType{}, typename T::AccType{}) } -> Types::same_as<typename T::AccType>;
};

template <typename T>
concept ReduceHasSingleRed = requires {
    typename T::AccType;
    { T::reduce_combine(typename T::AccType{}) } -> Types::same_as<typename T::AccType>;
};

template <typename T>
concept ReduceHasFifoRed = requires {
    T::template reduce_combine_push<true>(float8_128{});
    T::template reduce_combine_push<false>(float8_128{});
    { T::template reduce_combine_pop<true>() } -> Types::same_as<float8_128>;
    { T::template reduce_combine_pop<false>() } -> Types::same_as<float8_128>;
};

template <typename T>
concept ReduceCalc = ReduceHasIdent<T> && ReduceHasCombine<T> && ReduceHasSingleRed<T>;

/*
```
       |<-------------dim1------------>|
       |               |<-----dim3---->|
       |               |       |<-vmW->|
idx0 > +---------------+-------+-------+ ---------------
       |               |       |       |        ^      ^
       |               |       |       |        |      |
       |               |       |       |        |      |
       |        idx2 > |-------+-------+ --     + dim2 |
       |               |       |░░░░░░░| ^      |      |
       |               |       |░░░░░░░| + vmH  |      |
       |               |       |░░░░░░░| v      v      |
       |---------------+---------------+ --------      + dim0
       |               |       ^       |               |
       |               |       idx3    |               |
       |               |               |               |
       |               |               |               |
       |               |               |               |
       |               |               |               v
       +-------------------------------+ ---------------
                       ^
                       idx1
```
load ░:
(b2h) times load offset (i1h * b1h * mw + i1w * b1w + i2h * b2h * mw + i2w * b2w) stride (mw)
*/
template <int PLACE = HBM>
inline void load_mat(SIM_X86::tensor src, SIM_X86::tensor dst, const int dim2, const int dim3, const int idx2, const int idx3,
                     const int needH, const int needW, const int src_stride, const int dst_stride) {
    // const int offset = i1h * b1h * mw + i1w * b1w + i2h * b2h * mw + i2w * b2w;
    const int offset = idx2 * dim3 + idx3;

    const int unitLen = needW / 128;
    for (int i = 0; i < unitLen; i += 1) {
        int h = dlc_dma(tensor_slice(src, offset / 32 + i * 4), PLACE, tensor_slice(dst, i * 4), VMEM,
                        needH * 128, src_stride, dst_stride, 128, 7);
        dlc_sync(h);
    }
}

inline float8_128 reduce8(float8_128 val, reduceBinaryOp_t comb) {
    float8_128 v1 = v_row_rotate(val, 0);
    val = comb(val, v1);
    float8_128 v2 = v_row_rotate(v1, 0);
    val = comb(val, v2);
    float8_128 v3 = v_row_rotate(v2, 0);
    val = comb(val, v3);
    float8_128 v4 = v_row_rotate(v3, 0);
    val = comb(val, v4);
    float8_128 v5 = v_row_rotate(v4, 0);
    val = comb(val, v5);
    float8_128 v6 = v_row_rotate(v5, 0);
    val = comb(val, v6);
    float8_128 v7 = v_row_rotate(v6, 0);
    val = comb(val, v7);
    return val;
}

// 设置vmem中长度为size的地方为同一个值
template <class T> inline void memset_vmem(CxxTensor vmem, const Uint31 size, const T val) {
    Uint31 sized1024 = size / 1024 * 1024;
    float8_128 rval = bitAs<float8_128>(broadcast(val));
    for (Uint31 i = 0; i < sized1024; i += 1024) {
        v_f32_st_tnsr_b(i / 32, vmem, rval);
    }
    Uint31 len = size % 1024;
    Uint31 stmk = (1 << (len / 128)) - 1;
    v_f32_st_tnsr_st_msk(sized1024 / 32, vmem, 1, stmk, rval);
}

inline void transform_vmem(CxxTensor vmem, const Uint31 size, reduceUnaryOp_t proj) {
    Uint31 sized1024 = size / 1024 * 1024;
    for (Uint31 i = 0; i < sized1024; i += 1024) {
        v_f32_st_tnsr_b(i / 32, vmem, proj(v_f32_ld_tnsr_b(i / 32, vmem)));
    }
    Uint31 len = size % 1024;
    if (len != 0) {
        int stmk = (1 << (len / 128)) - 1;
        v_f32_st_tnsr_st_msk(sized1024 / 32, vmem, 1, stmk,
                             proj(v_f32_ld_tnsr_st_msk(sized1024 / 32, vmem, 1, stmk)));
    }
}

// [P, pR] => [1, pR]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    requires ReduceCalc<Calc>
inline void reduce_00010_hbm(CxxTensor in, CxxTensor out, Uint31 P, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    CxxTensor vout = vmem;
    vmem = vout + pR / 32;
    vmemlen -= pR;
    Uint31 max_width = soft_sdiv(vmemlen, P);
    Uint31 max_height = P;
    max_width = max_width / 128 * 128;
    max_width = min(max_width, pR);
    if (max_width == 0) {
        max_width = 128;
        max_height = soft_sdiv(vmemlen, max_width);
    }
    memset_vmem(vout, pR, Calc::ident);
    for (Uint31 x = 0; x < pR; x += max_width) {
        for (Uint31 y = 0; y < P; y += max_height) {
            Uint31 curwidth = min(pR - x, max_width);
            Uint31 curheight = min(P - y, max_height);
            load_mat<IN_PLACE>(in, vmem, P, pR, y, x, curheight, curwidth, pR, max_width);
            for (Uint31 x2 = 0; x2 < curwidth; x2 += 128) {
                float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                    Uint31 curvh = min(curheight - y2, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    bool8_128 curhmsk =
                        v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                    float8_128 rval =
                        load8_128_stride_ldmk((y2 * max_width + x2) / 32, max_width / 128, vmem, ldmk);
                    if constexpr (ReduceHasMap<Calc>) {
                        rval = Calc::map(rval);
                    }
                    float8_128 val =
                        v_f32_sel(/*curwmsk & */ curhmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                    acc = Calc::combine(acc, val);
                }
                acc = reduce8(acc, Calc::combine);
                float8_128 oldacc = v_f32_ld_tnsr_st_msk((x + x2) / 32, vout, 1, 1);
                v_f32_st_tnsr_st_msk((x + x2) / 32, vout, 1, 1, Calc::combine(oldacc, acc));
            }
        }
    }
    if constexpr (ReduceHasProject<Calc>) {
        transform_vmem(vout, pR, Calc::project);
    }
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((pR - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, pR, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [P, pR] => [1, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    requires ReduceCalc<Calc>
inline void reduce_00011_hbm(CxxTensor in, CxxTensor out, Uint31 P, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 len = P * pR;
    float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
    for (Uint31 i = 0; i < len; i += vmemlen) {
        Uint31 curl = min(len - i, vmemlen);
        int syncL = dlc_dma(in + i / 32, IN_PLACE, vmem, VMEM, curl, 128, 128, 128, 7);
        dlc_sync(syncL);
        Uint31 curl1024 = curl / 1024 * 1024;
        for (Uint31 j = 0; j < curl1024; j += 1024) {
            float8_128 rval = v_f32_ld_tnsr_b(j / 32, vmem);
            if constexpr (ReduceHasMap<Calc>) {
                rval = Calc::map(rval);
            }
            int8_128 ridxW;
            soft_sdiv_remainder_1024(v_s32_add(get_core_id(), v_u32_move_i(i + j)), v_u32_move_i(pR), &ridxW);
            bool8_128 msk = v_s32_cmp(LS, ridxW, v_u32_move_i(R));
            float8_128 val = v_f32_sel(msk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
            acc = Calc::combine(acc, val);
        }
        if (curl1024 != curl) {
            Uint31 curc = curl % 1024;
            int ldmk = (1 << (curc / 128)) - 1;
            float8_128 rval = v_f32_ld_tnsr_st_msk(curl1024 / 32, vmem, 1, ldmk);
            if constexpr (ReduceHasMap<Calc>) {
                rval = Calc::map(rval);
            }
            int8_128 ridxW;
            soft_sdiv_remainder_1024(v_s32_add(get_core_id(), v_u32_move_i(i + curl1024)), v_u32_move_i(pR),
                                     &ridxW);
            bool8_128 msk = v_s32_cmp(LS, ridxW, v_u32_move_i(R));
            bool8_128 mskR =
                v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curc / 128));
            float8_128 val = v_f32_sel(msk & mskR, bitAs<float8_128>(broadcast(Calc::ident)), rval);
            acc = Calc::combine(acc, val);
        }
    }
    acc = reduce8(acc, Calc::combine);
    acc = Calc::reduce_combine(acc);
    if constexpr (ReduceHasProject<Calc>) {
        acc = Calc::project(acc);
    }
    //     _UNUSED volatile float s = vstore_wait(acc);
    v_f32_st_tnsr_st_msk(0, vmem, 1, 1, acc);
    int syncS = dlc_dma(vmem, VMEM, out, OUT_PLACE, 128, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [pR] => [1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    requires ReduceCalc<Calc>
inline void reduce_00001_hbm(CxxTensor in, CxxTensor out, Uint31 B, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    if (B <= 0) {
        return;
    }
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 bs = soft_sdiv(vmemlen, pR + 128);
    if (bs <= 0 || B == 1) {
        reduce_00011_hbm<Calc, IN_PLACE, OUT_PLACE>(in, out, B, R, vmem, vmemlen);
        return;
    }
    Uint31 voutlen = bs * 128;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    int8_128 premsk = v_u32_and(get_core_id(), v_u32_move_i(127));

    int syncS = DONE;
    for (Uint31 b = 0; b < B; b += bs) {
        Uint31 curb = min(B - b, bs);
        dlc_sync(syncS);
        int syncL = dlc_dma(in + b * pR / 32, IN_PLACE, vmem, VMEM, curb * pR, 128, 128, 128, 7);
        dlc_sync(syncL);
        if constexpr (ReduceHasFifoRed<Calc>) {
            Uint31 prefill = min(curb, 2);
            Uint31 fill = curb - prefill;
            for (Uint31 y = 0; y < prefill; ++y) {
                float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                Uint31 curvh = min(curb - y, 8);
                Uint31 ldmk = (1 << curvh) - 1;
                for (Uint31 x = 0; x < R; x += 128) {
                    Uint31 curvw = min(R - x, 128);
                    bool8_128 curwmsk = v_s32_cmp(LS, premsk, v_u32_move_i(curvw));
                    float8_128 rval = load8_128_stride_ldmk((y * pR + x) / 32, pR / 128, vmem, ldmk);
                    if constexpr (ReduceHasMap<Calc>) {
                        rval = Calc::map(rval);
                    }
                    float8_128 val = v_f32_sel(curwmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                    acc = Calc::combine(acc, val);
                }
                Calc::template reduce_combine_push<true>(acc);
            }
            for (Uint31 z = 0; z < fill; ++z) {
                Uint31 y = z + prefill;
                {
                    float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                    Uint31 curvh = min(curb - y, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    for (Uint31 x = 0; x < R; x += 128) {
                        Uint31 curvw = min(R - x, 128);
                        bool8_128 curwmsk = v_s32_cmp(LS, premsk, v_u32_move_i(curvw));
                        float8_128 rval = load8_128_stride_ldmk((y * pR + x) / 32, pR / 128, vmem, ldmk);
                        if constexpr (ReduceHasMap<Calc>) {
                            rval = Calc::map(rval);
                        }
                        float8_128 val = v_f32_sel(curwmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                        acc = Calc::combine(acc, val);
                    }
                    Calc::template reduce_combine_push<true>(acc);
                }
                {
                    Uint31 curvh = min(curb - z, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    float8_128 acc = Calc::template reduce_combine_pop<true>();
                    if constexpr (ReduceHasProject<Calc>) {
                        acc = Calc::project(acc);
                    }
                    v_f32_st_tnsr_st_msk(z * 128 / 32, vout, 1, ldmk, acc);
                }
            }
            for (Uint31 z = fill; z < curb; ++z) {
                Uint31 curvh = min(curb - z, 8);
                Uint31 ldmk = (1 << curvh) - 1;
                float8_128 acc = Calc::template reduce_combine_pop<true>();
                if constexpr (ReduceHasProject<Calc>) {
                    acc = Calc::project(acc);
                }
                v_f32_st_tnsr_st_msk(z * 128 / 32, vout, 1, ldmk, acc);
            }
        } else {
            for (Uint31 y = 0; y < curb; y += 8) {
                Uint31 curvh = min(curb - y, 8);
                float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                Uint31 ldmk = (1 << curvh) - 1;
                for (Uint31 x = 0; x < R; x += 128) {
                    Uint31 curvw = min(R - x, 128);
                    bool8_128 curwmsk = v_s32_cmp(LS, premsk, v_u32_move_i(curvw));
                    float8_128 rval = load8_128_stride_ldmk((y * pR + x) / 32, pR / 128, vmem, ldmk);
                    if constexpr (ReduceHasMap<Calc>) {
                        rval = Calc::map(rval);
                    }
                    float8_128 val = v_f32_sel(curwmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                    acc = Calc::combine(acc, val);
                }
                acc = Calc::reduce_combine(acc);
                if constexpr (ReduceHasProject<Calc>) {
                    acc = Calc::project(acc);
                }
                v_f32_st_tnsr_st_msk(y * 128 / 32, vout, 1, ldmk, acc);
            }
        }
        // _UNUSED volatile float s =
        //     vstore_wait(v_f32_ld_tnsr_st_msk(max(0, (curb - 1)) * 128 / 32, vout, 1, 1));
        syncS = dlc_dma(vout, VMEM, out + b * 128 / 32, OUT_PLACE, curb * 128, 128, 128, 128, 7);
    }
    dlc_sync(syncS);
}

// [K, P, pR] => [1, P, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    requires ReduceCalc<Calc>
inline void reduce_00101_hbm(CxxTensor in, CxxTensor out, Uint31 K, Uint31 P, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = P * 128;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;
    Uint31 PpR = P * pR;
    Uint31 max_width = soft_sdiv(vmemlen, K);
    Uint31 max_height = K;
    max_width = max_width / 128 * 128;
    max_width = min(max_width, PpR);
    if (max_width == 0) {
        max_width = 128;
        max_height = soft_sdiv(vmemlen, max_width);
    }
    memset_vmem(vout, voutlen, Calc::ident);
    for (Uint31 x = 0; x < PpR; x += max_width) {
        for (Uint31 y = 0; y < K; y += max_height) {
            Uint31 curw = min(PpR - x, max_width);
            Uint31 curh = min(K - y, max_height);
            load_mat<IN_PLACE>(in, vmem, K, PpR, y, x, curh, curw, PpR, max_width);
            for (Uint31 x2 = 0; x2 < curw; x2 += 128) {
                // pR and curw all multiply of 128, so each 128 always in a P;
                Uint31 pidx = soft_sdiv(x + x2, pR);
                float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                bool8_128 curwmsk = v_s32_cmp(LS, v_u32_and(get_core_id(), v_u32_move_i(127)),
                                              v_u32_move_i(pidx * pR + R - x - x2));
                for (Uint31 y2 = 0; y2 < curh; y2 += 8) {
                    Uint31 curvh = min(curh - y2, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    bool8_128 curhmsk =
                        v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                    float8_128 rval =
                        load8_128_stride_ldmk((y2 * max_width + x2) / 32, max_width / 128, vmem, ldmk);
                    if constexpr (ReduceHasMap<Calc>) {
                        rval = Calc::map(rval);
                    }
                    float8_128 val =
                        v_f32_sel(curwmsk & curhmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                    acc = Calc::combine(acc, val);
                }
                acc = reduce8(acc, Calc::combine);
                acc = Calc::reduce_combine(acc);
                float8_128 oldacc = v_f32_ld_tnsr_st_msk(pidx * 128 / 32, vout, 1, 1);
                v_f32_st_tnsr_st_msk(pidx * 128 / 32, vout, 1, 1, Calc::combine(oldacc, acc));
            }
        }
    }
    if constexpr (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((voutlen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, voutlen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [H, K, P, pR] => [1, K, 1, pR]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    requires ReduceCalc<Calc>
inline void reduce_01010_hbm(CxxTensor in, CxxTensor out, Uint31 H, Uint31 K, Uint31 P, Uint31 R,
                             CxxTensor vmem, Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = K * pR;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;

    Uint31 max_width = soft_sdiv(vmemlen, P);
    Uint31 max_height = P;
    max_width = max_width / 128 * 128;
    max_width = min(max_width, pR);
    if (max_width == 0) {
        max_width = 128;
        max_height = soft_sdiv(vmemlen, max_width);
    }

    memset_vmem(vout, voutlen, Calc::ident);

    for (Uint31 h = 0; h < H; h += 1) {
        for (Uint31 k = 0; k < K; k += 1) {
            for (Uint31 x = 0; x < pR; x += max_width) {
                for (Uint31 y = 0; y < P; y += max_height) {
                    Uint31 curwidth = min(pR - x, max_width);
                    Uint31 curheight = min(P - y, max_height);
                    load_mat<IN_PLACE>(in + (h * K * P * pR + k * P * pR) / 32, vmem, P, pR, y, x, curheight,
                                       curwidth, pR, max_width);
                    for (Uint31 x2 = 0; x2 < curwidth; x2 += 128) {
                        float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                        for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                            Uint31 curvh = min(curheight - y2, 8);
                            Uint31 ldmk = (1 << curvh) - 1;
                            bool8_128 curhmsk =
                                v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                            float8_128 rval = load8_128_stride_ldmk((y2 * max_width + x2) / 32,
                                                                    max_width / 128, vmem, ldmk);
                            if constexpr (ReduceHasMap<Calc>) {
                                rval = Calc::map(rval);
                            }
                            float8_128 val =
                                v_f32_sel(curhmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                            acc = Calc::combine(acc, val);
                        }
                        acc = reduce8(acc, Calc::combine);
                        Uint31 outaddr = k * pR + (x + x2);
                        float8_128 oldacc = v_f32_ld_tnsr_st_msk(outaddr / 32, vout, 1, 1);
                        v_f32_st_tnsr_st_msk(outaddr / 32, vout, 1, 1, Calc::combine(oldacc, acc));
                    }
                }
            }
        }
    }
    if constexpr (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((voutlen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, voutlen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [H, K, P, pR] => [1, K, 1, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    requires ReduceCalc<Calc>
inline void reduce_01011_hbm(CxxTensor in, CxxTensor out, Uint31 H, Uint31 K, Uint31 P, Uint31 R,
                             CxxTensor vmem, Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = K * 128;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;

    Uint31 max_width = soft_sdiv(vmemlen, P);
    Uint31 max_height = P;
    max_width = max_width / 128 * 128;
    max_width = min(max_width, pR);
    if (max_width == 0) {
        max_width = 128;
        max_height = soft_sdiv(vmemlen, max_width);
    }

    memset_vmem(vout, voutlen, Calc::ident);

    for (Uint31 h = 0; h < H; h += 1) {
        for (Uint31 k = 0; k < K; k += 1) {
            for (Uint31 x = 0; x < pR; x += max_width) {
                for (Uint31 y = 0; y < P; y += max_height) {
                    Uint31 curwidth = min(pR - x, max_width);
                    Uint31 curheight = min(P - y, max_height);
                    load_mat<IN_PLACE>(in + (h * K * P * pR + k * P * pR) / 32, vmem, P, pR, y, x, curheight,
                                       curwidth, pR, max_width);
                    for (Uint31 x2 = 0; x2 < curwidth; x2 += 128) {
                        float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                        bool8_128 curwmsk = v_s32_cmp(LS, v_u32_and(get_core_id(), v_u32_move_i(127)),
                                                      v_u32_move_i(R - x - x2));
                        for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                            Uint31 curvh = min(curheight - y2, 8);
                            Uint31 ldmk = (1 << curvh) - 1;
                            bool8_128 curhmsk =
                                v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                            float8_128 rval = load8_128_stride_ldmk((y2 * max_width + x2) / 32,
                                                                    max_width / 128, vmem, ldmk);
                            if constexpr (ReduceHasMap<Calc>) {
                                rval = Calc::map(rval);
                            }
                            float8_128 val =
                                v_f32_sel(curwmsk & curhmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                            acc = Calc::combine(acc, val);
                        }
                        acc = reduce8(acc, Calc::combine);
                        acc = Calc::reduce_combine(acc);
                        Uint31 outaddr = k * 128;
                        float8_128 oldacc = v_f32_ld_tnsr_st_msk(outaddr / 32, vout, 1, 1);
                        v_f32_st_tnsr_st_msk(outaddr / 32, vout, 1, 1, Calc::combine(oldacc, acc));
                    }
                }
            }
        }
    }
    if constexpr (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((voutlen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, voutlen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [T, H, K, P, pR] => [1, H, 1, P, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    requires ReduceCalc<Calc>
inline void reduce_10101_hbm(CxxTensor in, CxxTensor out, Uint31 T, Uint31 H, Uint31 K, Uint31 P, Uint31 R,
                             CxxTensor vmem, Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = H * P * 128;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;
    Uint31 PpR = P * pR;
    Uint31 max_width = soft_sdiv(vmemlen, K);
    Uint31 max_height = K;
    max_width = max_width / 128 * 128;
    max_width = min(max_width, PpR);
    if (max_width == 0) {
        max_width = 128;
        max_height = soft_sdiv(vmemlen, max_width);
    }
    memset_vmem(vout, voutlen, Calc::ident);

    for (Uint31 t = 0; t < T; ++t) {
        for (Uint31 h = 0; h < H; ++h) {
            for (Uint31 x = 0; x < PpR; x += max_width) {
                for (Uint31 y = 0; y < K; y += max_height) {
                    Uint31 curw = min(PpR - x, max_width);
                    Uint31 curh = min(K - y, max_height);
                    load_mat<IN_PLACE>(in + (t * H * K * PpR + h * K * PpR) / 32, vmem, K, PpR, y, x, curh,
                                       curw, PpR, max_width);
                    for (Uint31 x2 = 0; x2 < curw; x2 += 128) {
                        // pR and curw all multiply of 128, so each 128 always in a P;
                        Uint31 pidx = soft_sdiv(x + x2, pR);
                        float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                        bool8_128 curwmsk = v_s32_cmp(LS, v_u32_and(get_core_id(), v_u32_move_i(127)),
                                                      v_u32_move_i(pidx * pR + R - x - x2));
                        for (Uint31 y2 = 0; y2 < curh; y2 += 8) {
                            Uint31 curvh = min(curh - y2, 8);
                            Uint31 ldmk = (1 << curvh) - 1;
                            bool8_128 curhmsk =
                                v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                            float8_128 rval = load8_128_stride_ldmk((y2 * max_width + x2) / 32,
                                                                    max_width / 128, vmem, ldmk);
                            if constexpr (ReduceHasMap<Calc>) {
                                rval = Calc::map(rval);
                            }
                            float8_128 val =
                                v_f32_sel(curwmsk & curhmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                            acc = Calc::combine(acc, val);
                        }
                        acc = reduce8(acc, Calc::combine);
                        acc = Calc::reduce_combine(acc);
                        uint outaddr = h * P * 128 + pidx * 128;
                        float8_128 oldacc = v_f32_ld_tnsr_st_msk(outaddr / 32, vout, 1, 1);
                        v_f32_st_tnsr_st_msk(outaddr / 32, vout, 1, 1, Calc::combine(oldacc, acc));
                    }
                }
            }
        }
    }

    if constexpr (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((voutlen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, voutlen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

inline float128_128 load128_128cmem_h(const int off, SIM_X86::tensor t, int h) {
    float8_128 a[16];
    for (int i = 0; i < 16; i++) {
        a[i] = v_f32_fxc_load(off + i * 32, t, 1, 255);
    }
    return v_concat_16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
                       a[14], a[15]);
}

// [s2, s1, 1] => [s2, s1]
inline void sqeeuze_low_cmem_in_hbm_out(CxxTensor cmem, Uint31 s2, Uint31 s1, CxxTensor vmem,
                                        CxxTensor hbmout) {
    sync_device();
    if (get_device_id() == 1) {
        return;
    }
    // Print(const_cast<char *>("s2: %d\n"), s2.sval);
    // Print(const_cast<char *>("s1: %d\n"), s1.sval);
    Uint31 ps1 = (s1 + 127) & (-128);
    for (Uint31 i2 = 0; i2 < s2; i2 += 1) {
        for (Uint31 i1 = 0; i1 < s1; i1 += 128) {
            Uint31 curs1 = min(s1 - i1, 128);
            float128_128 r = load128_128cmem_h((i2 * s1 * 128 + i1 * 128) / 32, cmem, curs1);
            float8_128 tr = sub_vector(m_transpose_128_128_nws(r, 0), 0);
            v_f32_st_tnsr_st_msk((i2 * ps1 + i1) / 32, vmem, 1, 1, tr);
        }
    }
    // _UNUSED volatile float x = vstore_wait(v_f32_ld_tnsr_st_msk(s2 * ps1 - 128 / 32, vmem, 1, 1));
    // _UNUSED volatile float y = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem, 1, 1));
    dlc_sync(dlc_dma(vmem, VMEM, hbmout, HBM, s2 * ps1, 128, 128, 128, 7));
}

// reduce_d0 as true
inline void get_sqeeuze_size(Uint31 d0, Uint31 d1, Uint31 d2, Uint31 d3, Uint31 d4, Uint31 reduce_d0,
                             Uint31 reduce_d1, Uint31 reduce_d2, Uint31 reduce_d3, Uint31 reduce_d4,
                             Uint31 &s2, Uint31 &s1) {
    d0 = reduce_d0 != 0 ? Uint31(1) : d0;
    d1 = reduce_d1 != 0 ? Uint31(1) : d1;
    d2 = reduce_d2 != 0 ? Uint31(1) : d2;
    d3 = reduce_d3 != 0 ? Uint31(1) : d3;
    d4 = reduce_d4 != 0 ? Uint31(1) : d4;

    s2 = d4 * d3 * d2;
    s1 = d1;

    if (reduce_d1 != 0) {
        s2 = d4 * d3;
        s1 = d2;
    } else {
        return;
    }

    if (reduce_d2 != 0) {
        s2 = d4;
        s1 = d3;
    } else {
        return;
    }

    if (reduce_d3 != 0) {
        s2 = 1;
        s1 = d4;
    } else {
        return;
    }

    if (reduce_d4 != 0) {
        s2 = 1;
        s1 = 1;
    } else {
        return;
    }
}

// d0 need reduce
template <class Calc>
    requires ReduceCalc<Calc>
inline void reduce_hbm_dimlist_2xys_sqeeuze(CxxTensor in, CxxTensor hbmout, Uint31 d0, Uint31 d1, Uint31 d2,
                                            Uint31 d3, Uint31 d4, Uint31 reduce_d0, Uint31 reduce_d1,
                                            Uint31 reduce_d2, Uint31 reduce_d3, Uint31 reduce_d4,
                                            CxxTensor vmem, Uint31 vmemlen, CxxTensor out) {
    Uint31 s2(0), s1(0);
    get_sqeeuze_size(d0, d1, d2, d3, d4, reduce_d0, reduce_d1, reduce_d2, reduce_d3, reduce_d4, s2, s1);
    Uint31 pd0 = (d0 + 127) / 128 * 128;
    if (reduce_d4 == reduce_d3) {
        d3 *= d4;
        d4 = 1;
        reduce_d4 = 0;
    }

    if (reduce_d2 == reduce_d3) {
        d2 *= d3;
        d3 = d4;
        reduce_d3 = reduce_d4;
        d4 = 1;
        reduce_d4 = 0;
    }

    if (reduce_d1 == reduce_d2) {
        d1 *= d2;
        d2 = d3;
        reduce_d2 = reduce_d3;
        d3 = d4;
        reduce_d3 = reduce_d4;
        d4 = 1;
        reduce_d4 = 0;
    }

    uint reducePat = (reduce_d4 << 4) | (reduce_d3 << 3) | (reduce_d2 << 2) | (reduce_d1 << 1) | reduce_d0;
    if (reducePat == 0b00001) {
        Uint31 B = d4 * d3 * d2 * d1;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        reduce_00001_hbm<Calc, HBM, CMEM>(in + off * pd0 / 32, out + off * 128 / 32, len, d0, vmem, vmemlen);
    } else if (reducePat == 0b00011) {
        Uint31 B = d4 * d3 * d2;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00011_hbm<Calc, HBM, CMEM>(in + (off + i) * d1 * pd0 / 32, out + (off + i) * 128 / 32, d1,
                                              d0, vmem, vmemlen);
        }
    } else if (reducePat == 0b00101) {
        Uint31 B = d4 * d3;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00101_hbm<Calc, HBM, CMEM>(in + (off + i) * d2 * d1 * pd0 / 32,
                                              out + (off + i) * d1 * 128 / 32, d2, d1, d0, vmem, vmemlen);
        }
    } else if (reducePat == 0b01011) {
        Uint31 B = d4;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        Uint31 uil = d3 * d2 * d1 * pd0;
        Uint31 uol = d2 * 128;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_01011_hbm<Calc, HBM, CMEM>(in + (off + i) * uil / 32, out + (off + i) * uol / 32, d3, d2,
                                              d1, d0, vmem, vmemlen);
        }
    } else if (reducePat == 0b10101) {
        if (get_device_id() == 0) {
            reduce_10101_hbm<Calc, HBM, CMEM>(in, out, d4, d3, d2, d1, d0, vmem, vmemlen);
        }
    }
    sqeeuze_low_cmem_in_hbm_out(out, s2, s1, vmem, hbmout);
}

template <class Calc>
    requires ReduceCalc<Calc>
inline void reduce_hbm_dimlist_2xys(CxxTensor in, CxxTensor out, Uint31 d0, Uint31 d1, Uint31 d2, Uint31 d3,
                                    Uint31 d4, Uint31 reduce_d0, Uint31 reduce_d1, Uint31 reduce_d2,
                                    Uint31 reduce_d3, Uint31 reduce_d4, CxxTensor vmem, Uint31 vmemlen,
                                    Uint31 keepDim = 1, SIM_X86::DLCMem *info = nullptr) {
    Uint31 pd0 = (d0 + 127) / 128 * 128;

    if (reduce_d0 != 0 && keepDim == 0) {
        reduce_hbm_dimlist_2xys_sqeeuze<Calc>(in, out, d0, d1, d2, d3, d4, reduce_d0, reduce_d1, reduce_d2,
                                              reduce_d3, reduce_d4, vmem, vmemlen, info->cmem_addr);
        return;
    }

    if (reduce_d4 == reduce_d3) {
        d3 *= d4;
        d4 = 1;
        reduce_d4 = 0;
    }

    if (reduce_d2 == reduce_d3) {
        d2 *= d3;
        d3 = d4;
        reduce_d3 = reduce_d4;
        d4 = 1;
        reduce_d4 = 0;
    }

    if (reduce_d1 == reduce_d2) {
        d1 *= d2;
        d2 = d3;
        reduce_d2 = reduce_d3;
        d3 = d4;
        reduce_d3 = reduce_d4;
        d4 = 1;
        reduce_d4 = 0;
    }

    // in this case, no need to care about padding, because it not take part in result we need
    // d0 lost padding info
    if (reduce_d0 == 0 && reduce_d1 == 0) {
        d0 = pd0 * d1;
        pd0 = d0;
        d1 = d2;
        reduce_d1 = reduce_d2;
        d2 = d3;
        reduce_d2 = reduce_d3;
        d3 = d4;
        reduce_d3 = reduce_d4;
        d4 = 1;
        reduce_d4 = 0;
    }

    uint reducePat = (reduce_d4 << 4) | (reduce_d3 << 3) | (reduce_d2 << 2) | (reduce_d1 << 1) | reduce_d0;

    // Print("reducePat: %d\n", reducePat);

    if (reducePat == 0b00001) {
        Uint31 B = d4 * d3 * d2 * d1;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        reduce_00001_hbm<Calc>(in + off * pd0 / 32, out + off * 128 / 32, len, d0, vmem, vmemlen);
    } else if (reducePat == 0b00010) {
        Uint31 B = d4 * d3 * d2;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00010_hbm<Calc>(in + (off + i) * d1 * pd0 / 32, out + (off + i) * pd0 / 32, d1, d0, vmem,
                                   vmemlen);
        }
    } else if (reducePat == 0b00011) {
        Uint31 B = d4 * d3 * d2;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00011_hbm<Calc>(in + (off + i) * d1 * pd0 / 32, out + (off + i) * 128 / 32, d1, d0, vmem,
                                   vmemlen);
        }
    } else if (reducePat == 0b00101) {
        Uint31 B = d4 * d3;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00101_hbm<Calc>(in + (off + i) * d2 * d1 * pd0 / 32, out + (off + i) * d1 * 128 / 32, d2,
                                   d1, d0, vmem, vmemlen);
        }
    } else if (reducePat == 0b01010) {
        Uint31 B = d4;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        Uint31 uil = d3 * d2 * d1 * pd0;
        Uint31 uol = d2 * pd0;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_01010_hbm<Calc>(in + (off + i) * uil / 32, out + (off + i) * uol / 32, d3, d2, d1, d0,
                                   vmem, vmemlen);
        }
    } else if (reducePat == 0b01011) {
        Uint31 B = d4;
        Uint31 B0 = B / 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        Uint31 uil = d3 * d2 * d1 * pd0;
        Uint31 uol = d2 * 128;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_01011_hbm<Calc>(in + (off + i) * uil / 32, out + (off + i) * uol / 32, d3, d2, d1, d0,
                                   vmem, vmemlen);
        }
    } else if (reducePat == 0b10101) {
        if (get_device_id() == 0) {
            reduce_10101_hbm<Calc>(in, out, d4, d3, d2, d1, d0, vmem, vmemlen);
        }
    }
}