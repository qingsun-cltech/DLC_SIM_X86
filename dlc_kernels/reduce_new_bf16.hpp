#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #pragma once
#include "bf16.h"
#include "cxxbase.hpp"
#include "ldst.h"
// #include "typehint.h"

inline int8_128 broadcast(int v) { return v_u32_move_i(v); }
inline float8_128 broadcast(float v) { return v_u32_move_f(v); }

typedef float8_128 (*reduceBinaryOp_t)(float8_128, float8_128);
typedef float8_128 (*reduceUnaryOp_t)(float8_128);

template <typename T>
/* concept */ ReduceHasMap = /* requires */ {
    { T::map(float8_128{}) } -> Types::same_as<float8_128>;
};

template <typename T>
/* concept */ ReduceHasProject = /* requires */ {
    { T::project(float8_128{}) } -> Types::same_as<float8_128>;
};

template <typename T>
/* concept */ ReduceHasIdent = /* requires */ { T::ident; };

template <typename T>
/* concept */ ReduceHasCombine = /* requires */ {
    { T::combine(float8_128{}, float8_128{}) } -> Types::same_as<float8_128>;
};

template <typename T>
/* concept */ ReduceHasSingleRed = /* requires */ {
    { T::reduce_combine(float8_128{}) } -> Types::same_as<float8_128>;
};

template <typename T>
/* concept */ ReduceHasFifoRed = /* requires */ {
    T::template reduce_combine_push<true>(float8_128{});
    T::template reduce_combine_push<false>(float8_128{});
    { T::template reduce_combine_pop<true>() } -> Types::same_as<float8_128>;
    { T::template reduce_combine_pop<false>() } -> Types::same_as<float8_128>;
};

template <typename T>
/* concept */ ReduceCalc = ReduceHasIdent<T> && ReduceHasCombine<T> && ReduceHasSingleRed<T>;

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
    const int dim3bf = ((dim3 + 255) / 256 * 256) / 2;
    const int idx3bf = idx3 / 256 * 128;
    const int offsetbf = idx2 * dim3bf + idx3bf;

    const int ssbf = src_stride / 2;
    const int dsbf = dst_stride / 2;

    const int unitLenbf = dsbf / 128;
    for (int i = 0; i < unitLenbf; i += 1) {
        int h = dlc_dma(tensor_slice(src, offsetbf / 32 + i * 4), PLACE, tensor_slice(dst, i * 4), VMEM,
                        needH * 128, ssbf, dsbf, 128, 7);
        dlc_sync(h);
    }
    bf16ToF32_h(dst, needH * dsbf, needH, dst_stride);
}

template <int PLACE = HBM>
inline void load_mat_f32(SIM_X86::tensor src, SIM_X86::tensor dst, const int dim2, const int dim3, const int idx2,
                         const int idx3, const int needH, const int needW, const int src_stride,
                         const int dst_stride) {
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
inline void memset_vmem(CxxTensor vmem, const Uint31 size, const float val) {
    Uint31 sized1024 = size / 1024 * 1024;
    for (Uint31 i = 0; i < sized1024; i += 1024) {
        v_f32_st_tnsr_b(i / 32, vmem, v_u32_move_f(val));
    }
    Uint31 len = size % 1024;
    Uint31 stmk = (1 << (len / 128)) - 1;
    v_f32_st_tnsr_st_msk(sized1024 / 32, vmem, 1, stmk, v_u32_move_f(val));
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

// [pR] => [1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_00001_hbm(CxxTensor in, CxxTensor out, Uint31 B, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 bs = soft_sdiv(vmemlen, pR256 / 2 + 128);
    Uint31 voutlen = bs * 128;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    int8_128 premsk = v_u32_and(get_core_id(), v_u32_move_i(127));
    int syncS = DONE;
    for (Uint31 b = 0; b < B; b += bs) {
        Uint31 curb = min(B - b, bs);
        dlc_sync(syncS);
        int syncL = dlc_dma(in + b * pR256 / 64, IN_PLACE, vmem, VMEM, curb * pR256 / 2, 128, 128, 128, 7);
        dlc_sync(syncL);
        for (Uint31 y = 0; y < curb; y += 8) {
            Uint31 curvh = min(curb - y, 8);
            float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
            Uint31 ldmk = (1 << curvh) - 1;
            Uint31 R256 = pR / 256 * 256;
            for (Uint31 x = 0; x < R256; x += 256) {
                // Uint31 xlo = x;
                Uint31 xhi = x + 128;
                // lo no need filter padding
                Uint31 curvwhi = min(R - xhi, 128);
                bool8_128 curwmskhi = v_s32_cmp(LS, premsk, v_u32_move_i(curvwhi));
                float8_128 rval =
                    load8_128_stride_ldmk((y * pR256 / 2 + x / 2) / 32, pR256 / 256, vmem, ldmk);
                int8_128 tmphi = v_u32_and(*(int8_128 *)(&rval), v_u32_move_i(0xffff0000));
                int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                float8_128 rvallo = *(float8_128 *)(&tmplo);
                float8_128 rvalhi = *(float8_128 *)(&tmphi);
                if /* constexpr */ (ReduceHasMap<Calc>) {
                    rvallo = Calc::map(rvallo);
                    rvalhi = Calc::map(rvalhi);
                }
                acc = Calc::combine(acc, rvallo);
                float8_128 valhi = v_f32_sel(curwmskhi, v_u32_move_f(Calc::ident), rvalhi);
                acc = Calc::combine(acc, valhi);
            }
            if (R256 != pR) {
                Uint31 x = R256;
                Uint31 curvw = min(R - x, 128);
                bool8_128 curwmsk = v_s32_cmp(LS, premsk, v_u32_move_i(curvw));
                float8_128 rval =
                    load8_128_stride_ldmk((y * pR256 / 2 + x / 2) / 32, pR256 / 256, vmem, ldmk);
                int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                rval = *(float8_128 *)(&tmplo);
                if /* constexpr */ (ReduceHasMap<Calc>) {
                    rval = Calc::map(rval);
                }
                float8_128 val = v_f32_sel(curwmsk, v_u32_move_f(Calc::ident), rval);
                acc = Calc::combine(acc, val);
            }
            acc = Calc::reduce_combine(acc);
            if /* constexpr */ (ReduceHasProject<Calc>) {
                acc = Calc::project(acc);
            }
            int8_128 accb = float_to_bfloat16(acc, acc);
            v_f32_st_tnsr_st_msk(y * 128 / 32, vout, 1, ldmk, *(float8_128 *)(&accb));
        }
        // _UNUSED volatile float s =
        //     vstore_wait(v_f32_ld_tnsr_st_msk(max(0, (curb - 1)) * 128 / 32, vout, 1, 1));
        syncS = dlc_dma(vout, VMEM, out + b * 128 / 32, OUT_PLACE, curb * 128, 128, 128, 128, 7);
    }
    dlc_sync(syncS);
}

// [P, pR] => [1, pR]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_00010_hbm(CxxTensor in, CxxTensor out, Uint31 P, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 pR = (R + 127) / 128 * 128;
    CxxTensor vout = vmem;
    vmem = vout + pR / 32;
    vmemlen -= pR;
    Uint31 max_width_s = soft_sdiv(vmemlen, P);
    Uint31 max_height = P;
    max_width_s = max_width_s / 128 * 128;
    max_width_s = min(max_width_s, pR256 / 2);
    if (max_width_s == 0) {
        max_width_s = 128;
        max_height = soft_sdiv(vmemlen, max_width_s);
    }
    Uint31 max_width = max_width_s * 2;
    memset_vmem(vout, pR, Calc::ident);
    for (Uint31 x = 0; x < pR; x += max_width) {
        for (Uint31 y = 0; y < P; y += max_height) {
            Uint31 curwidth = min(pR - x, max_width);
            Uint31 curheight = min(P - y, max_height);
            load_mat_f32<IN_PLACE>(in, vmem, P, pR256 / 2, y, x / 2, curheight, (curwidth + 255) / 256 * 128,
                                   pR256 / 2, max_width_s);
            Uint31 curwidth256 = curwidth / 256 * 256;
            for (Uint31 x2 = 0; x2 < curwidth256; x2 += 256) {
                Uint31 x2lo = x2;
                // Uint31 x2hi = x2 + 128;
                float8_128 acchi = bitAs<float8_128>(broadcast(Calc::ident));
                float8_128 acclo = bitAs<float8_128>(broadcast(Calc::ident));
                for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                    Uint31 curvh = min(curheight - y2, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    bool8_128 curhmsk =
                        v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                    float8_128 rval =
                        load8_128_stride_ldmk((y2 * max_width_s + x2 / 2) / 32, max_width / 256, vmem, ldmk);
                    int8_128 tmphi = v_u32_and(*(int8_128 *)(&rval), v_u32_move_i(0xffff0000));
                    int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                    float8_128 rvalhi = *(float8_128 *)(&tmphi);
                    float8_128 rvallo = *(float8_128 *)(&tmplo);
                    if /* constexpr */ (ReduceHasMap<Calc>) {
                        rvalhi = Calc::map(rvalhi);
                        rvallo = Calc::map(rvallo);
                    }
                    float8_128 valhi = v_f32_sel(curhmsk, v_u32_move_f(Calc::ident), rvalhi);
                    float8_128 vallo = v_f32_sel(curhmsk, v_u32_move_f(Calc::ident), rvallo);
                    acchi = Calc::combine(acchi, valhi);
                    acclo = Calc::combine(acclo, vallo);
                }
                acchi = reduce8(acchi, Calc::combine);
                acclo = reduce8(acclo, Calc::combine);
                float8_128 acc2 =
                    v_f32_sel(v_s32_cmp(LS, get_core_id(), v_u32_move_i(128)), v_row_rotate(acchi, 1), acclo);
                float8_128 oldacc2 = v_f32_ld_tnsr_st_msk((x + x2lo) / 32, vout, 1, 0b11);
                // float8_128 oldacclo = v_f32_ld_tnsr_st_msk((x + x2lo) / 32, vout, 1, 1);
                v_f32_st_tnsr_st_msk((x + x2lo) / 32, vout, 1, 0b11, Calc::combine(oldacc2, acc2));
                // v_f32_st_tnsr_st_msk((x + x2lo) / 32, vout, 1, 1, Calc::combine(oldacclo, acclo));
            }
            if (curwidth256 != curwidth) {
                Uint31 x2 = curwidth256;
                Uint31 x2lo = x2;
                float8_128 acclo = v_u32_move_f(Calc::ident);
                for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                    Uint31 curvh = min(curheight - y2, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    bool8_128 curhmsk =
                        v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                    float8_128 rval =
                        load8_128_stride_ldmk((y2 * max_width_s + x2 / 2) / 32, max_width / 256, vmem, ldmk);
                    int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                    float8_128 rvallo = *(float8_128 *)(&tmplo);
                    if /* constexpr */ (ReduceHasMap<Calc>) {
                        rvallo = Calc::map(rvallo);
                    }
                    float8_128 vallo = v_f32_sel(/*curwmsk & */ curhmsk, v_u32_move_f(Calc::ident), rvallo);
                    acclo = Calc::combine(acclo, vallo);
                }
                acclo = reduce8(acclo, Calc::combine);
                float8_128 oldacclo = v_f32_ld_tnsr_st_msk((x + x2lo) / 32, vout, 1, 1);
                v_f32_st_tnsr_st_msk((x + x2lo) / 32, vout, 1, 1, Calc::combine(oldacclo, acclo));
            }
        }
    }
    if /* constexpr */ (ReduceHasProject<Calc>) {
        transform_vmem(vout, pR, Calc::project);
    }
    Uint31 bfolen = pR256 / 2;
    f32ToBf16_h(vout, bfolen, 1, pR);
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((bfolen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, bfolen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [P, pR] => [1, pR]
// Rs, Rl is numel, Rl, Rs should 256k
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_00010_hbm_split_R(CxxTensor in, CxxTensor out, Uint31 P, Uint31 R, Uint31 Rs, Uint31 Rl,
                                     CxxTensor vmem, Uint31 vmemlen) {
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 pR = (R + 127) / 128 * 128;
    CxxTensor vout = vmem;
    vmem = vout + pR / 32;
    vmemlen -= pR;
    Uint31 max_width_s = soft_sdiv(vmemlen, P);
    Uint31 max_height = P;
    max_width_s = max_width_s / 128 * 128;
    max_width_s = min(max_width_s, Rl / 2);
    if (max_width_s == 0) {
        max_width_s = 128;
        max_height = soft_sdiv(vmemlen, max_width_s);
    }
    Uint31 max_width = max_width_s * 2;
    memset_vmem(vout, pR, Calc::ident);
    for (Uint31 x = 0; x < Rl; x += max_width) {
        for (Uint31 y = 0; y < P; y += max_height) {
            Uint31 curwidth = min(pR - (x + Rs), max_width);
            Uint31 curheight = min(P - y, max_height);
            load_mat_f32<IN_PLACE>(in, vmem, P, pR256 / 2, y, (x + Rs) / 2, curheight,
                                   (curwidth + 255) / 256 * 128, pR256 / 2, max_width_s);
            Uint31 curwidth256 = curwidth / 256 * 256;
            for (Uint31 x2 = 0; x2 < curwidth256; x2 += 256) {
                Uint31 x2lo = x2;
                float8_128 acchi = bitAs<float8_128>(broadcast(Calc::ident));
                float8_128 acclo = bitAs<float8_128>(broadcast(Calc::ident));
                for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                    Uint31 curvh = min(curheight - y2, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    bool8_128 curhmsk =
                        v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                    float8_128 rval =
                        load8_128_stride_ldmk((y2 * max_width_s + x2 / 2) / 32, max_width / 256, vmem, ldmk);
                    int8_128 tmphi = v_u32_and(*(int8_128 *)(&rval), v_u32_move_i(0xffff0000));
                    int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                    float8_128 rvalhi = *(float8_128 *)(&tmphi);
                    float8_128 rvallo = *(float8_128 *)(&tmplo);
                    if /* constexpr */ (ReduceHasMap<Calc>) {
                        rvalhi = Calc::map(rvalhi);
                        rvallo = Calc::map(rvallo);
                    }
                    float8_128 valhi = v_f32_sel(curhmsk, bitAs<float8_128>(broadcast(Calc::ident)), rvalhi);
                    float8_128 vallo = v_f32_sel(curhmsk, bitAs<float8_128>(broadcast(Calc::ident)), rvallo);
                    acchi = Calc::combine(acchi, valhi);
                    acclo = Calc::combine(acclo, vallo);
                }
                acchi = reduce8(acchi, Calc::combine);
                acclo = reduce8(acclo, Calc::combine);
                float8_128 acc2 =
                    v_f32_sel(v_s32_cmp(LS, get_core_id(), v_u32_move_i(128)), v_row_rotate(acchi, 1), acclo);
                float8_128 oldacc2 = v_f32_ld_tnsr_st_msk((x + x2lo) / 32, vout, 1, 0b11);
                v_f32_st_tnsr_st_msk((x + x2lo) / 32, vout, 1, 0b11, Calc::combine(oldacc2, acc2));
            }
            if (curwidth256 != curwidth) {
                Uint31 x2 = curwidth256;
                Uint31 x2lo = x2;
                float8_128 acclo = v_u32_move_f(Calc::ident);
                for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                    Uint31 curvh = min(curheight - y2, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    bool8_128 curhmsk =
                        v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                    float8_128 rval =
                        load8_128_stride_ldmk((y2 * max_width_s + x2 / 2) / 32, max_width / 256, vmem, ldmk);
                    int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                    float8_128 rvallo = *(float8_128 *)(&tmplo);
                    if /* constexpr */ (ReduceHasMap<Calc>) {
                        rvallo = Calc::map(rvallo);
                    }
                    float8_128 vallo = v_f32_sel(curhmsk, v_u32_move_f(Calc::ident), rvallo);
                    acclo = Calc::combine(acclo, vallo);
                }
                acclo = reduce8(acclo, Calc::combine);
                float8_128 oldacclo = v_f32_ld_tnsr_st_msk((x + x2lo) / 32, vout, 1, 1);
                v_f32_st_tnsr_st_msk((x + x2lo) / 32, vout, 1, 1, Calc::combine(oldacclo, acclo));
            }
        }
    }
    if /* constexpr */ (ReduceHasProject<Calc>) {
        transform_vmem(vout, pR, Calc::project);
    }
    Uint31 bfolen = Rl / 2;
    f32ToBf16_h(vout, bfolen, 1, Rl);
    // set load bflen - 128 will case llvm make bad base address
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk(0, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out + Rs / 64, OUT_PLACE, bfolen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [P, pR] => [1, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_00011_hbm_spilt_R(CxxTensor in, CxxTensor out, Uint31 P, Uint31 R, CxxTensor vmem,
                                     Uint31 vmemlen) {

    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 pRlen = pR256 / 2;
    float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
    int8_128 idx = get_core_id();
    idx = v_u32_and(idx, v_u32_move_i(127)) | v_u32_shl(v_u32_shr(idx, v_u32_move_i(7)), v_u32_move_i(8));
    for (Uint31 p = 0; p < P; p += 1) {
        for (Uint31 b = 0; b < pRlen; b += vmemlen) {
            Uint31 curb = min(pRlen - b, vmemlen);
            dlc_sync(dlc_dma(in + (p * pRlen + b) / 32, IN_PLACE, vmem, VMEM, curb, 128, 128, 128, 7));
            Uint31 curelem = min(R - b * 2, curb * 2);
            for (Uint31 r = 0; r < curelem; r += 2048) {
                Uint31 curr = min(curelem - r, 2048);
                float8_128 rval = v_f32_ld_tnsr_b(r / 64, vmem);
                int8_128 tmphi = v_u32_and(*(int8_128 *)(&rval), v_u32_move_i(0xffff0000));
                int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                float8_128 rvallo = *(float8_128 *)(&tmplo);
                float8_128 rvalhi = *(float8_128 *)(&tmphi);
                bool8_128 masklo = v_s32_cmp(LS, idx, broadcast(curr.sval));
                bool8_128 maskhi = v_s32_cmp(LS, v_s32_add(v_u32_move_i(128), idx), broadcast(curr.sval));
                if /* constexpr */ (ReduceHasMap<Calc>) {
                    rvallo = Calc::map(rvallo);
                    rvalhi = Calc::map(rvalhi);
                }
                float8_128 vallo = v_f32_sel(masklo, bitAs<float8_128>(broadcast(Calc::ident)), rvallo);
                float8_128 valhi = v_f32_sel(maskhi, bitAs<float8_128>(broadcast(Calc::ident)), rvalhi);
                acc = Calc::combine(acc, vallo);
                acc = Calc::combine(acc, valhi);
            }
        }
    }
    acc = reduce8(acc, Calc::combine);
    acc = Calc::reduce_combine(acc);
    if /* constexpr */ (ReduceHasProject<Calc>) {
        acc = Calc::project(acc);
    }
    int8_128 accb = float_to_bfloat16(acc, acc);
    v_f32_st_tnsr_st_msk(0, vmem, 1, 1, bitAs<float8_128>(accb));
    dlc_sync(dlc_dma(vmem, VMEM, out, OUT_PLACE, 128, 128, 128, 128, 7));
}

// [P, pR] => [1, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_00011_hbm(CxxTensor in, CxxTensor out, Uint31 P, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 bs = soft_sdiv(vmemlen, pR256 / 2);
    if (bs == 0) {
        reduce_00011_hbm_spilt_R<Calc, IN_PLACE, OUT_PLACE>(in, out, P, R, vmem, vmemlen);
        return;
    }
    int8_128 premsk = v_u32_and(get_core_id(), v_u32_move_i(127));
    float8_128 aacc = bitAs<float8_128>(broadcast(Calc::ident));
    for (Uint31 b = 0; b < P; b += bs) {
        Uint31 curb = min(P - b, bs);
        int syncL = dlc_dma(in + b * pR256 / 64, IN_PLACE, vmem, VMEM, curb * pR256 / 2, 128, 128, 128, 7);
        dlc_sync(syncL);
        for (Uint31 y = 0; y < curb; y += 8) {
            Uint31 curvh = min(curb - y, 8);
            float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
            Uint31 ldmk = (1 << curvh) - 1;
            Uint31 R256 = pR / 256 * 256;
            for (Uint31 x = 0; x < R256; x += 256) {
                // Uint31 xlo = x;
                Uint31 xhi = x + 128;
                // lo no need filter padding
                Uint31 curvwhi = min(R - xhi, 128);
                bool8_128 curwmskhi = v_s32_cmp(LS, premsk, v_u32_move_i(curvwhi));
                float8_128 rval =
                    load8_128_stride_ldmk((y * pR256 / 2 + x / 2) / 32, pR256 / 256, vmem, ldmk);
                int8_128 tmphi = v_u32_and(*(int8_128 *)(&rval), v_u32_move_i(0xffff0000));
                int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                float8_128 rvallo = *(float8_128 *)(&tmplo);
                float8_128 rvalhi = *(float8_128 *)(&tmphi);
                if /* constexpr */ (ReduceHasMap<Calc>) {
                    rvallo = Calc::map(rvallo);
                    rvalhi = Calc::map(rvalhi);
                }
                acc = Calc::combine(acc, rvallo);
                float8_128 valhi = v_f32_sel(curwmskhi, bitAs<float8_128>(broadcast(Calc::ident)), rvalhi);
                acc = Calc::combine(acc, valhi);
            }
            if (R256 != pR) {
                Uint31 x = R256;
                Uint31 curvw = min(R - x, 128);
                bool8_128 curwmsk = v_s32_cmp(LS, premsk, v_u32_move_i(curvw));
                float8_128 rval =
                    load8_128_stride_ldmk((y * pR256 / 2 + x / 2) / 32, pR256 / 256, vmem, ldmk);
                int8_128 tmplo = v_u32_shl(*(int8_128 *)(&rval), v_u32_move_i(16));
                rval = *(float8_128 *)(&tmplo);
                if /* constexpr */ (ReduceHasMap<Calc>) {
                    rval = Calc::map(rval);
                }
                float8_128 val = v_f32_sel(curwmsk, bitAs<float8_128>(broadcast(Calc::ident)), rval);
                acc = Calc::combine(acc, val);
            }
            aacc = Calc::combine(aacc, acc);
        }
    }
    aacc = reduce8(aacc, Calc::combine);
    aacc = Calc::reduce_combine(aacc);
    if /* constexpr */ (ReduceHasProject<Calc>) {
        aacc = Calc::project(aacc);
    }
    int8_128 aaccb = float_to_bfloat16(aacc, aacc);
    // _UNUSED volatile float s = vstore_wait(*(float8_128 *)(&aaccb));
    v_f32_st_tnsr_st_msk(0, vmem, 1, 1, *(float8_128 *)(&aaccb));
    int syncS = dlc_dma(vmem, VMEM, out, OUT_PLACE, 128, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [K, P, pR] => [1, P, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_00101_hbm(CxxTensor in, CxxTensor out, Uint31 K, Uint31 P, Uint31 R, CxxTensor vmem,
                             Uint31 vmemlen) {
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = P * 128;
    Uint31 voutlen256 = P * 256;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;
    Uint31 PpR = P * pR;
    Uint31 PpR256 = P * pR256;
    Uint31 max_width = soft_sdiv(vmemlen, K);
    Uint31 max_height = K;
    max_width = max_width / 256 * 256;
    max_width = min(max_width, PpR256);
    if (max_width == 0) {
        max_width = 256;
        max_height = soft_sdiv(vmemlen, max_width);
    }
    memset_vmem(vout, voutlen, Calc::ident);
    for (Uint31 x = 0; x < PpR256; x += max_width) {
        for (Uint31 y = 0; y < K; y += max_height) {
            Uint31 curw = min(PpR256 - x, max_width);
            Uint31 curh = min(K - y, max_height);
            load_mat<IN_PLACE>(in, vmem, K, PpR, y, x, curh, curw, PpR256, max_width);
            for (Uint31 x2 = 0; x2 < curw; x2 += 128) {
                // pR and curw all multiply of 128, so each 128 always in a P;
                Uint31 pidx = soft_sdiv(x + x2, pR256);
                float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                bool8_128 curwmsk = v_s32_cmp(LS, v_u32_and(get_core_id(), v_u32_move_i(127)),
                                              v_u32_move_i(pidx * pR256 + R - x - x2));
                for (Uint31 y2 = 0; y2 < curh; y2 += 8) {
                    Uint31 curvh = min(curh - y2, 8);
                    Uint31 ldmk = (1 << curvh) - 1;
                    bool8_128 curhmsk =
                        v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                    float8_128 rval =
                        load8_128_stride_ldmk((y2 * max_width + x2) / 32, max_width / 128, vmem, ldmk);
                    if /* constexpr */ (ReduceHasMap<Calc>) {
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
    if /* constexpr */ (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    Uint31 bfolen = voutlen256 / 2;
    f32ToBf16_h(vout, bfolen, P, 128);
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((bfolen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, bfolen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [H, K, P, pR] => [1, K, 1, pR]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_01010_hbm(CxxTensor in, CxxTensor out, Uint31 H, Uint31 K, Uint31 P, Uint31 R,
                             CxxTensor vmem, Uint31 vmemlen) {
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = K * pR;
    Uint31 voutlen256 = K * pR256;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;

    Uint31 max_width = soft_sdiv(vmemlen, P);
    Uint31 max_height = P;
    max_width = max_width / 256 * 256;
    max_width = min(max_width, pR256);
    if (max_width == 0) {
        max_width = 256;
        max_height = soft_sdiv(vmemlen, max_width);
    }

    memset_vmem(vout, voutlen, Calc::ident);

    for (Uint31 h = 0; h < H; h += 1) {
        for (Uint31 k = 0; k < K; k += 1) {
            for (Uint31 x = 0; x < pR; x += max_width) {
                for (Uint31 y = 0; y < P; y += max_height) {
                    Uint31 curwidth = min(pR - x, max_width);
                    Uint31 curheight = min(P - y, max_height);
                    load_mat<IN_PLACE>(in + (h * K * P * pR256 + k * P * pR256) / 64, vmem, P, pR, y, x,
                                       curheight, curwidth, pR256, max_width);
                    for (Uint31 x2 = 0; x2 < curwidth; x2 += 128) {
                        float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                        for (Uint31 y2 = 0; y2 < curheight; y2 += 8) {
                            Uint31 curvh = min(curheight - y2, 8);
                            Uint31 ldmk = (1 << curvh) - 1;
                            bool8_128 curhmsk =
                                v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                            float8_128 rval = load8_128_stride_ldmk((y2 * max_width + x2) / 32,
                                                                    max_width / 128, vmem, ldmk);
                            if /* constexpr */ (ReduceHasMap<Calc>) {
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
    if /* constexpr */ (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    Uint31 bfolen = voutlen256 / 2;
    f32ToBf16_h(vout, bfolen, K, pR);
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((bfolen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, bfolen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [H, K, P, pR] => [1, K, 1, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_01011_hbm(CxxTensor in, CxxTensor out, Uint31 H, Uint31 K, Uint31 P, Uint31 R,
                             CxxTensor vmem, Uint31 vmemlen) {
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = K * 128;
    Uint31 voutlen256 = K * 256;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;

    Uint31 max_width = soft_sdiv(vmemlen, P);
    Uint31 max_height = P;
    max_width = max_width / 256 * 256;
    max_width = min(max_width, pR256);
    if (max_width == 0) {
        max_width = 256;
        max_height = soft_sdiv(vmemlen, max_width);
    }

    memset_vmem(vout, voutlen, Calc::ident);

    for (Uint31 h = 0; h < H; h += 1) {
        for (Uint31 k = 0; k < K; k += 1) {
            for (Uint31 x = 0; x < pR; x += max_width) {
                for (Uint31 y = 0; y < P; y += max_height) {
                    Uint31 curwidth = min(pR - x, max_width);
                    Uint31 curheight = min(P - y, max_height);
                    load_mat<IN_PLACE>(in + (h * K * P * pR256 + k * P * pR256) / 64, vmem, P, pR, y, x,
                                       curheight, curwidth, pR256, max_width);
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
                            if /* constexpr */ (ReduceHasMap<Calc>) {
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
    if /* constexpr */ (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    Uint31 bfolen = voutlen256 / 2;
    f32ToBf16_h(vout, bfolen, K, 128);
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((bfolen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, bfolen, 128, 128, 128, 7);
    dlc_sync(syncS);
}

// [T, H, K, P, pR] => [1, H, 1, P, 1]
template <class Calc, int IN_PLACE = HBM, int OUT_PLACE = HBM>
    /* requires */ ReduceCalc<Calc>
inline void reduce_10101_hbm(CxxTensor in, CxxTensor out, Uint31 T, Uint31 H, Uint31 K, Uint31 P, Uint31 R,
                             CxxTensor vmem, Uint31 vmemlen) {
    Uint31 pR256 = (R + 255) / 256 * 256;
    Uint31 pR = (R + 127) / 128 * 128;
    Uint31 voutlen = H * P * 128;
    Uint31 voutlen256 = H * P * 256;
    CxxTensor vout = vmem;
    vmem = vout + voutlen / 32;
    vmemlen -= voutlen;
    Uint31 PpR = P * pR;
    Uint31 PpR256 = P * pR256;
    Uint31 max_width = soft_sdiv(vmemlen, K);
    Uint31 max_height = K;
    max_width = max_width / 256 * 256;
    max_width = min(max_width, PpR256);
    if (max_width == 0) {
        max_width = 256;
        max_height = soft_sdiv(vmemlen, max_width);
    }
    memset_vmem(vout, voutlen, Calc::ident);

    for (Uint31 t = 0; t < T; ++t) {
        for (Uint31 h = 0; h < H; ++h) {
            for (Uint31 x = 0; x < PpR256; x += max_width) {
                for (Uint31 y = 0; y < K; y += max_height) {
                    Uint31 curw = min(PpR256 - x, max_width);
                    Uint31 curh = min(K - y, max_height);
                    load_mat<IN_PLACE>(in + (t * H * K * PpR256 + h * K * PpR256) / 64, vmem, K, PpR, y, x,
                                       curh, curw, PpR256, max_width);
                    for (Uint31 x2 = 0; x2 < curw; x2 += 128) {
                        // pR and curw all multiply of 128, so each 128 always in a P;
                        Uint31 pidx = soft_sdiv(x + x2, pR256);
                        float8_128 acc = bitAs<float8_128>(broadcast(Calc::ident));
                        bool8_128 curwmsk = v_s32_cmp(LS, v_u32_and(get_core_id(), v_u32_move_i(127)),
                                                      v_u32_move_i(pidx * pR256 + R - x - x2));
                        for (Uint31 y2 = 0; y2 < curh; y2 += 8) {
                            Uint31 curvh = min(curh - y2, 8);
                            Uint31 ldmk = (1 << curvh) - 1;
                            bool8_128 curhmsk =
                                v_s32_cmp(LS, v_u32_shr(get_core_id(), v_u32_move_i(7)), v_u32_move_i(curvh));
                            float8_128 rval = load8_128_stride_ldmk((y2 * max_width + x2) / 32,
                                                                    max_width / 128, vmem, ldmk);
                            if /* constexpr */ (ReduceHasMap<Calc>) {
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

    if /* constexpr */ (ReduceHasProject<Calc>) {
        transform_vmem(vout, voutlen, Calc::project);
    }
    Uint31 bfolen = voutlen256 / 2;
    f32ToBf16_h(vout, bfolen, H * P, 128);
    // _UNUSED volatile float s = vstore_wait(v_f32_ld_tnsr_st_msk((bfolen - 128) / 32, vout, 1, 1));
    int syncS = dlc_dma(vout, VMEM, out, OUT_PLACE, bfolen, 128, 128, 128, 7);
    dlc_sync(syncS);
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

inline float8_128 bf16packlo2(float8_128 outlo, float8_128 outhi) {
    int8_128 loi = bitAs<int8_128>(outlo);
    int8_128 hii = bitAs<int8_128>(outhi);
    int8_128 out = v_u32_and(loi, v_u32_move_i(0x0000ffff)) | v_u32_shl(hii, v_u32_move_i(16));
    return bitAs<float8_128>(out);
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
    // // Print(const_cast<char *>("s2: %d\n"), s2.sval);
    // // Print(const_cast<char *>("s1: %d\n"), s1.sval);
    Uint31 ps1 = (s1 + 127) & (-128);
    Uint31 ps1256 = (ps1) & (-256);
    Uint31 p256s1 = (s1 + 255) & (-256);
    for (Uint31 is2 = 0; is2 < s2; is2 += 1) {
        Uint31 i1 = 0;
        for (; i1 < ps1256; i1 += 256) {
            Uint31 curs2 = min(s1 - i1 - 128, 128);
            Uint31 i2 = i1 + 128;
            float128_128 r1 = load128_128cmem_h((is2 * s1 + i1) * 4, cmem, 128);
            float128_128 r2 = load128_128cmem_h((is2 * s1 + i2) * 4, cmem, curs2);
            float8_128 tr1 = sub_vector(m_transpose_128_128_nws(r1, 0), 0);
            float8_128 tr2 = sub_vector(m_transpose_128_128_nws(r2, 0), 0);
            float8_128 tr = bf16packlo2(tr1, tr2);
            // // Print(const_cast<char*>("%y\n"), tr);
            v_f32_st_tnsr_st_msk((is2 * p256s1 + i1) / 64, vmem, 1, 1, tr);
        }
        if (i1 < s1) {
            Uint31 curs1 = min(s1 - i1, 128);
            float128_128 r1 = load128_128cmem_h((is2 * s1 + i1) * 4, cmem, curs1);
            float8_128 tr1 = sub_vector(m_transpose_128_128_nws(r1, 0), 0);
            v_f32_st_tnsr_st_msk((is2 * p256s1 + i1) / 64, vmem, 1, 1, tr1);
        }
    }
    // _UNUSED volatile float x = vstore_wait(v_f32_ld_tnsr_st_msk(s2 * ps1 - 128 / 32, vmem, 1, 1));
    // _UNUSED volatile float y = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem, 1, 1));
    dlc_sync(dlc_dma(vmem, VMEM, hbmout, HBM, s2 * p256s1 / 2, 128, 128, 128, 7));
}

// d0 need reduce
template <class Calc>
    /* requires */ ReduceCalc<Calc>
inline void reduce_hbm_dimlist_2xys_sqeeuze(CxxTensor in, CxxTensor hbmout, Uint31 d0, Uint31 d1, Uint31 d2,
                                            Uint31 d3, Uint31 d4, Uint31 reduce_d0, Uint31 reduce_d1,
                                            Uint31 reduce_d2, Uint31 reduce_d3, Uint31 reduce_d4,
                                            CxxTensor vmem, Uint31 vmemlen, CxxTensor out) {
    Uint31 s2(0), s1(0);
    get_sqeeuze_size(d0, d1, d2, d3, d4, reduce_d0, reduce_d1, reduce_d2, reduce_d3, reduce_d4, s2, s1);
    Uint31 pd0 = (d0 + 255) / 256 * 256;
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
        Uint31 B0 = B / 4 * 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        reduce_00001_hbm<Calc, HBM, CMEM>(in + off * pd0 / 64, out + off * 256 / 64, len, d0, vmem, vmemlen);
    } else if (reducePat == 0b00011) {
        Uint31 B = d4 * d3 * d2;
        Uint31 B0 = B / 4 * 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00011_hbm<Calc, HBM, CMEM>(in + (off + i) * d1 * pd0 / 64, out + (off + i) * 256 / 64, d1,
                                              d0, vmem, vmemlen);
        }
    } else if (reducePat == 0b00101) {
        Uint31 B = d4 * d3;
        Uint31 B0 = B / 2;
        if (d1 % 2 != 0) {
            B0 = B0 / 2 * 2;
        }
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00101_hbm<Calc, HBM, CMEM>(in + (off + i) * d2 * d1 * pd0 / 64,
                                              out + (off + i) * d1 * 256 / 64, d2, d1, d0, vmem, vmemlen);
        }
    } else if (reducePat == 0b01011) {
        Uint31 B = d4;
        Uint31 B0 = B / 2;
        if (d2 % 2 != 0) {
            B0 = B0 / 2 * 2;
        }
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        Uint31 uil = d3 * d2 * d1 * pd0;
        Uint31 uol = d2 * 256;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_01011_hbm<Calc, HBM, CMEM>(in + (off + i) * uil / 64, out + (off + i) * uol / 64, d3, d2,
                                              d1, d0, vmem, vmemlen);
        }
    } else if (reducePat == 0b10101) {
        if (get_device_id() == 0) {
            reduce_10101_hbm<Calc, HBM, CMEM>(in, out, d4, d3, d2, d1, d0, vmem, vmemlen);
        }
    }
    // // Print(const_cast<char*>("START SQEEUZE\n"));
    sqeeuze_low_cmem_in_hbm_out(out, s2, s1, vmem, hbmout);
}

template <class Calc>
    /* requires */ ReduceCalc<Calc>
inline void reduce_hbm_dimlist_2xys(CxxTensor in, CxxTensor out, Uint31 d0, Uint31 d1, Uint31 d2, Uint31 d3,
                                    Uint31 d4, Uint31 reduce_d0, Uint31 reduce_d1, Uint31 reduce_d2,
                                    Uint31 reduce_d3, Uint31 reduce_d4, CxxTensor vmem, Uint31 vmemlen,
                                    Uint31 keepDim = 1, SIM_X86::DLCMem *info = nullptr) {
    Uint31 pd0 = (d0 + 255) / 256 * 256;

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

    // // Print("reducePat: %d\n", reducePat);

    if (reducePat == 0b00001) {
        Uint31 B = d4 * d3 * d2 * d1;
        Uint31 B0 = B / 4 * 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        reduce_00001_hbm<Calc>(in + off * pd0 / 64, out + off * 256 / 64, len, d0, vmem, vmemlen);
    } else if (reducePat == 0b00010) {
        Uint31 B = d4 * d3 * d2;
        if (B == 1) {
            Uint31 pD0256 = (d0 + 255) / 256 * 256;
            Uint31 R0 = ((pD0256 / 2) + 255) / 256 * 256;
            Uint31 R1 = (pD0256 - R0 + 255) / 256 * 256;
            Uint31 Rs = get_device_id() == 0 ? Uint31(0) : R0;
            Uint31 Rl = get_device_id() == 0 ? R0 : R1;
            if (Rl != 0) {
                reduce_00010_hbm_split_R<Calc>(in, out, d1, d0, Rs, Rl, vmem, vmemlen);
            }
        } else {
            Uint31 B0 = B / 2;
            Uint31 B1 = B - B0;
            Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
            Uint31 len = get_device_id() == 0 ? B0 : B1;
            for (Uint31 i = 0; i < len; ++i) {
                reduce_00010_hbm<Calc>(in + (off + i) * d1 * pd0 / 64, out + (off + i) * pd0 / 64, d1, d0,
                                       vmem, vmemlen);
            }
        }
    } else if (reducePat == 0b00011) {
        Uint31 B = d4 * d3 * d2;
        Uint31 B0 = B / 4 * 2;
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00011_hbm<Calc>(in + (off + i) * d1 * pd0 / 64, out + (off + i) * 256 / 64, d1, d0, vmem,
                                   vmemlen);
        }
    } else if (reducePat == 0b00101) {
        Uint31 B = d4 * d3;
        Uint31 B0 = B / 2;
        if (d1 % 2 != 0) {
            B0 = B0 / 2 * 2;
        }
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_00101_hbm<Calc>(in + (off + i) * d2 * d1 * pd0 / 64, out + (off + i) * d1 * 256 / 64, d2,
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
            reduce_01010_hbm<Calc>(in + (off + i) * uil / 64, out + (off + i) * uol / 64, d3, d2, d1, d0,
                                   vmem, vmemlen);
        }
    } else if (reducePat == 0b01011) {
        Uint31 B = d4;
        Uint31 B0 = B / 2;
        if (d2 % 2 != 0) {
            B0 = B0 / 2 * 2;
        }
        Uint31 B1 = B - B0;
        Uint31 off = get_device_id() == 0 ? Uint31(0) : B0;
        Uint31 len = get_device_id() == 0 ? B0 : B1;
        Uint31 uil = d3 * d2 * d1 * pd0;
        Uint31 uol = d2 * 256;
        for (Uint31 i = 0; i < len; ++i) {
            reduce_01011_hbm<Calc>(in + (off + i) * uil / 64, out + (off + i) * uol / 64, d3, d2, d1, d0,
                                   vmem, vmemlen);
        }
    } else if (reducePat == 0b10101) {
        if (get_device_id() == 0) {
            reduce_10101_hbm<Calc>(in, out, d4, d3, d2, d1, d0, vmem, vmemlen);
        }
    }
}
