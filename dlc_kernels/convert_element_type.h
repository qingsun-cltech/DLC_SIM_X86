#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "bf16.h"
#include "ldst.h"

#include "align.h"
#include "constval.h"

#include "libdevice.h"

inline int8_128 __dlc_half_as_int(short8_128 a) {
    int8_128 result0;
    asm volatile("{ V0@(pr0)       vr10 = mov.u32 %[input]; }" : : [input] "x"(a) : "vr10");
    asm volatile("{"
                 "V1@(pr0)	vr10 = mov.u32 vr10;"
                 "}"
                 :
                 :
                 : "vr10");

    asm volatile("{V0@(pr0)        %[res] = mov.u32 vr10;}" : [res] "=x"(result0) : : "vr10");

    return result0;
}

inline int8_128 __dlc_char_as_int(char8_128 a) {
    int8_128 result0;
    asm volatile("{ V0@(pr0)       vr10 = mov.u32 %[input]; }" : : [input] "x"(a) : "vr10");
    asm volatile("{"
                 "V1@(pr0)	vr10 = mov.u32 vr10;"
                 "}"
                 :
                 :
                 : "vr10");

    asm volatile("{V0@(pr0)        %[res] = mov.u32 vr10;}" : [res] "=x"(result0) : : "vr10");

    return result0;
}

inline void fp32Toi32(SIM_X86::tensor input, SIM_X86::tensor output, int VMEMsize) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648); // inf, -inf, nan, 1e38
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);

    int i = 0;
    for (; i + 1024 < VMEMsize; i += 1024) {
        float8_128 x = v_f32_ld_tnsr_b(i / 32, input);
        int8_128 y = __dlc_float2int_rz(x);
        float8_128 x_ = v_f32_abs(x);
        int8_128 x1 = *(int8_128 *)(&x_);
        int8_128 y1 = v_s32_sel(v_f32_cmp(GTEQ, x, v_2147483647), y, limit_value);
        int8_128 y2 = v_s32_sel(v_s32_cmp(LSEQ, x1, neg_inf_v), y1, limit_value);
        int8_128 y3 = v_s32_sel(v_s32_cmp(GTEQ, x1, inf_v), y2, limit_value);
        v_st_generic(i / 32, output, 1, 255, y3);
    }
    if (i < VMEMsize) {
        int len = VMEMsize - i;
        int ldst_vmask = pre_exp2(len / 128);
        float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, input, 1, ldst_vmask);
        int8_128 y = __dlc_float2int_rz(x);
        float8_128 x_ = v_f32_abs(x);
        int8_128 x1 = *(int8_128 *)(&x_);
        int8_128 y1 = v_s32_sel(v_f32_cmp(GTEQ, x, v_2147483647), y, limit_value);
        int8_128 y2 = v_s32_sel(v_s32_cmp(LSEQ, x1, neg_inf_v), y1, limit_value);
        int8_128 y3 = v_s32_sel(v_s32_cmp(GTEQ, x1, inf_v), y2, limit_value);
        v_st_generic(i / 32, output, 1, ldst_vmask, y3);
    }
}

inline void i32Tofp32(SIM_X86::tensor input, SIM_X86::tensor output, int VMEMsize) {
    int i = 0;
    for (; i < VMEMsize; i += 1024) {
        int8_128 x = v_i32_ld_tnsr(i / 32, input, 1, 255);
        float8_128 y = __dlc_int2float_rn(x);
        v_st_generic(i / 32, output, 1, 255, y);
    }
    if (i < VMEMsize) {
        int len = VMEMsize - i;
        int ldst_vmask = pre_exp2(len / 128);
        int8_128 x = v_i32_ld_tnsr(i / 32, input, 1, ldst_vmask);
        float8_128 y = __dlc_int2float_rn(x);
        v_st_generic(i / 32, output, 1, ldst_vmask, y);
    }
}

inline void _32To32(SIM_X86::tensor input, SIM_X86::tensor output, int VMEMsize, DLCType in_dtype, DLCType out_dtype) {
    for (int i = 0; i < VMEMsize; i += 1024) {
        int len = min(VMEMsize - i, 1024);
        int ldst_vmask = pre_exp2(len / 128);
        if ((in_dtype == dlc_fp32) && (out_dtype == dlc_int32)) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, input, 1, ldst_vmask);
            int8_128 y = __dlc_float2int_rz(x);
            v_st_generic(i / 32, output, 1, ldst_vmask, y);
        } else if ((in_dtype == dlc_int32) && (out_dtype == dlc_fp32)) {
            int8_128 x = v_i32_ld_tnsr(i / 32, input, 1, ldst_vmask);
            float8_128 y = __dlc_int2float_rn(x);
            v_st_generic(i / 32, output, 1, ldst_vmask, y);
        }
    }
}

inline void __fp32Tofp16_256(SIM_X86::tensor input, SIM_X86::tensor output, int _len, bool is_128) {
    int len = _len * 2;
    for (int i = 0, j = 0; i < len; i += 2048, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        if ((i + 2048) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
        }
        float8_128 x1 = load8_128_stride_with_ldmask(i / 32, 2, ldst_mask, input);
        float8_128 x2 = load8_128_stride_with_ldmask((i + 128) / 32, 2, ldst_mask2, input);

        short8_128 y1 = __dlc_float2half_rn(x1);
        short8_128 y2 = __dlc_float2half_rn(x2);

        int8_128 y3 = v_u32_shl(__dlc_half_as_int(y2), 16);
        int8_128 result = v_s32_add(__dlc_half_as_int(y1), y3);

        store8_128_stride_with_stmask(j / 32, 1, ldst_mask, output, __$F(result));
    }
}

inline void __f32Toi16_256(SIM_X86::tensor input, SIM_X86::tensor output, int _len, bool is_128) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    int8_128 zero_v = v_u32_move_i(0);
    int len = _len * 2;
    for (int i = 0, j = 0; i < len; i += 2048, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        if ((i + 2048) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
        }
        float8_128 x1 = load8_128_stride_with_ldmask(i / 32, 2, ldst_mask, input);
        float8_128 x2 = load8_128_stride_with_ldmask((i + 128) / 32, 2, ldst_mask2, input);
        // 大于inf 1e38
        int8_128 y1 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x1), inf_v), __dlc_float2int_rz(x1), zero_v);
        int8_128 y2 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x2), inf_v), __dlc_float2int_rz(x2), zero_v);
        v_st_generic(j / 32, output, 1, ldst_mask, int_to_int16(y2, y1));
    }
}

inline void f32To16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, DLCType out_dtype) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        if (out_dtype == dlc_fp16) {
            __fp32Tofp16_256(mem, dst, len, false);
        } else if (out_dtype == dlc_bf16) {
            __f32ToBf16_256(mem, dst, len, false);
        } else if (out_dtype == dlc_int16) {
            __f32Toi16_256(mem, dst, len, false);
        }
    } else {
        int bd0 = ALIGN256(d0);
        int h = len / (bd0 / 2);
        if (out_dtype == dlc_fp16) {
            for (int i = 0; i < h; i++) {
                __fp32Tofp16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32),
                                 bd0 / 2, true);
            }
        } else if (out_dtype == dlc_bf16) {
            for (int i = 0; i < h; i++) {
                __f32ToBf16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32),
                                bd0 / 2, true);
            }
        } else if (out_dtype == dlc_int16) {
            for (int i = 0; i < h; i++) {
                __f32Toi16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32),
                               bd0 / 2, true);
            }
        }
    }
}

inline void fp32Tofp16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __fp32Tofp16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __fp32Tofp16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                             true);
        }
    }
}

inline void fp32Tobf16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32ToBf16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __f32ToBf16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                            true);
        }
    }
}

inline void fp32Toint16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32Toi16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __f32Toi16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                           true);
        }
    }
}

inline void __i32ToBf16_256(SIM_X86::tensor input, SIM_X86::tensor output, int _len, bool is_128) {
    int len = _len * 2;
    for (int i = 0, j = 0; i < len; i += 2048, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        if ((i + 2048) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
        }
        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 2, ldst_mask, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 2, ldst_mask2, input);
        store8_128_stride_with_stmask(
            j / 32, 1, ldst_mask, output,
            __$F(float_to_bfloat16(__dlc_int2float_rn(x2), __dlc_int2float_rn(x1))));
    }
}

inline void __i32Tofp16_256(SIM_X86::tensor input, SIM_X86::tensor output, int _len, bool is_128) {
    int len = _len * 2;
    for (int i = 0, j = 0; i < len; i += 2048, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        if ((i + 2048) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
        }
        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 2, ldst_mask, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 2, ldst_mask2, input);
        short8_128 y1 = __dlc_float2half_rn(__dlc_int2float_rn(x1));
        short8_128 y2 = __dlc_float2half_rn(__dlc_int2float_rn(x2));
        int8_128 y3 = v_u32_shl(__dlc_half_as_int(y2), 16);
        int8_128 result = v_s32_add(__dlc_half_as_int(y1), y3);
        store8_128_stride_with_stmask(j / 32, 1, ldst_mask, output, __$F(result));
    }
}

inline void __i32Toi16_256(SIM_X86::tensor input, SIM_X86::tensor output, int _len, bool is_128) {
    int len = _len * 2;
    for (int i = 0, j = 0; i < len; i += 2048, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        if ((i + 2048) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
        }
        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 2, ldst_mask, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 2, ldst_mask2, input);
        v_st_generic(j / 32, output, 1, ldst_mask, int_to_int16(x2, x1));
    }
}

inline void i32Tofp16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i32Tofp16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __i32Tofp16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                            true);
        }
    }
}

inline void i32Tobf16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i32ToBf16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __i32ToBf16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                            true);
        }
    }
}

inline void i32Toi16(SIM_X86::tensor mem, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i32Toi16_256(mem, mem, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __i32Toi16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), bd0 / 2,
                           true);
        }
    }
}

inline void __f32Toi8(SIM_X86::tensor input, SIM_X86::tensor output, int _len, int dim0, bool is_128) {
    int len = _len * 4;
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    int8_128 zero_v = v_u32_move_i(0);
    for (int i = 0, j = 0; i < len; i += 4096, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        int ldst_mask3 = pre_exp2(l / 128);
        int ldst_mask4 = pre_exp2(l / 128);
        if (((dim0 - i * 1024) == 384) && is_128) {
            ldst_mask4 = pre_exp2(l / 128 - 1);
        } else if (((dim0 - i * 1024) == 256) >= len && is_128) {
            ldst_mask3 = pre_exp2(l / 128 - 1);
            ldst_mask4 = pre_exp2(l / 128 - 1);
        } else if (((dim0 - i * 1024) == 128) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
            ldst_mask3 = pre_exp2(l / 128 - 1);
            ldst_mask4 = pre_exp2(l / 128 - 1);
        }

        float8_128 x1 = load8_128_stride_with_ldmask(i / 32, 4, ldst_mask, input);
        float8_128 x2 = load8_128_stride_with_ldmask((i + 128) / 32, 4, ldst_mask2, input);
        float8_128 x3 = load8_128_stride_with_ldmask((i + 256) / 32, 4, ldst_mask3, input);
        float8_128 x4 = load8_128_stride_with_ldmask((i + 384) / 32, 4, ldst_mask4, input);

        int8_128 z1 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x1), inf_v), __dlc_float2int_rz(x1), zero_v);
        int8_128 z2 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x2), inf_v), __dlc_float2int_rz(x2), zero_v);
        int8_128 z3 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x3), inf_v), __dlc_float2int_rz(x3), zero_v);
        int8_128 z4 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x4), inf_v), __dlc_float2int_rz(x4), zero_v);

        int8_128 x = v_u32_and(z1, 0x80000000);
        int8_128 y1 = v_u32_and(z1, 0xff);
        x = v_u32_and(z2, 0x80000000);
        int8_128 y2 = v_u32_and(z2, 0xff);
        y2 = v_u32_shl(y2, 8) | v_u32_shl(x, 8);
        x = v_u32_and(z3, 0x80000000);
        int8_128 y3 = v_u32_and(z3, 0xff);
        y3 = v_u32_shl(y3, 16) | v_u32_shl(x, 16);
        x = v_u32_and(z4, 0x80000000);
        int8_128 y4 = v_u32_and(z4, 0xff);
        y4 = v_u32_shl(y4, 24) | v_u32_shl(x, 24);
        int8_128 result = v_s32_add(v_s32_add(v_s32_add(y4, y3), y2), y1);

        v_st_generic(j / 32, output, 1, ldst_mask, result);
    }
}

inline void __f32Touint8(SIM_X86::tensor input, SIM_X86::tensor output, int _len, int dim0, bool is_128) {
    int len = _len * 4;
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    int8_128 zero_v = v_u32_move_i(0);
    for (int i = 0, j = 0; i < len; i += 4096, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        int ldst_mask3 = pre_exp2(l / 128);
        int ldst_mask4 = pre_exp2(l / 128);
        if (((dim0 - i * 1024) == 384) && is_128) {
            ldst_mask4 = pre_exp2(l / 128 - 1);
        } else if (((dim0 - i * 1024) == 256) >= len && is_128) {
            ldst_mask3 = pre_exp2(l / 128 - 1);
            ldst_mask4 = pre_exp2(l / 128 - 1);
        } else if (((dim0 - i * 1024) == 128) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
            ldst_mask3 = pre_exp2(l / 128 - 1);
            ldst_mask4 = pre_exp2(l / 128 - 1);
        }

        float8_128 x1 = load8_128_stride_with_ldmask(i / 32, 4, ldst_mask, input);
        float8_128 x2 = load8_128_stride_with_ldmask((i + 128) / 32, 4, ldst_mask2, input);
        float8_128 x3 = load8_128_stride_with_ldmask((i + 256) / 32, 4, ldst_mask3, input);
        float8_128 x4 = load8_128_stride_with_ldmask((i + 384) / 32, 4, ldst_mask4, input);

        int8_128 y1 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x1), inf_v), __dlc_float2int_rz(x1), zero_v);
        int8_128 y2 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x2), inf_v), __dlc_float2int_rz(x2), zero_v);
        int8_128 y3 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x3), inf_v), __dlc_float2int_rz(x3), zero_v);
        int8_128 y4 =
            v_s32_sel(v_s32_cmp(GTEQ, __dlc_float2int_rz(x4), inf_v), __dlc_float2int_rz(x4), zero_v);

        y1 = v_u32_and(y1, 0xff);
        y2 = v_u32_and(y2, 0xff);
        y3 = v_u32_and(y3, 0xff);
        y4 = v_u32_and(y4, 0xff);

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(j / 32, output, 1, ldst_mask, result);
    }
}

inline void __f32Tobool(SIM_X86::tensor input, SIM_X86::tensor output, int _len, int dim0, bool is_128) {
    int len = _len * 4;
    for (int i = 0, j = 0; i < len; i += 4096, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        int ldst_mask3 = pre_exp2(l / 128);
        int ldst_mask4 = pre_exp2(l / 128);
        if (((dim0 - i * 1024) == 384) && is_128) {
            ldst_mask4 = pre_exp2(l / 128 - 1);
        } else if (((dim0 - i * 1024) == 256) >= len && is_128) {
            ldst_mask3 = pre_exp2(l / 128 - 1);
            ldst_mask4 = pre_exp2(l / 128 - 1);
        } else if (((dim0 - i * 1024) == 128) >= len && is_128) {
            ldst_mask2 = pre_exp2(l / 128 - 1);
            ldst_mask3 = pre_exp2(l / 128 - 1);
            ldst_mask4 = pre_exp2(l / 128 - 1);
        }

        float8_128 x1 = load8_128_stride_with_ldmask(i / 32, 4, ldst_mask, input);
        float8_128 x2 = load8_128_stride_with_ldmask((i + 128) / 32, 4, ldst_mask2, input);
        float8_128 x3 = load8_128_stride_with_ldmask((i + 256) / 32, 4, ldst_mask3, input);
        float8_128 x4 = load8_128_stride_with_ldmask((i + 384) / 32, 4, ldst_mask4, input);

        int8_128 y1 = v_s32_sel(v_f32_cmp(EQ, x1, v_u32_move_f(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y2 = v_s32_sel(v_f32_cmp(EQ, x2, v_u32_move_f(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y3 = v_s32_sel(v_f32_cmp(EQ, x3, v_u32_move_f(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y4 = v_s32_sel(v_f32_cmp(EQ, x4, v_u32_move_f(0)), v_u32_move_i(1), v_u32_move_i(0));

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(j / 32, output, 1, ldst_mask, result);
    }
}

inline void __i32Toi8(SIM_X86::tensor input, SIM_X86::tensor output, int _len, int dim0, bool is_128) {
    int len1024 = _len & 0xffffffc00;
    int len128 = _len & 0x3ff;
    int i, j;
    for (i = 0, j = 0; j < len1024; i += 4096, j += 1024) {

        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 4, 255, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 4, 255, input);
        int8_128 x3 = load8_128_stride_with_ldmask_i((i + 256) / 32, 4, 255, input);
        int8_128 x4 = load8_128_stride_with_ldmask_i((i + 384) / 32, 4, 255, input);

        int8_128 x = v_u32_and(x1, 0x80000000);
        int8_128 y1 = v_u32_and(x1, 0xff);
        x = v_u32_and(x2, 0x80000000);
        int8_128 y2 = v_u32_and(x2, 0xff);
        y2 = v_u32_shl(y2, 8) | v_u32_shl(x, 8);
        x = v_u32_and(x3, 0x80000000);
        int8_128 y3 = v_u32_and(x3, 0xff);
        y3 = v_u32_shl(y3, 16) | v_u32_shl(x, 16);
        x = v_u32_and(x4, 0x80000000);
        int8_128 y4 = v_u32_and(x4, 0xff);
        y4 = v_u32_shl(y4, 24) | v_u32_shl(x, 24);
        x = v_u32_and(x1, 0x80000000);

        int8_128 result = v_s32_add(v_s32_add(v_s32_add(y4, y3), y2), y1);

        v_st_generic(j / 32, output, 1, 255, result);
    }
    if (len128) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2(len128 / 128);
        int ldst_mask3 = pre_exp2(len128 / 128);
        int ldst_mask4 = pre_exp2(len128 / 128);
        if (((dim0 % 512) == 384) && is_128) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if (((dim0 % 512) == 256) && is_128) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if (((dim0 % 512) == 128) && is_128) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }

        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 4, ldst_mask, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 4, ldst_mask2, input);
        int8_128 x3 = load8_128_stride_with_ldmask_i((i + 256) / 32, 4, ldst_mask3, input);
        int8_128 x4 = load8_128_stride_with_ldmask_i((i + 384) / 32, 4, ldst_mask4, input);

        int8_128 x = v_u32_and(x1, 0x80000000);
        int8_128 y1 = v_u32_and(x1, 0xff);
        x = v_u32_and(x2, 0x80000000);
        int8_128 y2 = v_u32_and(x2, 0xff);
        y2 = v_u32_shl(y2, 8) | v_u32_shl(x, 8);
        x = v_u32_and(x3, 0x80000000);
        int8_128 y3 = v_u32_and(x3, 0xff);
        y3 = v_u32_shl(y3, 16) | v_u32_shl(x, 16);
        x = v_u32_and(x4, 0x80000000);
        int8_128 y4 = v_u32_and(x4, 0xff);
        y4 = v_u32_shl(y4, 24) | v_u32_shl(x, 24);
        x = v_u32_and(x1, 0x80000000);

        int8_128 result = v_s32_add(v_s32_add(v_s32_add(y4, y3), y2), y1);
        v_st_generic(j / 32, output, 1, ldst_mask, result);
    }
}

inline void __i32Touint8(SIM_X86::tensor input, SIM_X86::tensor output, int _len, int dim0, bool is_128) {
    int len1024 = _len & 0xffffffc00;
    int len128 = _len & 0x3ff;
    int i, j;
    for (i = 0, j = 0; j < len1024; i += 4096, j += 1024) {
        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 4, 255, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 4, 255, input);
        int8_128 x3 = load8_128_stride_with_ldmask_i((i + 256) / 32, 4, 255, input);
        int8_128 x4 = load8_128_stride_with_ldmask_i((i + 384) / 32, 4, 255, input);

        int8_128 y1 = v_u32_and(x1, 0xff);
        int8_128 y2 = v_u32_and(x2, 0xff);
        int8_128 y3 = v_u32_and(x3, 0xff);
        int8_128 y4 = v_u32_and(x4, 0xff);

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(j / 32, output, 1, 255, result);
    }
    if (len128) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2(len128 / 128);
        int ldst_mask3 = pre_exp2(len128 / 128);
        int ldst_mask4 = pre_exp2(len128 / 128);
        if (((dim0 % 512) == 384) && is_128) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if (((dim0 % 512) == 256) && is_128) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if (((dim0 % 512) == 128) && is_128) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }
        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 4, ldst_mask, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 4, ldst_mask2, input);
        int8_128 x3 = load8_128_stride_with_ldmask_i((i + 256) / 32, 4, ldst_mask3, input);
        int8_128 x4 = load8_128_stride_with_ldmask_i((i + 384) / 32, 4, ldst_mask4, input);

        int8_128 y1 = v_u32_and(x1, 0xff);
        int8_128 y2 = v_u32_and(x2, 0xff);
        int8_128 y3 = v_u32_and(x3, 0xff);
        int8_128 y4 = v_u32_and(x4, 0xff);

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(j / 32, output, 1, ldst_mask, result);
    }
}

inline void __i32Tobool(SIM_X86::tensor input, SIM_X86::tensor output, int _len, int dim0, bool is_128) {
    int len1024 = _len & 0xffffffc00;
    int len128 = _len & 0x3ff;
    int i, j;
    for (i = 0, j = 0; j < len1024; i += 4096, j += 1024) {
        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 4, 255, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 4, 255, input);
        int8_128 x3 = load8_128_stride_with_ldmask_i((i + 256) / 32, 4, 255, input);
        int8_128 x4 = load8_128_stride_with_ldmask_i((i + 384) / 32, 4, 255, input);

        int8_128 y1 = v_s32_sel(v_s32_cmp(EQ, x1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y2 = v_s32_sel(v_s32_cmp(EQ, x2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y3 = v_s32_sel(v_s32_cmp(EQ, x3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y4 = v_s32_sel(v_s32_cmp(EQ, x4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

        y1 = v_u32_and(y1, 0xff);
        y2 = v_u32_and(y2, 0xff);
        y3 = v_u32_and(y3, 0xff);
        y4 = v_u32_and(y4, 0xff);

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(j / 32, output, 1, 255, result);
    }
    if (len128) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2(len128 / 128);
        int ldst_mask3 = pre_exp2(len128 / 128);
        int ldst_mask4 = pre_exp2(len128 / 128);
        if (((dim0 % 512) == 384) && is_128) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if (((dim0 % 512) == 256) && is_128) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if (((dim0 % 512) == 128) && is_128) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }
        int8_128 x1 = load8_128_stride_with_ldmask_i(i / 32, 4, ldst_mask, input);
        int8_128 x2 = load8_128_stride_with_ldmask_i((i + 128) / 32, 4, ldst_mask2, input);
        int8_128 x3 = load8_128_stride_with_ldmask_i((i + 256) / 32, 4, ldst_mask3, input);
        int8_128 x4 = load8_128_stride_with_ldmask_i((i + 384) / 32, 4, ldst_mask4, input);

        int8_128 y1 = v_s32_sel(v_s32_cmp(EQ, x1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y2 = v_s32_sel(v_s32_cmp(EQ, x2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y3 = v_s32_sel(v_s32_cmp(EQ, x3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        int8_128 y4 = v_s32_sel(v_s32_cmp(EQ, x4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

        y1 = v_u32_and(y1, 0xff);
        y2 = v_u32_and(y2, 0xff);
        y3 = v_u32_and(y3, 0xff);
        y4 = v_u32_and(y4, 0xff);

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(j / 32, output, 1, ldst_mask, result);
    }
}

inline void _fp32To8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, DLCType out_dtype) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        if (out_dtype == dlc_int8) {
            __f32Toi8(mem, dst, len, d0, false);
        } else if (out_dtype == dlc_bool) {
            __f32Tobool(mem, dst, len, d0, false);
        } else {
            __f32Touint8(mem, dst, len, d0, false);
        }
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        if (out_dtype == dlc_int8) {
            for (int i = 0; i < h; i++) {
                __f32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4,
                          d0, true);
            }
        } else if (out_dtype == dlc_bool) {
            for (int i = 0; i < h; i++) {
                __f32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4,
                            d0, true);
            }
        } else {
            for (int i = 0; i < h; i++) {
                __f32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4,
                             d0, true);
            }
        }
    }
}

inline void fp32Toi8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Toi8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void fp32Tobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Tobool(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                        true);
        }
    }
}

inline void fp32Touint8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Touint8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void i32Touint8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Touint8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void i32Toi8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Toi8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void i32Tobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Tobool(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                        true);
        }
    }
}

inline void __f16ToF32_256(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            store8_128_stride(j / 32, 2, dst, __dlc_half2float(x1));
            store8_128_stride((j + 128) / 32, 2, dst, __dlc_half2float(x2));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        store8_128_stride_with_stmask(0, 2, ldst_mask, dst, __dlc_half2float(x1));
        store8_128_stride_with_stmask(4, 2, ldst_mask, dst, __dlc_half2float(x2));
    }
}

inline void __f16ToF32_128(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_b(i / 32, mem);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            store8_128_stride(j / 32, 2, dst, __dlc_half2float(x1));
            if (i == (len - 1024)) {
                store8_128_stride_stmk((j + 128) / 32, 2, dst, __dlc_half2float(x2), 127);
            } else {
                store8_128_stride((j + 128) / 32, 2, dst, __dlc_half2float(x2));
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        store8_128_stride_with_stmask(0, 2, ldst_mask, dst, __dlc_half2float(x1));
        store8_128_stride_with_stmask(4, 2, ldst_mask2, dst, __dlc_half2float(x2));
    }
}

inline void f16ToF32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __f16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
}

inline void __f16Toi32_256(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);

            float8_128 y1 = __dlc_half2float(x1);
            float8_128 y2 = __dlc_half2float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

            store8_128_stride_i(j / 32, 2, dst, w1);
            store8_128_stride_i((j + 128) / 32, 2, dst, w2);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = __dlc_half2float(x1);
        float8_128 y2 = __dlc_half2float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask, dst, w2);
    }
}

inline void __f16Toi32_128(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_b(i / 32, mem);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            float8_128 y1 = __dlc_half2float(x1);
            float8_128 y2 = __dlc_half2float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);
            store8_128_stride_i(j / 32, 2, dst, w1);
            if (i == (len - 1024)) {
                store8_128_stride_stmk_i((j + 128) / 32, 2, dst, w2, 127);
            } else {
                store8_128_stride_i((j + 128) / 32, 2, dst, w2);
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = __dlc_half2float(x1);
        float8_128 y2 = __dlc_half2float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask2, dst, w2);
    }
}

inline void __f16Toi64_256(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int8_128 CheckSign1, CheckSign2;
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);

            float8_128 y1 = __dlc_half2float(x1);
            float8_128 y2 = __dlc_half2float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

            CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
            CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);
            store8_128_stride_i(j / 32, 2, dst, w1);
            store8_128_stride_i((j + 128) / 32, 2, dst, w2);
            store8_128_stride_with_stmask_cmem((j) / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign1));
            store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign2));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = __dlc_half2float(x1);
        float8_128 y2 = __dlc_half2float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);
        CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
        CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask, dst, w2);
        store8_128_stride_with_stmask_cmem(0, 2, ldst_mask, cmem, *(float8_128 *)(&CheckSign1));
        store8_128_stride_with_stmask_cmem(4, 2, ldst_mask, cmem, *(float8_128 *)(&CheckSign2));
    }
}

inline void __f16Toi64_128(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int8_128 CheckSign1, CheckSign2;

    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_b(i / 32, mem);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            float8_128 y1 = __dlc_half2float(x1);
            float8_128 y2 = __dlc_half2float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

            CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
            CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);
            store8_128_stride_i(j / 32, 2, dst, w1);
            store8_128_stride_with_stmask_cmem(j / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign1));
            if (i == (len - 1024)) {
                store8_128_stride_stmk_i((j + 128) / 32, 2, dst, w2, 127);
                store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 127, cmem, *(float8_128 *)(&CheckSign2));
            } else {
                store8_128_stride_i((j + 128) / 32, 2, dst, w2);
                store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign2));
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = __dlc_half2float(x1);
        float8_128 y2 = __dlc_half2float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
        CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask2, dst, w2);
        store8_128_stride_with_stmask_cmem(0, 2, ldst_mask, cmem, *(float8_128 *)(&CheckSign1));
        store8_128_stride_with_stmask_cmem(4, 2, ldst_mask2, cmem, *(float8_128 *)(&CheckSign2));
    }
}

inline void f16Toi32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __f16Toi32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
}

inline void f16Tobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __f16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    d0 = ALIGN128(d0);
    d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Tobool(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                        true);
        }
    }
}

inline void f16Tobf16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024; i >= 0; i -= 1024) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            store8_128_stride_i(i / 32, 1, dst,
                                float_to_bfloat16(__dlc_half2float(x2), __dlc_half2float(x1)));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        store8_128_stride_with_stmask_i(0, 1, ldst_mask, dst,
                                        float_to_bfloat16(__dlc_half2float(x2), __dlc_half2float(x1)));
    }
}

inline void f16Toi16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024; i >= 0; i -= 1024) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);

            float8_128 y1 = __dlc_half2float(x1);
            float8_128 y2 = __dlc_half2float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

            store8_128_stride_i(i / 32, 1, dst, int_to_int16(w2, w1));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = __dlc_half2float(x1);
        float8_128 y2 = __dlc_half2float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        store8_128_stride_with_stmask_i(0, 1, ldst_mask, dst, int_to_int16(w2, w1));
    }
}

inline void f16Toi8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __f16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    if (d0k128 % 4 == 0) {
        __f32Toi8(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void f16Touint8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __f16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    if (d0k128 % 4 == 0) {
        __f32Touint8(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void f16ToAnother(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, DLCType out_dtype) {
    if (out_dtype == dlc_fp32) {
        f16ToF32(mem, dst, len, d0);
    } else if (out_dtype == dlc_int32) {
        f16Toi32(mem, dst, len, d0);
    } else if (out_dtype == dlc_bf16) {
        f16Tobf16(mem, dst, len);
    } else if (out_dtype == dlc_int16) {
        f16Toi16(mem, dst, len);
    } else if (out_dtype == dlc_int8) {
        f16Toi8(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_bool) {
        f16Tobool(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_uint8) {
        f16Touint8(mem, dst, len, len_, d0);
    }
}

inline void __bf16Toi32_256(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            float8_128 y1 = bfloat16_to_float(x1);
            float8_128 y2 = bfloat16_to_float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

            store8_128_stride_i(j / 32, 2, dst, w1);
            store8_128_stride_i((j + 128) / 32, 2, dst, w2);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = bfloat16_to_float(x1);
        float8_128 y2 = bfloat16_to_float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask, dst, w2);
    }
}

inline void __bf16Toi32_128(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_b(i / 32, mem);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            float8_128 y1 = bfloat16_to_float(x1);
            float8_128 y2 = bfloat16_to_float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

            store8_128_stride_i(j / 32, 2, dst, w1);
            if (i == (len - 1024)) {
                store8_128_stride_stmk_i((j + 128) / 32, 2, dst, w2, 127);
            } else {
                store8_128_stride_i((j + 128) / 32, 2, dst, w2);
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = bfloat16_to_float(x1);
        float8_128 y2 = bfloat16_to_float(x2);
        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask2, dst, w2);
    }
}

inline void __bf16Toi64_256(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int8_128 CheckSign1, CheckSign2;
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            float8_128 y1 = bfloat16_to_float(x1);
            float8_128 y2 = bfloat16_to_float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

            CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
            CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);
            store8_128_stride_i(j / 32, 2, dst, w1);
            store8_128_stride_i((j + 128) / 32, 2, dst, w2);
            store8_128_stride_with_stmask_cmem(j / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign1));
            store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign2));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = bfloat16_to_float(x1);
        float8_128 y2 = bfloat16_to_float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
        CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask, dst, w2);
        store8_128_stride_with_stmask_cmem(0, 2, ldst_mask, cmem, *(float8_128 *)(&CheckSign1));
        store8_128_stride_with_stmask_cmem(4, 2, ldst_mask, cmem, *(float8_128 *)(&CheckSign2));
    }
}

inline void __bf16Toi64_128(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int8_128 CheckSign1, CheckSign2;
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_b(i / 32, mem);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            float8_128 y1 = bfloat16_to_float(x1);
            float8_128 y2 = bfloat16_to_float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);
            CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
            CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);

            store8_128_stride_i(j / 32, 2, dst, w1);
            store8_128_stride_with_stmask_cmem(j / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign1));
            if (i == (len - 1024)) {
                store8_128_stride_stmk_i((j + 128) / 32, 2, dst, w2, 127);
                store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 127, cmem, *(float8_128 *)(&CheckSign2));
            } else {
                store8_128_stride_i((j + 128) / 32, 2, dst, w2);
                store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 255, cmem, *(float8_128 *)(&CheckSign2));
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = bfloat16_to_float(x1);
        float8_128 y2 = bfloat16_to_float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);

        CheckSign1 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y1), 0), 0, 0xffffffff);
        CheckSign2 = v_s32_sel(v_s32_cmp(LS, __dlc_float2int_rz(y2), 0), 0, 0xffffffff);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, w1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask2, dst, w2);

        store8_128_stride_with_stmask_cmem(0, 2, ldst_mask, cmem, *(float8_128 *)(&CheckSign1));
        store8_128_stride_with_stmask_cmem(4, 2, ldst_mask2, cmem, *(float8_128 *)(&CheckSign2));
    }
}

inline void bf16Toi32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __bf16Toi32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
}

inline void bf16Tobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __bf16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    d0 = ALIGN128(d0);
    d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Tobool(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                        true);
        }
    }
}

inline void bf16Tof16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024; i >= 0; i -= 1024) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            int8_128 result = pack_16b(__dlc_float2half_rn(bfloat16_to_float(x2)),
                                       __dlc_float2half_rn(bfloat16_to_float(x1)));
            store8_128_stride_i(i / 32, 1, dst, result);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        int8_128 result =
            pack_16b(__dlc_float2half_rn(bfloat16_to_float(x2)), __dlc_float2half_rn(bfloat16_to_float(x1)));
        store8_128_stride_with_stmask_i(0, 1, ldst_mask, dst, result);
    }
}

inline void bf16Toi16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648);
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024; i >= 0; i -= 1024) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            float8_128 y1 = bfloat16_to_float(x1);
            float8_128 y2 = bfloat16_to_float(x2);

            float8_128 y1_ = v_f32_abs(y1);
            float8_128 y2_ = v_f32_abs(y2);
            int8_128 z1 = *(int8_128 *)(&y1_);
            int8_128 z2 = *(int8_128 *)(&y2_);

            int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
            int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

            w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

            w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
            w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);
            store8_128_stride_i(i / 32, 1, dst, int_to_int16(w2, w1));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        float8_128 y1 = bfloat16_to_float(x1);
        float8_128 y2 = bfloat16_to_float(x2);

        float8_128 y1_ = v_f32_abs(y1);
        float8_128 y2_ = v_f32_abs(y2);
        int8_128 z1 = *(int8_128 *)(&y1_);
        int8_128 z2 = *(int8_128 *)(&y2_);

        int8_128 w1 = v_s32_sel(v_f32_cmp(GTEQ, y1, v_2147483647), __dlc_float2int_rz(y1), limit_value);
        int8_128 w2 = v_s32_sel(v_f32_cmp(GTEQ, y2, v_2147483647), __dlc_float2int_rz(y2), limit_value);

        w1 = v_s32_sel(v_s32_cmp(LSEQ, z1, neg_inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(LSEQ, z2, neg_inf_v), w2, limit_value);

        w1 = v_s32_sel(v_s32_cmp(GTEQ, z1, inf_v), w1, limit_value);
        w2 = v_s32_sel(v_s32_cmp(GTEQ, z2, inf_v), w2, limit_value);
        store8_128_stride_with_stmask_i(0, 1, ldst_mask, dst, int_to_int16(w2, w1));
    }
}

inline void bf16Toi8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __bf16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    d0 = ALIGN128(d0);
    d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Toi8(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void bf16Touint8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __bf16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    d0 = ALIGN128(d0);
    d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Touint8(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __f32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void bf16ToAnother(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, DLCType out_dtype) {
    if (out_dtype == dlc_fp32) {
        bf16ToF32(mem, len, d0); // mem 和dst是一个
    } else if (out_dtype == dlc_int32) {
        bf16Toi32(mem, dst, len, d0);
    } else if (out_dtype == dlc_fp16) {
        bf16Tof16(mem, dst, len);
    } else if (out_dtype == dlc_int16) {
        bf16Toi16(mem, dst, len);
    } else if (out_dtype == dlc_int8) {
        bf16Toi8(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_bool) {
        bf16Tobool(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_uint8) {
        bf16Touint8(mem, dst, len, len_, d0);
    }
}

inline void __i16ToF32_256(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo

            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
            int8_128 _result =
                v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));

            store8_128_stride(j / 32, 2, dst, __dlc_int2float_rn(result));
            store8_128_stride((j + 128) / 32, 2, dst, __dlc_int2float_rn(_result));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, 255);

        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);

        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));
        store8_128_stride_with_stmask(0, 2, ldst_mask, dst, __dlc_int2float_rn(result));
        store8_128_stride_with_stmask(4, 2, ldst_mask, dst, __dlc_int2float_rn(_result));
    }
}

inline void __i16ToF32_128(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo

            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
            int8_128 _result =
                v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));
            store8_128_stride(j / 32, 2, dst, __dlc_int2float_rn(result));
            if (i == (len - 1024)) {
                store8_128_stride_stmk((j + 128) / 32, 2, dst, __dlc_int2float_rn(_result), 127);
            } else {
                store8_128_stride((j + 128) / 32, 2, dst, __dlc_int2float_rn(_result));
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, ldst_mask);
        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);
        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));
        store8_128_stride_with_stmask(0, 2, ldst_mask, dst, __dlc_int2float_rn(result));
        store8_128_stride_with_stmask(4, 2, ldst_mask2, dst, __dlc_int2float_rn(_result));
    }
}

inline void i16ToF32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __i16ToF32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
}

inline void __i16Toi32_256(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo

            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
            int8_128 _result =
                v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));
            store8_128_stride_i(j / 32, 2, dst, result);
            store8_128_stride_i((j + 128) / 32, 2, dst, _result);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, 255);

        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);

        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, result);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask, dst, _result);
    }
}

inline void __i16Toi32_128(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo

            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
            int8_128 _result =
                v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));
            store8_128_stride_i(j / 32, 2, dst, result);

            if (i == (len - 1024)) {
                store8_128_stride_stmk_i((j + 128) / 32, 2, dst, _result, 127);
            } else {
                store8_128_stride_i((j + 128) / 32, 2, dst, _result);
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, 255);

        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);

        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, result);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask2, dst, _result);
    }
}

inline void __i16Toi64_256(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    int8_128 Sign1, Sign2;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo

            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            bool8_128 CheckSign1 = v_s32_cmp(GT, y, v_u32_move_i(32767));
            bool8_128 CheckSign2 = v_s32_cmp(GT, y1, v_u32_move_i(32767));
            int8_128 result = v_s32_sel(CheckSign1, y, y - v_u32_move_i(65536));
            int8_128 _result = v_s32_sel(CheckSign2, y1, y1 - v_u32_move_i(65536));

            Sign1 = v_s32_sel(CheckSign1, 0, 0xffffffff);
            Sign2 = v_s32_sel(CheckSign2, 0, 0xffffffff);

            store8_128_stride_i(j / 32, 2, dst, result);
            store8_128_stride_i((j + 128) / 32, 2, dst, _result);
            store8_128_stride_with_stmask_cmem(j / 32, 2, 255, cmem, *(float8_128 *)(&Sign1));
            store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 255, cmem, *(float8_128 *)(&Sign2));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, 255);

        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);

        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        bool8_128 CheckSign1 = v_s32_cmp(GT, y, v_u32_move_i(32767));
        bool8_128 CheckSign2 = v_s32_cmp(GT, y1, v_u32_move_i(32767));
        int8_128 result = v_s32_sel(CheckSign1, y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(CheckSign2, y1, y1 - v_u32_move_i(65536));

        Sign1 = v_s32_sel(CheckSign1, 0, 0xffffffff);
        Sign2 = v_s32_sel(CheckSign2, 0, 0xffffffff);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, result);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask, dst, _result);
        store8_128_stride_with_stmask_cmem(0, 2, ldst_mask, cmem, *(float8_128 *)(&Sign1));
        store8_128_stride_with_stmask_cmem(4, 2, ldst_mask, cmem, *(float8_128 *)(&Sign2));
    }
}

inline void __i16Toi64_128(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    int8_128 Sign1, Sign2;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo

            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            bool8_128 CheckSign1 = v_s32_cmp(GT, y, v_u32_move_i(32767));
            bool8_128 CheckSign2 = v_s32_cmp(GT, y1, v_u32_move_i(32767));
            int8_128 result = v_s32_sel(CheckSign1, y, y - v_u32_move_i(65536));
            int8_128 _result = v_s32_sel(CheckSign2, y1, y1 - v_u32_move_i(65536));
            Sign1 = v_s32_sel(CheckSign1, 0, 0xffffffff);
            Sign2 = v_s32_sel(CheckSign2, 0, 0xffffffff);
            store8_128_stride_i(j / 32, 2, dst, result);
            store8_128_stride_with_stmask_cmem(j / 32, 2, 255, cmem, *(float8_128 *)(&Sign1));
            if (i == (len - 1024)) {
                store8_128_stride_stmk_i((j + 128) / 32, 2, dst, _result, 127);
                store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 127, cmem, *(float8_128 *)(&Sign2));
            } else {
                store8_128_stride_i((j + 128) / 32, 2, dst, _result);
                store8_128_stride_with_stmask_cmem((j + 128) / 32, 2, 255, cmem, *(float8_128 *)(&Sign2));
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, 255);

        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);

        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        bool8_128 CheckSign1 = v_s32_cmp(GT, y, v_u32_move_i(32767));
        bool8_128 CheckSign2 = v_s32_cmp(GT, y1, v_u32_move_i(32767));
        int8_128 result = v_s32_sel(CheckSign1, y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(CheckSign2, y1, y1 - v_u32_move_i(65536));

        Sign1 = v_s32_sel(CheckSign1, 0, 0xffffffff);
        Sign2 = v_s32_sel(CheckSign2, 0, 0xffffffff);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, result);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask2, dst, _result);

        store8_128_stride_with_stmask_cmem(0, 2, ldst_mask, cmem, *(float8_128 *)(&Sign1));
        store8_128_stride_with_stmask_cmem(4, 2, ldst_mask2, cmem, *(float8_128 *)(&Sign2));
    }
}

inline void i16Toi32(SIM_X86::tensor mem, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16Toi32_256(mem, mem, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __i16Toi32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)mem + i * d0 / 32), bd0 / 2);
        }
    }
}

inline void i16Tobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __i16Toi32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    d0 = ALIGN128(d0);
    d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Tobool(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                        true);
        }
    }
}

inline void i16Tof16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024; i >= 0; i -= 1024) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo

            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
            int8_128 _result =
                v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));

            short8_128 y2 = __dlc_float2half_rn(__dlc_int2float_rn(result));
            short8_128 y3 = __dlc_float2half_rn(__dlc_int2float_rn(_result));
            int8_128 reult1 = pack_16b(y3, y2);
            v_st_generic(i / 32, dst, 1, 255, reult1);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, 255);
        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);
        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));

        short8_128 y2 = __dlc_float2half_rn(__dlc_int2float_rn(result));
        short8_128 y3 = __dlc_float2half_rn(__dlc_int2float_rn(_result));
        int8_128 reult1 = pack_16b(y3, y2);
        v_st_generic(0, dst, 1, ldst_mask, reult1);
    }
}

inline void i16Tobf16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024; i >= 0; i -= 1024) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(x, 1); // hi
            short8_128 x1 = unpack_16b(x, 0); // lo
            int8_128 y = __dlc_half_as_int(x1);
            int8_128 y1 = __dlc_half_as_int(x2);
            int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
            int8_128 _result =
                v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));
            v_st_generic(i / 32, dst, 1, 255,
                         float_to_bfloat16(__dlc_int2float_rn(_result), __dlc_int2float_rn(result)));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, 255);
        short8_128 x2 = unpack_16b(x, 1);
        short8_128 x1 = unpack_16b(x, 0);
        int8_128 y = __dlc_half_as_int(x1);
        int8_128 y1 = __dlc_half_as_int(x2);
        int8_128 result = v_s32_sel(v_s32_cmp(GT, y, v_u32_move_i(32767)), y, y - v_u32_move_i(65536));
        int8_128 _result = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(32767)), y1, y1 - v_u32_move_i(65536));
        v_st_generic(0, dst, 1, ldst_mask,
                     float_to_bfloat16(__dlc_int2float_rn(_result), __dlc_int2float_rn(result)));
    }
}

inline void i16Toi8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __i16Toi32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    d0 = ALIGN128(d0);
    d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Toi8(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void i16Touint8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __i16Toi32_128((SIM_X86::tensor )((int)mem + i * bd0 / 2 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 2);
        }
    }
    d0 = ALIGN128(d0);
    d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Touint8(mem, dst, len_, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = len_ / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void i16ToAnother(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, DLCType out_dtype) {
    if (out_dtype == dlc_fp32) {
        i16ToF32(mem, dst, len, d0);
    } else if (out_dtype == dlc_int32) {
        i16Toi32(mem, len, d0);
    } else if (out_dtype == dlc_fp16) {
        i16Tof16(mem, dst, len);
    } else if (out_dtype == dlc_bf16) {
        i16Tobf16(mem, dst, len);
    } else if (out_dtype == dlc_int8) {
        i16Toi8(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_bool) {
        i16Tobool(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_uint8) {
        i16Touint8(mem, dst, len, len_, d0);
    }
}

inline void __i8ToF32_512(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);

            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y0 = __dlc_char_as_int(_x0);
            int8_128 y1 = __dlc_char_as_int(_x1);
            int8_128 y2 = __dlc_char_as_int(_x2);
            int8_128 y3 = __dlc_char_as_int(_x3);

            int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
            int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
            int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
            int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

            store8_128_stride(j / 32, 4, dst, __dlc_int2float_rn(result0));
            store8_128_stride((j + 128) / 32, 4, dst, __dlc_int2float_rn(result1));
            store8_128_stride((j + 256) / 32, 4, dst, __dlc_int2float_rn(result2));
            store8_128_stride((j + 384) / 32, 4, dst, __dlc_int2float_rn(result3));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y0 = __dlc_char_as_int(_x0);
        int8_128 y1 = __dlc_char_as_int(_x1);
        int8_128 y2 = __dlc_char_as_int(_x2);
        int8_128 y3 = __dlc_char_as_int(_x3);

        int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
        int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
        int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
        int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

        store8_128_stride_with_stmask(0 / 32, 4, ldst_mask, dst, __dlc_int2float_rn(result0));
        store8_128_stride_with_stmask((128) / 32, 4, ldst_mask, dst, __dlc_int2float_rn(result1));
        store8_128_stride_with_stmask((256) / 32, 4, ldst_mask, dst, __dlc_int2float_rn(result2));
        store8_128_stride_with_stmask((384) / 32, 4, ldst_mask, dst, __dlc_int2float_rn(result3));
    }
}

inline void __i8ToF32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int dim0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128));
        int ldst_mask3 = pre_exp2((len128 / 128));
        int ldst_mask4 = pre_exp2((len128 / 128));
        if ((384 == (dim0 % 512))) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((256 == (dim0 % 512))) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((128 == (dim0 % 512))) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }

        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y0 = __dlc_char_as_int(_x0);
        int8_128 y1 = __dlc_char_as_int(_x1);
        int8_128 y2 = __dlc_char_as_int(_x2);
        int8_128 y3 = __dlc_char_as_int(_x3);
        int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
        int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
        int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
        int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

        store8_128_stride_with_stmask(len1024 * 4 / 32 + 0 / 32, 4, ldst_mask, dst,
                                      __dlc_int2float_rn(result0));
        store8_128_stride_with_stmask(len1024 * 4 / 32 + (128) / 32, 4, ldst_mask2, dst,
                                      __dlc_int2float_rn(result1));
        store8_128_stride_with_stmask(len1024 * 4 / 32 + (256) / 32, 4, ldst_mask3, dst,
                                      __dlc_int2float_rn(result2));
        store8_128_stride_with_stmask(len1024 * 4 / 32 + (384) / 32, 4, ldst_mask4, dst,
                                      __dlc_int2float_rn(result3));
    }
    if (len1024 != 0) {
        for (int i = len1024 - 1024, j = len1024 * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y0 = __dlc_char_as_int(_x0);
            int8_128 y1 = __dlc_char_as_int(_x1);
            int8_128 y2 = __dlc_char_as_int(_x2);
            int8_128 y3 = __dlc_char_as_int(_x3);

            int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
            int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
            int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
            int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

            store8_128_stride(j / 32, 4, dst, __dlc_int2float_rn(result0));
            store8_128_stride((j + 128) / 32, 4, dst, __dlc_int2float_rn(result1));
            store8_128_stride((j + 256) / 32, 4, dst, __dlc_int2float_rn(result2));
            store8_128_stride((j + 384) / 32, 4, dst, __dlc_int2float_rn(result3));
        }
    }
}

inline void i8ToF32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8ToF32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void __uint8ToF32_512(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);

            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y0 = __dlc_char_as_int(_x0);
            int8_128 y1 = __dlc_char_as_int(_x1);
            int8_128 y2 = __dlc_char_as_int(_x2);
            int8_128 y3 = __dlc_char_as_int(_x3);

            store8_128_stride(j / 32, 4, dst, __dlc_int2float_rn(y0));
            store8_128_stride((j + 128) / 32, 4, dst, __dlc_int2float_rn(y1));
            store8_128_stride((j + 256) / 32, 4, dst, __dlc_int2float_rn(y2));
            store8_128_stride((j + 384) / 32, 4, dst, __dlc_int2float_rn(y3));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y0 = __dlc_char_as_int(_x0);
        int8_128 y1 = __dlc_char_as_int(_x1);
        int8_128 y2 = __dlc_char_as_int(_x2);
        int8_128 y3 = __dlc_char_as_int(_x3);

        store8_128_stride_with_stmask(0 / 32, 4, ldst_mask, dst, __dlc_int2float_rn(y0));
        store8_128_stride_with_stmask((128) / 32, 4, ldst_mask, dst, __dlc_int2float_rn(y1));
        store8_128_stride_with_stmask((256) / 32, 4, ldst_mask, dst, __dlc_int2float_rn(y2));
        store8_128_stride_with_stmask((384) / 32, 4, ldst_mask, dst, __dlc_int2float_rn(y3));
    }
}

inline void __uint8ToF32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int dim0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;

    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128));
        int ldst_mask3 = pre_exp2((len128 / 128));
        int ldst_mask4 = pre_exp2((len128 / 128));
        if ((384 == (dim0 % 512))) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((256 == (dim0 % 512))) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((128 == (dim0 % 512))) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }

        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 result0 = __dlc_char_as_int(_x0);
        int8_128 result1 = __dlc_char_as_int(_x1);
        int8_128 result2 = __dlc_char_as_int(_x2);
        int8_128 result3 = __dlc_char_as_int(_x3);

        store8_128_stride_with_stmask(len1024 * 4 / 32 + 0 / 32, 4, ldst_mask, dst,
                                      __dlc_int2float_rn(result0));
        store8_128_stride_with_stmask(len1024 * 4 / 32 + (128) / 32, 4, ldst_mask2, dst,
                                      __dlc_int2float_rn(result1));
        store8_128_stride_with_stmask(len1024 * 4 / 32 + (256) / 32, 4, ldst_mask3, dst,
                                      __dlc_int2float_rn(result2));
        store8_128_stride_with_stmask(len1024 * 4 / 32 + (384) / 32, 4, ldst_mask4, dst,
                                      __dlc_int2float_rn(result3));
    }
    if (len1024 != 0) {
        for (int i = len1024 - 1024, j = len1024 * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 result0 = __dlc_char_as_int(_x0);
            int8_128 result1 = __dlc_char_as_int(_x1);
            int8_128 result2 = __dlc_char_as_int(_x2);
            int8_128 result3 = __dlc_char_as_int(_x3);

            store8_128_stride(j / 32, 4, dst, __dlc_int2float_rn(result0));
            store8_128_stride((j + 128) / 32, 4, dst, __dlc_int2float_rn(result1));
            store8_128_stride((j + 256) / 32, 4, dst, __dlc_int2float_rn(result2));
            store8_128_stride((j + 384) / 32, 4, dst, __dlc_int2float_rn(result3));
        }
    }
}

inline void uint8ToF32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8ToF32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void __i8Toi32_512(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);

            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y0 = __dlc_char_as_int(_x0);
            int8_128 y1 = __dlc_char_as_int(_x1);
            int8_128 y2 = __dlc_char_as_int(_x2);
            int8_128 y3 = __dlc_char_as_int(_x3);

            int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
            int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
            int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
            int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

            store8_128_stride_i(j / 32, 4, dst, result0);
            store8_128_stride_i((j + 128) / 32, 4, dst, result1);
            store8_128_stride_i((j + 256) / 32, 4, dst, result2);
            store8_128_stride_i((j + 384) / 32, 4, dst, result3);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y0 = __dlc_char_as_int(_x0);
        int8_128 y1 = __dlc_char_as_int(_x1);
        int8_128 y2 = __dlc_char_as_int(_x2);
        int8_128 y3 = __dlc_char_as_int(_x3);

        int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
        int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
        int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
        int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

        store8_128_stride_with_stmask_i(0 / 32, 4, ldst_mask, dst, result0);
        store8_128_stride_with_stmask_i((128) / 32, 4, ldst_mask, dst, result1);
        store8_128_stride_with_stmask_i((256) / 32, 4, ldst_mask, dst, result2);
        store8_128_stride_with_stmask_i((384) / 32, 4, ldst_mask, dst, result3);
    }
}

inline void __i8Toi32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int dim0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;

    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128));
        int ldst_mask3 = pre_exp2((len128 / 128));
        int ldst_mask4 = pre_exp2((len128 / 128));
        if ((384 == (dim0 % 512))) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((256 == (dim0 % 512))) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((128 == (dim0 % 512))) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }

        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y0 = __dlc_char_as_int(_x0);
        int8_128 y1 = __dlc_char_as_int(_x1);
        int8_128 y2 = __dlc_char_as_int(_x2);
        int8_128 y3 = __dlc_char_as_int(_x3);

        int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
        int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
        int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
        int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + 0 / 32, 4, ldst_mask, dst, result0);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (128) / 32, 4, ldst_mask2, dst, result1);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (256) / 32, 4, ldst_mask3, dst, result2);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (384) / 32, 4, ldst_mask4, dst, result3);
    }
    if (len1024 != 0) {
        for (int i = len1024 - 1024, j = len1024 * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y0 = __dlc_char_as_int(_x0);
            int8_128 y1 = __dlc_char_as_int(_x1);
            int8_128 y2 = __dlc_char_as_int(_x2);
            int8_128 y3 = __dlc_char_as_int(_x3);

            int8_128 result0 = v_s32_sel(v_s32_cmp(GT, y0, v_u32_move_i(127)), y0, y0 - v_u32_move_i(256));
            int8_128 result1 = v_s32_sel(v_s32_cmp(GT, y1, v_u32_move_i(127)), y1, y1 - v_u32_move_i(256));
            int8_128 result2 = v_s32_sel(v_s32_cmp(GT, y2, v_u32_move_i(127)), y2, y2 - v_u32_move_i(256));
            int8_128 result3 = v_s32_sel(v_s32_cmp(GT, y3, v_u32_move_i(127)), y3, y3 - v_u32_move_i(256));

            store8_128_stride_i(j / 32, 4, dst, result0);
            store8_128_stride_i((j + 128) / 32, 4, dst, result1);
            store8_128_stride_i((j + 256) / 32, 4, dst, result2);
            store8_128_stride_i((j + 384) / 32, 4, dst, result3);
        }
    }
}

inline void __i8Toi64_512(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int d0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);

            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y0 = __dlc_char_as_int(_x0);
            int8_128 y1 = __dlc_char_as_int(_x1);
            int8_128 y2 = __dlc_char_as_int(_x2);
            int8_128 y3 = __dlc_char_as_int(_x3);

            bool8_128 CheckSign0 = v_s32_cmp(GT, y0, v_u32_move_i(127));
            bool8_128 CheckSign1 = v_s32_cmp(GT, y1, v_u32_move_i(127));
            bool8_128 CheckSign2 = v_s32_cmp(GT, y2, v_u32_move_i(127));
            bool8_128 CheckSign3 = v_s32_cmp(GT, y3, v_u32_move_i(127));

            int8_128 result0 = v_s32_sel(CheckSign0, y0, y0 - v_u32_move_i(256));
            int8_128 result1 = v_s32_sel(CheckSign1, y1, y1 - v_u32_move_i(256));
            int8_128 result2 = v_s32_sel(CheckSign2, y2, y2 - v_u32_move_i(256));
            int8_128 result3 = v_s32_sel(CheckSign3, y3, y3 - v_u32_move_i(256));

            int8_128 Sign1 = v_s32_sel(CheckSign0, 0, 0xffffffff);
            int8_128 Sign2 = v_s32_sel(CheckSign1, 0, 0xffffffff);
            int8_128 Sign3 = v_s32_sel(CheckSign2, 0, 0xffffffff);
            int8_128 Sign4 = v_s32_sel(CheckSign3, 0, 0xffffffff);

            store8_128_stride_i(j / 32, 4, dst, result0);
            store8_128_stride_i((j + 128) / 32, 4, dst, result1);
            store8_128_stride_i((j + 256) / 32, 4, dst, result2);
            store8_128_stride_i((j + 384) / 32, 4, dst, result3);

            store8_128_stride_with_stmask_cmem(j / 32, 4, 255, cmem, *(float8_128 *)(&Sign1));
            store8_128_stride_with_stmask_cmem((j + 128) / 32, 4, 255, cmem, *(float8_128 *)(&Sign2));
            store8_128_stride_with_stmask_cmem((j + 256) / 32, 4, 255, cmem, *(float8_128 *)(&Sign3));
            store8_128_stride_with_stmask_cmem((j + 384) / 32, 4, 255, cmem, *(float8_128 *)(&Sign4));

        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y0 = __dlc_char_as_int(_x0);
        int8_128 y1 = __dlc_char_as_int(_x1);
        int8_128 y2 = __dlc_char_as_int(_x2);
        int8_128 y3 = __dlc_char_as_int(_x3);

        bool8_128 CheckSign0 = v_s32_cmp(GT, y0, v_u32_move_i(127));
        bool8_128 CheckSign1 = v_s32_cmp(GT, y1, v_u32_move_i(127));
        bool8_128 CheckSign2 = v_s32_cmp(GT, y2, v_u32_move_i(127));
        bool8_128 CheckSign3 = v_s32_cmp(GT, y3, v_u32_move_i(127));

        int8_128 result0 = v_s32_sel(CheckSign0, y0, y0 - v_u32_move_i(256));
        int8_128 result1 = v_s32_sel(CheckSign1, y1, y1 - v_u32_move_i(256));
        int8_128 result2 = v_s32_sel(CheckSign2, y2, y2 - v_u32_move_i(256));
        int8_128 result3 = v_s32_sel(CheckSign3, y3, y3 - v_u32_move_i(256));

        int8_128 Sign1 = v_s32_sel(CheckSign0, 0, 0xffffffff);
        int8_128 Sign2 = v_s32_sel(CheckSign1, 0, 0xffffffff);
        int8_128 Sign3 = v_s32_sel(CheckSign2, 0, 0xffffffff);
        int8_128 Sign4 = v_s32_sel(CheckSign3, 0, 0xffffffff);

        store8_128_stride_with_stmask_i(0, 4, ldst_mask, dst, result0);
        store8_128_stride_with_stmask_i(4, 4, ldst_mask, dst, result1);
        store8_128_stride_with_stmask_i(8, 4, ldst_mask, dst, result2);
        store8_128_stride_with_stmask_i(12, 4, ldst_mask, dst, result3);

        store8_128_stride_with_stmask_cmem(0, 4, ldst_mask, cmem, *(float8_128 *)(&Sign1));
        store8_128_stride_with_stmask_cmem(4, 4, ldst_mask, cmem, *(float8_128 *)(&Sign2));
        store8_128_stride_with_stmask_cmem(8, 4, ldst_mask, cmem, *(float8_128 *)(&Sign3));
        store8_128_stride_with_stmask_cmem(12, 4, ldst_mask, cmem, *(float8_128 *)(&Sign4));

    }
}

inline void __i8Toi64(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int dim0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128));
        int ldst_mask3 = pre_exp2((len128 / 128));
        int ldst_mask4 = pre_exp2((len128 / 128));
        if ((384 == (dim0 % 512))) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((256 == (dim0 % 512))) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((128 == (dim0 % 512))) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }

        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y0 = __dlc_char_as_int(_x0);
        int8_128 y1 = __dlc_char_as_int(_x1);
        int8_128 y2 = __dlc_char_as_int(_x2);
        int8_128 y3 = __dlc_char_as_int(_x3);

        bool8_128 CheckSign0 = v_s32_cmp(GT, y0, v_u32_move_i(127));
        bool8_128 CheckSign1 = v_s32_cmp(GT, y1, v_u32_move_i(127));
        bool8_128 CheckSign2 = v_s32_cmp(GT, y2, v_u32_move_i(127));
        bool8_128 CheckSign3 = v_s32_cmp(GT, y3, v_u32_move_i(127));

        int8_128 result0 = v_s32_sel(CheckSign0, y0, y0 - v_u32_move_i(256));
        int8_128 result1 = v_s32_sel(CheckSign1, y1, y1 - v_u32_move_i(256));
        int8_128 result2 = v_s32_sel(CheckSign2, y2, y2 - v_u32_move_i(256));
        int8_128 result3 = v_s32_sel(CheckSign3, y3, y3 - v_u32_move_i(256));

        int8_128 Sign1 = v_s32_sel(CheckSign0, 0, 0xffffffff);
        int8_128 Sign2 = v_s32_sel(CheckSign1, 0, 0xffffffff);
        int8_128 Sign3 = v_s32_sel(CheckSign2, 0, 0xffffffff);
        int8_128 Sign4 = v_s32_sel(CheckSign3, 0, 0xffffffff);

        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + 0 / 32, 4, ldst_mask, dst, result0);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (128) / 32, 4, ldst_mask2, dst, result1);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (256) / 32, 4, ldst_mask3, dst, result2);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (384) / 32, 4, ldst_mask4, dst, result3);

        store8_128_stride_with_stmask_cmem(len1024 * 4 / 32 + 0 / 32, 4, ldst_mask, cmem, *(float8_128 *)(&Sign1));
        store8_128_stride_with_stmask_cmem(len1024 * 4 / 32 + (128) / 32, 4, ldst_mask2, cmem, *(float8_128 *)(&Sign2));
        store8_128_stride_with_stmask_cmem(len1024 * 4 / 32 + (256) / 32, 4, ldst_mask3, cmem, *(float8_128 *)(&Sign3));
        store8_128_stride_with_stmask_cmem(len1024 * 4 / 32 + (384) / 32, 4, ldst_mask4, cmem, *(float8_128 *)(&Sign4));
    }
    if (len1024 != 0) {
        for (int i = len1024 - 1024, j = len1024 * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y0 = __dlc_char_as_int(_x0);
            int8_128 y1 = __dlc_char_as_int(_x1);
            int8_128 y2 = __dlc_char_as_int(_x2);
            int8_128 y3 = __dlc_char_as_int(_x3);

            bool8_128 CheckSign0 = v_s32_cmp(GT, y0, v_u32_move_i(127));
            bool8_128 CheckSign1 = v_s32_cmp(GT, y1, v_u32_move_i(127));
            bool8_128 CheckSign2 = v_s32_cmp(GT, y2, v_u32_move_i(127));
            bool8_128 CheckSign3 = v_s32_cmp(GT, y3, v_u32_move_i(127));

            int8_128 result0 = v_s32_sel(CheckSign0, y0, y0 - v_u32_move_i(256));
            int8_128 result1 = v_s32_sel(CheckSign1, y1, y1 - v_u32_move_i(256));
            int8_128 result2 = v_s32_sel(CheckSign2, y2, y2 - v_u32_move_i(256));
            int8_128 result3 = v_s32_sel(CheckSign3, y3, y3 - v_u32_move_i(256));

            int8_128 Sign1 = v_s32_sel(CheckSign0, 0, 0xffffffff);
            int8_128 Sign2 = v_s32_sel(CheckSign1, 0, 0xffffffff);
            int8_128 Sign3 = v_s32_sel(CheckSign2, 0, 0xffffffff);
            int8_128 Sign4 = v_s32_sel(CheckSign3, 0, 0xffffffff);

            store8_128_stride_i(j / 32, 4, dst, result0);
            store8_128_stride_i((j + 128) / 32, 4, dst, result1);
            store8_128_stride_i((j + 256) / 32, 4, dst, result2);
            store8_128_stride_i((j + 384) / 32, 4, dst, result3);

            store8_128_stride_with_stmask_cmem(j / 32, 4, 255, cmem, *(float8_128 *)(&Sign1));
            store8_128_stride_with_stmask_cmem((j + 128) / 32, 4, 255, cmem, *(float8_128 *)(&Sign2));
            store8_128_stride_with_stmask_cmem((j + 256) / 32, 4, 255, cmem, *(float8_128 *)(&Sign3));
            store8_128_stride_with_stmask_cmem((j + 384) / 32, 4, 255, cmem, *(float8_128 *)(&Sign4));

        }
    }
}

inline void i8Toi32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void __uint8Toi32_512(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);

            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 result0 = __dlc_char_as_int(_x0);
            int8_128 result1 = __dlc_char_as_int(_x1);
            int8_128 result2 = __dlc_char_as_int(_x2);
            int8_128 result3 = __dlc_char_as_int(_x3);

            store8_128_stride_i(j / 32, 4, dst, result0);
            store8_128_stride_i((j + 128) / 32, 4, dst, result1);
            store8_128_stride_i((j + 256) / 32, 4, dst, result2);
            store8_128_stride_i((j + 384) / 32, 4, dst, result3);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(0, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 result0 = __dlc_char_as_int(_x0);
        int8_128 result1 = __dlc_char_as_int(_x1);
        int8_128 result2 = __dlc_char_as_int(_x2);
        int8_128 result3 = __dlc_char_as_int(_x3);

        store8_128_stride_with_stmask_i(0 / 32, 4, ldst_mask, dst, result0);
        store8_128_stride_with_stmask_i((128) / 32, 4, ldst_mask, dst, result1);
        store8_128_stride_with_stmask_i((256) / 32, 4, ldst_mask, dst, result2);
        store8_128_stride_with_stmask_i((384) / 32, 4, ldst_mask, dst, result3);
    }
}

inline void __uint8Toi32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int dim0, bool is_128) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;

    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128));
        int ldst_mask3 = pre_exp2((len128 / 128));
        int ldst_mask4 = pre_exp2((len128 / 128));
        if ((384 == (dim0 % 512))) {
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((256 == (dim0 % 512))) {
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        } else if ((128 == (dim0 % 512))) {
            ldst_mask2 = pre_exp2(len128 / 128 - 1);
            ldst_mask3 = pre_exp2(len128 / 128 - 1);
            ldst_mask4 = pre_exp2(len128 / 128 - 1);
        }

        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 result0 = __dlc_char_as_int(_x0);
        int8_128 result1 = __dlc_char_as_int(_x1);
        int8_128 result2 = __dlc_char_as_int(_x2);
        int8_128 result3 = __dlc_char_as_int(_x3);

        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + 0 / 32, 4, ldst_mask, dst, result0);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (128) / 32, 4, ldst_mask2, dst, result1);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (256) / 32, 4, ldst_mask3, dst, result2);
        store8_128_stride_with_stmask_i(len1024 * 4 / 32 + (384) / 32, 4, ldst_mask4, dst, result3);
    }
    if (len1024 != 0) {
        for (int i = len1024 - 1024, j = len1024 * 4 - 4096; i >= 0; i -= 1024, j -= 4096) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 result0 = __dlc_char_as_int(_x0);
            int8_128 result1 = __dlc_char_as_int(_x1);
            int8_128 result2 = __dlc_char_as_int(_x2);
            int8_128 result3 = __dlc_char_as_int(_x3);

            store8_128_stride_i(j / 32, 4, dst, result0);
            store8_128_stride_i((j + 128) / 32, 4, dst, result1);
            store8_128_stride_i((j + 256) / 32, 4, dst, result2);
            store8_128_stride_i((j + 384) / 32, 4, dst, result3);
        }
    }
}

inline void fp32Toi64(SIM_X86::tensor input, SIM_X86::tensor output, int VMEMsize) {
    int8_128 inf_v = v_u32_move_i(0x7f800000);
    float8_128 neg_inf_v_ = v_u32_move_f(MIN_FLT);
    int8_128 neg_inf_v = *(int8_128 *)(&neg_inf_v_);
    int8_128 limit_value = v_u32_move_i(-2147483648); // inf, -inf, nan, 1e38
    float8_128 v_2147483647 = v_u32_move_f((float)2147483647);
    int8_128 CheckSign;
    int i = 0, j = 0;
    for (; i + 1024 < VMEMsize; i += 1024, j += 2048) {
        float8_128 x = v_f32_ld_tnsr_b(i / 32, input);
        int8_128 y = __dlc_float2int_rz(x);
        float8_128 x_ = v_f32_abs(x);
        int8_128 x1 = *(int8_128 *)(&x_);
        int8_128 y1 = v_s32_sel(v_f32_cmp(GTEQ, x, v_2147483647), y, limit_value);
        int8_128 y2 = v_s32_sel(v_s32_cmp(LSEQ, x1, neg_inf_v), y1, limit_value);
        int8_128 y3 = v_s32_sel(v_s32_cmp(GTEQ, x1, inf_v), y2, limit_value);
        CheckSign = v_s32_sel(v_s32_cmp(LS, y3, 0), 0, 0xffffffff);
        store8_128_stride2_with_stmask_i(j / 32, 2, 255, output, y3);
        store8_128_stride2_with_stmask_i((j + 128) / 32, 2, 255, output, CheckSign);
    }
    if (i < VMEMsize) {
        int len = VMEMsize - i;
        int ldst_vmask = pre_exp2(len / 128);
        float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, input, 1, ldst_vmask);
        int8_128 y = __dlc_float2int_rz(x);
        float8_128 x_ = v_f32_abs(x);
        int8_128 x1 = *(int8_128 *)(&x_);
        int8_128 y1 = v_s32_sel(v_f32_cmp(GTEQ, x, v_2147483647), y, limit_value);
        int8_128 y2 = v_s32_sel(v_s32_cmp(LSEQ, x1, neg_inf_v), y1, limit_value);
        int8_128 y3 = v_s32_sel(v_s32_cmp(GTEQ, x1, inf_v), y2, limit_value);
        CheckSign = v_s32_sel(v_s32_cmp(LS, y3, 0), 0, 0xffffffff);
        store8_128_stride2_with_stmask_i(j / 32, 2, ldst_vmask, output, y3);
        store8_128_stride2_with_stmask_i((j + 128) / 32, 2, ldst_vmask, output, CheckSign);
    }
}

inline void int32Toi64(SIM_X86::tensor input, SIM_X86::tensor output, int VMEMsize) {
    int8_128 CheckSign;
    int i = 0, j = 0;
    for (; i + 1024 < VMEMsize; i += 1024, j += 2048) {
        int8_128 x = v_i32_ld_tnsr(i / 32, input, 1, 255);
        CheckSign = v_s32_sel(v_s32_cmp(LS, x, 0), 0, 0xffffffff);
        store8_128_stride2_with_stmask_i(j / 32, 2, 255, output, x);
        store8_128_stride2_with_stmask_i((j + 128) / 32, 2, 255, output, CheckSign);
    }
    if (i < VMEMsize) {
        int len = VMEMsize - i;
        int ldst_vmask = pre_exp2(len / 128);
        int8_128 x = v_i32_ld_tnsr(i / 32, input, 1, ldst_vmask);
        CheckSign = v_s32_sel(v_s32_cmp(LS, x, 0), 0, 0xffffffff);
        store8_128_stride2_with_stmask_i(j / 32, 2, ldst_vmask, output, x);
        store8_128_stride2_with_stmask_i((j + 128) / 32, 2, ldst_vmask, output, CheckSign);
    }
}

inline void uint8Toi32(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void uint8Toi32_sdiv(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = soft_sdiv(len, bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void i8Tobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
    if (d0k128 % 4 == 0) {
        __i32Tobool(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                        true);
        }
    }
}

inline void i8Tof16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8ToF32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
    if (d0k128 % 2 == 0) {
        __fp32Tofp16_256(mem, dst, len_, false); // mem和dst是一个tensor
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len_ + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __fp32Tofp16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                             true);
        }
    }
}

inline void i8Tobf16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8ToF32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
    f32ToBf16(dst, len_, d0);
}

inline void i8toi16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
    if (d0k128 % 2 == 0) {
        __i32Toi16_256(mem, dst, len_, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len_ + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __i32Toi16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                           true);
        }
    }
}

inline void i8Touint8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
    if (d0k128 % 4 == 0) {
        __i32Touint8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void i8ToAnother(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, DLCType out_dtype) {
    if (out_dtype == dlc_fp32) {
        i8ToF32(mem, dst, len, d0);
    } else if (out_dtype == dlc_int32) {
        i8Toi32(mem, dst, len, d0);
    } else if (out_dtype == dlc_fp16) {
        i8Tof16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_bf16) {
        i8Tobf16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_int16) {
        i8toi16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_bool) {
        i8Tobool(mem, dst, len, d0);
    } else if (out_dtype == dlc_uint8) {
        i8Touint8(mem, dst, len, d0);
    }
}

inline void uint8Tof16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8ToF32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
    if (d0k128 % 2 == 0) {
        __fp32Tofp16_256(mem, dst, len_, false); // mem和dst是一个tensor
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len_ + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __fp32Tofp16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                             true);
        }
    }
}

inline void uint8Tobf16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8ToF32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
    f32ToBf16(dst, len_, d0);
}

inline void uint8toi16(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
    if (d0k128 % 2 == 0) {
        __i32Toi16_256(mem, dst, len_, false);
    } else {
        int bd0 = ALIGN256(d0);
        int h = (len_ + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __i32Toi16_256((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 2 / 32), bd0 / 2,
                           true);
        }
    }
}

inline void uint8Toi8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
    if (d0k128 % 4 == 0) {
        __i32Toi8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void uint8Tobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __uint8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                         true);
        }
    }
    if (d0k128 % 4 == 0) {
        __i32Tobool(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Tobool((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                        true);
        }
    }
}

inline void uint8ToAnother(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, DLCType out_dtype) {
    if (out_dtype == dlc_fp32) {
        uint8ToF32(mem, dst, len, d0);
    } else if (out_dtype == dlc_int32) {
        uint8Toi32(mem, dst, len, d0);
    } else if (out_dtype == dlc_fp16) {
        uint8Tof16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_bf16) {
        uint8Tobf16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_int16) {
        uint8toi16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_int8) { // 这里uint8转int8,实际上和boolToint8是一样的
        uint8Toi8(mem, dst, len, d0);
    } else if (out_dtype == dlc_bool) {
        uint8Tobool(mem, dst, len, d0);
    }
}

inline void boolToi8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
    if (d0k128 % 4 == 0) {
        __i32Toi8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Toi8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                      true);
        }
    }
}

inline void boolTouint8(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi32((SIM_X86::tensor )((int)mem + i * bd0 / 4 / 32), (SIM_X86::tensor )((int)dst + i * d0 / 32), bd0 / 4, d0,
                      true);
        }
    }
    if (d0k128 % 4 == 0) {
        __i32Touint8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        int h = (len + (bd0 / 4) - 1) / (bd0 / 4);
        for (int i = 0; i < h; i++) {
            __i32Touint8((SIM_X86::tensor )((int)mem + i * d0 / 32), (SIM_X86::tensor )((int)dst + i * bd0 / 4 / 32), bd0 / 4, d0,
                         true);
        }
    }
}

inline void boolToAnother(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, DLCType out_dtype) {
    if (out_dtype == dlc_fp32) {
        uint8ToF32(mem, dst, len, d0);
    } else if (out_dtype == dlc_int32) {
        uint8Toi32(mem, dst, len, d0);
    } else if (out_dtype == dlc_fp16) {
        uint8Tof16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_bf16) {
        uint8Tobf16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_int16) {
        uint8toi16(mem, dst, len, len_, d0);
    } else if (out_dtype == dlc_int8) {
        boolToi8(mem, dst, len, d0);
    } else if (out_dtype == dlc_uint8) {
        boolTouint8(mem, dst, len, d0);
    }
}