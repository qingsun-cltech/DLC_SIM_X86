#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #pragma once
#include "ldst.h"
// #include "typehint.h"

inline int8_128 __$S(float8_128 a) {
    int8_128 result0 = *(int8_128*)(&a);
    return result0;
}

inline void __bf16ToF32_256(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            store8_128_stride(j / 32, 2, dst, bfloat16_to_float(x1));
            store8_128_stride((j + 128) / 32, 2, dst, bfloat16_to_float(x2));
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        store8_128_stride_with_stmask(0, 2, ldst_mask, dst, bfloat16_to_float(x1));
        store8_128_stride_with_stmask(4, 2, ldst_mask, dst, bfloat16_to_float(x2));
    }
}

inline void __bf16ToF32_128(SIM_X86::tensor mem, SIM_X86::tensor dst, int len) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_b(i / 32, mem);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);
            store8_128_stride(j / 32, 2, dst, bfloat16_to_float(x1));
            if (i == (len - 1024)) {
                store8_128_stride_stmk((j + 128) / 32, 2, dst, bfloat16_to_float(x2), 127);
            } else {
                store8_128_stride((j + 128) / 32, 2, dst, bfloat16_to_float(x2));
            }
        }
        // {
        //     int ii = len % 1024;
        //     int jj = (len * 2) % 2048;
        //     float8_128 x = v_f32_ld_tnsr_b(ii / 32, mem);
        //     short8_128 x2 = unpack_16b($S(x), 1);
        //     short8_128 x1 = unpack_16b($S(x), 0);
        //     store8_128_stride(jj / 32, 2, dst, bfloat16_to_float(x1));
        //     store8_128_stride_stmk((jj + 128) / 32, 2, dst, bfloat16_to_float(x2), 127);
        // }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if(len >= 1024){
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);
        store8_128_stride_with_stmask(0, 2, ldst_mask, dst, bfloat16_to_float(x1));
        store8_128_stride_with_stmask(4, 2, ldst_mask2, dst, bfloat16_to_float(x2));
    }
}

// len: src bf16 mem len (4B) 一定是(bd0 / 2)的倍数
// d0: before all padding
inline void bf16ToF32(SIM_X86::tensor mem, int len, int d0) {
    d0 = ((d0 + 127) / 128) * 128;
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16ToF32_256(mem, mem, len);
    } else {
        int bd0 = ((d0 + 255) / 256) * 256;
        //避免因为整数除法不精准，导致结果比真实值小1
        int h = (len + (bd0 / 2) - 1)/ (bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __bf16ToF32_128(tensor_slice(mem, i * bd0 / 2 / 32), tensor_slice(mem, i * d0 / 32), bd0 / 2);
        }
    }
}

inline void bf16ToF32_h(SIM_X86::tensor mem, int len, int h, int d0) {
    d0 = ((d0 + 127) / 128) * 128;
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16ToF32_256(mem, mem, len);
    } else {
        int bd0 = ((d0 + 255) / 256) * 256;
        for (int i = h - 1; i >= 0; i--) {
            __bf16ToF32_128(tensor_slice(mem, i * bd0 / 2 / 32), tensor_slice(mem, i * d0 / 32), bd0 / 2);
        }
    }
}

inline void bf16ToF32_sdiv(SIM_X86::tensor mem, int len, int d0) {
    d0 = ((d0 + 127) / 128) * 128;
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16ToF32_256(mem, mem, len);
    } else {
        int bd0 = ((d0 + 255) / 256) * 256;
        int h = soft_sdiv(len, bd0 / 2);
        for (int i = h - 1; i >= 0; i--) {
            __bf16ToF32_128(tensor_slice(mem, i * bd0 / 2 / 32), tensor_slice(mem, i * d0 / 32), bd0 / 2);
        }
    }
}

inline float8_128 __$F(int8_128 a) {
    float8_128 result0 = *(float8_128*)(&a);
    return result0;
}

inline void __f32ToBf16_256(SIM_X86::tensor input, SIM_X86::tensor output, int _len, bool is_128) {
    int len = _len * 2;
    for (int i = 0, j = 0; i < len; i += 2048, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        if((i + 2048) >= len && is_128){
            ldst_mask2 = pre_exp2(l / 128 - 1);
        }
        float8_128 x1 = load8_128_stride_with_ldmask(i / 32, 2, ldst_mask, input);
        float8_128 x2 = load8_128_stride_with_ldmask((i + 128) / 32, 2, ldst_mask2, input);
        store8_128_stride_with_stmask(j / 32, 1, ldst_mask, output, __$F(float_to_bfloat16(x2, x1)));
    }
}

// len: target bf16 mem len (4B)
// d0: before all padding
inline void f32ToBf16(SIM_X86::tensor mem, int len, int d0) {
    d0 = ((d0 + 127) / 128) * 128;
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32ToBf16_256(mem, mem, len, false);
    } else {
        int bd0 = ((d0 + 255) / 256) * 256;
        int h = (len + (bd0 / 2) - 1) / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __f32ToBf16_256(tensor_slice(mem, i * d0 / 32), tensor_slice(mem, i * bd0 / 2 / 32), bd0 / 2, true);
        }
    }
}

inline void f32ToBf16_sdiv(SIM_X86::tensor mem, int len, int d0) {
    d0 = ((d0 + 127) / 128) * 128;
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32ToBf16_256(mem, mem, len, false);
    } else {
        int bd0 = ((d0 + 255) / 256) * 256;
        int h = soft_sdiv(len, bd0 / 2);
        for (int i = 0; i < h; i++) {
            __f32ToBf16_256(tensor_slice(mem, i * d0 / 32), tensor_slice(mem, i * bd0 / 2 / 32), bd0 / 2, true);
        }
    }
}

inline void f32ToBf16_h(SIM_X86::tensor mem, int len, int h, int d0) {
    d0 = ((d0 + 127) / 128) * 128;
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32ToBf16_256(mem, mem, len, false);
    } else {
        int bd0 = ((d0 + 255) / 256) * 256;
        for (int i = 0; i < h; i++) {
            __f32ToBf16_256(tensor_slice(mem, i * d0 / 32), tensor_slice(mem, i * bd0 / 2 / 32), bd0 / 2, true);
        }
    }
}

inline void __f32ToBf16_256_v2c(SIM_X86::tensor input, SIM_X86::tensor output, int _len, bool is_128) {
    int len = _len * 2;
    for (int i = 0, j = 0; i < len; i += 2048, j += 1024) {
        int l = min(_len - j, 1024);
        int ldst_mask = pre_exp2(l / 128);
        int ldst_mask2 = pre_exp2(l / 128);
        if((i + 2048) >= len && is_128){
            ldst_mask2 = pre_exp2(l / 128 - 1);
        }
        float8_128 x1 = load8_128_stride_with_ldmask(i / 32, 2, ldst_mask, input);
        float8_128 x2 = load8_128_stride_with_ldmask((i + 128) / 32, 2, ldst_mask2, input);
        v_f32_fxc_store(j / 32, output, 1, ldst_mask, __$F(float_to_bfloat16(x2, x1)));
    }
}

inline void f32ToBf16_v2c(SIM_X86::tensor mem, SIM_X86::tensor cmem_output, int len, int d0) {
    d0 = ((d0 + 127) / 128) * 128;
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32ToBf16_256_v2c(mem, cmem_output, len, false);
    } else {
        int bd0 = ((d0 + 255) / 256) * 256;
        int h = len / (bd0 / 2);
        for (int i = 0; i < h; i++) {
            __f32ToBf16_256_v2c(tensor_slice(mem, i * d0 / 32), tensor_slice(cmem_output, i * bd0 / 2 / 32), bd0 / 2, true);
        }
    }
}



