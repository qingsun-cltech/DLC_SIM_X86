#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "convert_element_type.h"

inline void fp32Tofp16_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __fp32Tofp16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = 0; i < h; i++) {
            __fp32Tofp16_256(mem + i * d0 / 32, dst + i * bd0 / 2 / 32, bd0 / 2, true);
        }
    }
}

inline void fp32Tobf16_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32ToBf16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = 0; i < h; i++) {
            __f32ToBf16_256(mem + i * d0 / 32, dst + i * bd0 / 2 / 32, bd0 / 2, true);
        }
    }
}

inline void fp32Toi16_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f32Toi16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = 0; i < h; i++) {
            __f32Toi16_256(mem + i * d0 / 32, dst + i * bd0 / 2 / 32, bd0 / 2, true);
        }
    }
}

inline void fp32Toi8_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Toi8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = 0; i < h; i++) {
            __f32Toi8(mem + i * d0 / 32, dst + i * bd0 / 4 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void fp32Tobool_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Tobool(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = 0; i < h; i++) {
            __f32Tobool(mem + i * d0 / 32, dst + i * bd0 / 4 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void fp32Touint8_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __f32Touint8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = 0; i < h; i++) {
            __f32Touint8(mem + i * d0 / 32, dst + i * bd0 / 4 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void i32Tofp16_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i32Tofp16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = 0; i < h; i++) {
            __i32Tofp16_256(mem + i * d0 / 32, dst + i * bd0 / 2 / 32, bd0 / 2, true);
        }
    }
}

inline void i32Tobf16_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i32ToBf16_256(mem, dst, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = 0; i < h; i++) {
            __i32ToBf16_256(mem + i * d0 / 32, dst + i * bd0 / 2 / 32, bd0 / 2, true);
        }
    }
}

inline void i32Toi16_h(SIM_X86::tensor mem, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i32Toi16_256(mem, mem, len, false);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = 0; i < h; i++) {
            __i32Toi16_256(mem + i * d0 / 32, mem + i * bd0 / 2 / 32, bd0 / 2, true);
        }
    }
}

inline void i32Touint8_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Touint8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = 0; i < h; i++) {
            __i32Touint8(mem + i * d0 / 32, dst + i * bd0 / 4 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void i32Toi8_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Toi8(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = 0; i < h; i++) {
            __i32Toi8(mem + i * d0 / 32, dst + i * bd0 / 4 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void i32Tobool_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i32Tobool(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = 0; i < h; i++) {
            __i32Tobool(mem + i * d0 / 32, dst + i * bd0 / 4 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void f16ToF32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __f16ToF32_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void f16Toi32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __f16Toi32_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void f16Toi64_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __f16Toi64_256(mem, dst, cmem, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __f16Toi64_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, cmem + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void bf16Toi32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __bf16Toi32_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void bf16Toi64_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16Toi64_256(mem, dst, cmem, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __bf16Toi64_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, cmem + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void i16ToF32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16ToF32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __i16ToF32_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void i16Toi32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16Toi32_256(mem, dst, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __i16Toi32_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void i16Toi64_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __i16Toi64_256(mem, dst, cmem, len);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __i16Toi64_128(mem + i * bd0 / 2 / 32, dst + i * d0 / 32, cmem + i * d0 / 32, bd0 / 2);
        }
    }
}

inline void uint8ToF32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = h - 1; i >= 0; i--) {
            __uint8ToF32(mem + i * bd0 / 4 / 32, dst + i * d0 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void uint8Toi32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __uint8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = h - 1; i >= 0; i--) {
            __uint8Toi32(mem + i * bd0 / 4 / 32, dst + i * d0 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void i8ToF32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8ToF32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = h - 1; i >= 0; i--) {
            __i8ToF32(mem + i * bd0 / 4 / 32, dst + i * d0 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void i8Toi32_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi32_512(mem, dst, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi32(mem + i * bd0 / 4 / 32, dst + i * d0 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void i8Toi64_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int h, int d0) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 4 == 0) {
        __i8Toi64_512(mem, dst, cmem, len, d0, false);
    } else {
        int bd0 = ALIGN512(d0);
        for (int i = h - 1; i >= 0; i--) {
            __i8Toi64(mem + i * bd0 / 4 / 32, dst + i * d0 / 32, cmem + i * d0 / 32, bd0 / 4, d0, true);
        }
    }
}

inline void f16ToAnother_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int len_, int d0, int d0_256, int d0_512,
                           DLCType out_dtype) {
    int h = soft_sdiv(len, d0_256 / 2);
    int h_ = soft_sdiv(len_, d0_512 / 4);
    if (out_dtype == dlc_fp32) {
        f16ToF32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int32) {
        f16Toi32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int64) {
        f16Toi64_h(mem, dst, cmem, len, h, d0);
    }else if (out_dtype == dlc_bf16) {
        f16Tobf16(mem, dst, len);
    } else if (out_dtype == dlc_int16) {
        f16Toi16(mem, dst, len);
    } else if (out_dtype == dlc_int8) {
        f16ToF32_h(mem, dst, len, h, d0);
        fp32Toi8_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_bool) {
        f16ToF32_h(mem, dst, len, h, d0);
        fp32Tobool_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_uint8) {
        f16ToF32_h(mem, dst, len, h, d0);
        fp32Touint8_h(mem, dst, len_, h_, d0);
    }
}

inline void bf16ToAnother_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int len_, int d0, int d0_256, int d0_512,
                            DLCType out_dtype) {
    int h = soft_sdiv(len, d0_256 / 2);
    int h_ = soft_sdiv(len_, d0_512 / 4);
    if (out_dtype == dlc_fp32) {
        bf16ToF32_h(mem, len, h, d0); // mem 和dst是一个
    } else if (out_dtype == dlc_int32) {
        bf16Toi32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int64) {
        bf16Toi64_h(mem, dst, cmem, len, h, d0);
    } else if (out_dtype == dlc_fp16) {
        bf16Tof16(mem, dst, len);
    } else if (out_dtype == dlc_int16) {
        bf16Toi16(mem, dst, len);
    } else if (out_dtype == dlc_int8) {
        bf16ToF32_h(mem, len, h, d0);
        fp32Toi8_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_bool) {
        bf16ToF32_h(mem, len, h, d0);
        fp32Tobool_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_uint8) {
        bf16ToF32_h(mem, len, h, d0);
        fp32Touint8_h(mem, dst, len_, h_, d0);
    }
}

inline void i16ToAnother_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int len_, int d0, int d0_256, int d0_512,
                           DLCType out_dtype) {
    int h = soft_sdiv(len, d0_256 / 2);
    int h_ = soft_sdiv(len_, d0_512 / 4);
    if (out_dtype == dlc_fp32) {
        i16ToF32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int32) {
        i16Toi32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int64) {
        i16Toi64_h(mem, dst, cmem, len, h, d0);
    } else if (out_dtype == dlc_fp16) {
        i16Tof16(mem, dst, len);
    } else if (out_dtype == dlc_bf16) {
        i16Tobf16(mem, dst, len);
    } else if (out_dtype == dlc_int8) {
        i16Toi32_h(mem, dst, len, h, d0);
        i32Toi8_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_bool) {
        i16Toi32_h(mem, dst, len, h, d0);
        i32Tobool_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_uint8) {
        i16Toi32_h(mem, dst, len, h, d0);
        i32Touint8_h(mem, dst, len_, h_, d0);
    }
}

inline void uint8ToAnother_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, int d0_256, int d0_512,
                             DLCType out_dtype) {
    int h = soft_sdiv(len, d0_512 / 4);
    int h_ = soft_sdiv(len_, d0_256 / 2);
    if (out_dtype == dlc_fp32) {
        uint8ToF32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int32 || out_dtype == dlc_int64) {
        uint8Toi32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_fp16) {
        uint8ToF32_h(mem, dst, len, h, d0);
        fp32Tofp16_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_bf16) {
        uint8ToF32_h(mem, dst, len, h, d0);
        fp32Tobf16_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_int16) {
        uint8Toi32_h(mem, dst, len, h, d0);
        i32Toi16_h(mem, len_, h_, d0);
    } else if (out_dtype == dlc_int8) { // 这里uint8转int8,实际上和boolToint8是一样的
        uint8Toi32_h(mem, dst, len, h, d0);
        i32Toi8_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_bool) {
        uint8Toi32_h(mem, dst, len, h, d0);
        i32Tobool_h(mem, dst, len_, h_, d0);
    }
}

inline void boolToAnother_h(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int len_, int d0, int d0_256, int d0_512,
                            DLCType out_dtype) {
    int h = soft_sdiv(len, d0_512 / 4);
    int h_ = soft_sdiv(len_, d0_256 / 2);
    if (out_dtype == dlc_fp32) {
        uint8ToF32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int32 || out_dtype == dlc_int64) {
        uint8Toi32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_fp16) {
        uint8ToF32_h(mem, dst, len, h, d0);
        fp32Tofp16_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_bf16) {
        uint8ToF32_h(mem, dst, len, h, d0);
        fp32Tobf16_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_int16) {
        uint8Toi32_h(mem, dst, len, h, d0);
        i32Toi16_h(mem, len_, h_, d0);
    } else if (out_dtype == dlc_int8) {
        i8Toi32_h(mem, dst, len, h, d0);
        i32Toi8_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_uint8) {
        i8Toi32_h(mem, dst, len, h, d0);
        i32Touint8_h(mem, dst, len, h, d0);
    }
}

inline void i8ToAnother_h(SIM_X86::tensor mem, SIM_X86::tensor dst, SIM_X86::tensor cmem, int len, int len_, int d0, int d0_256, int d0_512,
                          DLCType out_dtype) {
    int h = soft_sdiv(len, d0_512 / 4);
    int h_ = soft_sdiv(len_, d0_256 / 2);
    if (out_dtype == dlc_fp32) {
        i8ToF32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int32) {
        i8Toi32_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_int64) {
        i8Toi64_h(mem, dst, cmem, len, h, d0);
    } else if (out_dtype == dlc_fp16) {
        i8ToF32_h(mem, dst, len, h, d0);
        fp32Tofp16_h(mem, dst, len_, h_, d0);
    } else if (out_dtype == dlc_bf16) {
        i8ToF32_h(mem, dst, len, h, d0);
        f32ToBf16_h(mem, len_, h_, d0);
    } else if (out_dtype == dlc_int16) {
        i8Toi32_h(mem, dst, len, h, d0);
        i32Toi16_h(mem, len_, h_, d0);
    } else if (out_dtype == dlc_bool) {
        i8Toi32_h(mem, dst, len, h, d0);
        i32Tobool_h(mem, dst, len, h, d0);
    } else if (out_dtype == dlc_uint8) {
        i8Toi32_h(mem, dst, len, h, d0);
        i32Touint8_h(mem, dst, len, h, d0);
    }
}
