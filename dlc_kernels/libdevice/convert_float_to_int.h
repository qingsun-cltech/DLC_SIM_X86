#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _CONVERT_FLOAT_TO_INT_H_X86_
#define _CONVERT_FLOAT_TO_INT_H_X86_



inline int8_128 __dlc_float2int_rd(float8_128 a) {
    int8_128 result0;
    asm volatile("{ V0@(pr0)       vr10 = mov.u32 %[input]; }" : : [input] "x"(a) : "vr10");
    asm volatile("{"
                 "V0@(pr0)	vr10 = cvtftoint.s32 vr10, r56;"
                 "}"
                 :
                 :
                 : "vr10");

    asm volatile("{V0@(pr0)        %[res] = mov.u32 vr10;}" : [res] "=x"(result0) : : "vr10");

    return result0;
}

inline int8_128 __dlc_float2int_ru(float8_128 a) {
    int8_128 result0;
    asm volatile("{ V0@(pr0)       vr10 = mov.u32 %[input]; }" : : [input] "x"(a) : "vr10");
    asm volatile("{"
                 "V0@(pr0)	vr10 = cvtftoint.s32 vr10, r46;"
                 "}"
                 :
                 :
                 : "vr10");

    asm volatile("{V0@(pr0)        %[res] = mov.u32 vr10;}" : [res] "=x"(result0) : : "vr10");

    return result0;
}

inline int8_128 __dlc_float2int_rn(float8_128 a) {
    int8_128 result0;
    asm volatile("{ V0@(pr0)       vr10 = mov.u32 %[input]; }" : : [input] "x"(a) : "vr10");
    asm volatile("{"
                 "pseudo@0	@pseudo imm_0 = 65535;"
                 "pseudo@0	@pseudo imm_1 = 32767;"
                 "V0@(pr0)	vr10 = cvtftoint.s32 vr10, r44;"
                 "}"
                 :
                 :
                 : "vr10");

    asm volatile("{V0@(pr0)        %[res] = mov.u32 vr10;}" : [res] "=x"(result0) : : "vr10");

    return result0;
}

#endif