#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FMAF_H_X86_
#define _FMAF_H_X86_

inline float8_128 __dlc_fmaf(float8_128 a, float8_128 b, float8_128 c)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{V0@(pr0)  vr12 = mov.u32 %[input2];}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr10, r50;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr12, r50;"
        "V1@(pr0)	vmsk2 = gteq.s32 vr12, r46;"
        "}"
        "{"
        "V1@(pr0)	vr5 = exte.s32 vr3;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 126;"
        "V0@(pr0)	vmsk3 = neq.s32 vr5, r32;"
        "V1@(pr0)	vr4 = exte.s32 vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 127;"
        "V0@(pr0)	vmsk1 = neq.s32 vr4, r32;"
        "V1@(pr0)	vmsk3 = vor.f32 vmsk2, vmsk3;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr1, vr11;"
        "V1@(pr0)	vmsk1 = vor.f32 vmsk1, vmsk3;"
        "}"
        "{"
        "V0@(pr0)	vr10 = mul.f32 vr10, vr11;"
        "V1@(pr0)	vr1 = add.f32 vr1, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr1, r51;"
        "V1@(pr0)	vr12 = add.f32 vr10, vr12;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk1 vr1, r46;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr10, vr12;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b),  [input2] "x" (c)
        :"vr1", "vr5", "vr4", "vr3", "vr11", "vr10", "vr12", "vmsk1", "vmsk2", "vmsk3"
        );
    return result0;
}

#endif // _FMAF_H_
