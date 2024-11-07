#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ROUNDF_H_X86_
#define _ROUNDF_H_X86_

inline float8_128 __dlc_roundf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_2 = 32;"
        "VL@(pr0)	vr11 = ld [vmem:0+0,0,0];"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr0 = and.u32 vr10, r47;"
        "V1@(pr0)	vr1 = cvtftoint.s32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	vr1 = cvtinttof.f32 vr1;"
        "V1@(pr0)	vr0 = or.u32 vr1, vr0;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 127;"
        "V0@(pr0)	vmsk0 = infnan.f32 vr10;"
        "V1@(pr0)	vr1 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vmsk1 = eq.s32 vr1, r46;"
        "V1@(pr0)	vr2 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20224;"
        "pseudo@0	@pseudo imm_2 = 65535;"
        "pseudo@0	@pseudo imm_3 = 32767;"
        "V0@(pr0)	vmsk2 = gt.f32 vr2, r36;"
        "V1@(pr0)	vr3 = mov.u32 r45;"
        "}"
        "{"
        "V0@(pr0)	vr4 = mov.u32 r46;"
        "V1@(pr0)	vr5 = sel vmsk0 vr4, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk1 vr3, vr4;"
        "V1@(pr0)	vr3 = and.u32 vr6, vr5;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr3, r46;"
        "V1@(pr0)	vr0 = sel vmsk0 vr3, vr0;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk2 vr0, vr10;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr11", "vr3", "vr10", "vr6", "vmsk0", "vmsk2", "vmsk1"
        );
    return result0;
}

#endif // _ROUNDF_H_
