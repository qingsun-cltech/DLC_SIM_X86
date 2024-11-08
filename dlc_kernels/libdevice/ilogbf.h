#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ILOGBF_H_X86_
#define _ILOGBF_H_X86_

inline float8_128 __dlc_ilogbf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr0 = exte.s32 vr10;"
        "V1@(pr0)	vmsk0 = infnan.f32 vr10;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mov.u32 r47;"
        "V1@(pr0)	vr2 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 127;"
        "V0@(pr0)	vr3 = and.u32 vr10, r44;"
        "V1@(pr0)	vmsk1 = eq.s32 vr3, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vr4 = and.u32 vr10, r36;"
        "V1@(pr0)	vmsk2 = eq.s32 vr4, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65408;"
        "V0@(pr0)	vr5 = mov.u32 r40;"
        "V1@(pr0)	vr6 = sel vmsk2 vr2, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk1 vr5, vr2;"
        "V1@(pr0)	vr5 = and.u32 vr6, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk0 vr2, vr1;"
        "V1@(pr0)	vr7 = sel vmsk1 vr1, vr2;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr6 = and.u32 vr6, vr7;"
        "V1@(pr0)	vr7 = sel vmsk0 vr2, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr8 = sel vmsk1 vr2, r44;"
        "V1@(pr0)	vr7 = and.u32 vr7, vr8;"
        "}"
        "{"
        "V0@(pr0)	vr8 = sel vmsk2 vr2, r47;"
        "V1@(pr0)	vr9 = sel vmsk1 vr2, r47;"
        "}"
        "{"
        "V0@(pr0)	vr8 = and.u32 vr8, vr9;"
        "V1@(pr0)	vr5 = or.u32 vr5, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr5 = or.u32 vr5, vr7;"
        "V1@(pr0)	vr5 = or.u32 vr5, vr8;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr5, r46;"
        "V1@(pr0)	%[res0] = sel vmsk0 vr5, vr0;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr3", "vr7", "vr8", "vr10", "vr6", "vr9", "vmsk0", "vmsk2", "vmsk1"
        );
    return result0;
}

#endif // _ILOGBF_H_
