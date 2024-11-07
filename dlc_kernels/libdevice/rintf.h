#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _RINTF_H_X86_
#define _RINTF_H_X86_

inline float8_128 __dlc_rintf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = and.u32 vr10, r47;"
        "V1@(pr0)	vr0 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23;"
        "V1@(pr0)	vr2 = shr.u32 vr0, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 127;"
        "V1@(pr0)	vr2 = sub.s32 vr2, r32;"
        "}"
        "{"
        "V0@(pr0)	vr4 = cvtftoint.s32 vr0, r46;"
        "V1@(pr0)	vmsk1 = gt.f32 vr0, r46;"
        "}"
        "{"
        "V0@(pr0)	vr5 = cvtftoint.s32 vr0, r56;"
        "V1@(pr0)	vr4 = sel vmsk1 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr5 = and.u32 vr4, r48;"
        "V1@(pr0)	vr4 = cvtinttof.f32 vr4;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr5, r48;"
        "V1@(pr0)	vr7 = add.f32 vr4, r49;"
        "}"
        "{"
        "V0@(pr0)	vr5 = sel vmsk1 vr4, vr7;"
        "V1@(pr0)	vr6 = sub.f32 vr0, vr4;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr6, r50;"
        "V1@(pr0)	vr4 = sel vmsk1 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = gt.s32 vr6, r50;"
        "V1@(pr0)	vr4 = sel vmsk1 vr4, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23;"
        "V0@(pr0)	vmsk1 = gteq.s32 vr2, r32;"
        "V1@(pr0)	vr10 = sel vmsk1 vr4, vr0;"
        "}"
        "{"
        "V0@(pr0)	vr10 = or.u32 vr10, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_1 = 32704;"
        "V0@(pr0)	vmsk1 = gt.s32 vr0, r36;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr10, r37;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr4", "vr1", "vr5", "vr0", "vr7", "vr10", "vr6", "vmsk1"
        );
    return result0;
}

#endif // _RINTF_H_
