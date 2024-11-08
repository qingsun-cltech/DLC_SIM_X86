#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FLOAT2HALF_RN_H_X86_
#define _FLOAT2HALF_RN_H_X86_

inline short8_128 __dlc_float2half_rn(float8_128 a)
{
    short8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr1 = shr.u32 vr10, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "pseudo@0	@pseudo imm_2 = 32768;"
        "V0@(pr0)	vr10 = and.u32 vr10, r44;"
        "V1@(pr0)	vr1 = and.u32 vr1, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 127;"
        "pseudo@0	@pseudo imm_2 = 23;"
        "V0@(pr0)	vr3 = and.u32 vr10, r44;"
        "V1@(pr0)	vr2 = shr.u32 vr10, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 127;"
        "pseudo@0	@pseudo imm_2 = 8191;"
        "V0@(pr0)	vr2 = sub.s32 vr2, r32;"
        "V1@(pr0)	vr0 = and.u32 vr10, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 4096;"
        "pseudo@0	@pseudo imm_1 = 13;"
        "V0@(pr0)	vmsk1 = gt.s32 vr0, r32;"
        "V1@(pr0)	vr3 = shr.u32 vr3, r33;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 15;"
        "V1@(pr0)	vr2 = add.s32 vr2, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 10;"
        "V1@(pr0)	vr2 = shl.u32 vr2, r32;"
        "}"
        "{"
        "V0@(pr0)	vr4 = or.u32 vr3, vr2;"
        "V1@(pr0)	vr5 = and.u32 vr3, r48;"
        "}"
        "{"
        "V1@(pr0)	vr6 = add.s32 vr4, r48;"
        "}"
        "{"
        "V0@(pr0)	vr4 = sel vmsk1 vr4, vr6;"
        "V1@(pr0)	vmsk1 = eq.s32 vr5, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 4096;"
        "V0@(pr0)	vr5 = sel vmsk1 vr4, vr6;"
        "V1@(pr0)	vmsk1 = eq.s32 vr0, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 1024;"
        "V0@(pr0)	vr4 = sel vmsk1 vr4, vr5;"
        "V1@(pr0)	vmsk1 = ls.s32 vr2, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 30720;"
        "V0@(pr0)	vr4 = sel vmsk1 vr4, r46;"
        "V1@(pr0)	vmsk1 = gt.s32 vr2, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 31744;"
        "pseudo@0	@pseudo imm_1 = 32640;"
        "V0@(pr0)	vr4 = sel vmsk1 vr4, r32;"
        "V1@(pr0)	vmsk1 = gt.s32 vr10, r37;"
        "}"
        "{"
        "V0@(pr0)	vr4 = or.u32 vr1, vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32767;"
        "V0@(pr0)	vr4 = sel vmsk1 vr4, r32;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = mov.u32 vr4;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr3", "vr10", "vr6", "vmsk1"
        );
    return result0;
}

#endif // _FLOAT2HALF_RN_H_
