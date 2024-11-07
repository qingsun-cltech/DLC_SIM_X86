#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FLOAT2UINT_RZ_H_X86_
#define _FLOAT2UINT_RZ_H_X86_

inline int8_128 __dlc_float2uint_rz(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vmsk1 = ls.f32 vr10, r46;"
        "V1@(pr0)	vr10 = sel vmsk1 vr10, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 65535;"
        "V0@(pr0)	vr0 = cvtftoint.s32 vr10, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 127;"
        "pseudo@0	@pseudo imm_2 = 128;"
        "V0@(pr0)	vr3 = and.u32 vr10, r44;"
        "V1@(pr0)	vr3 = or.u32 vr3, r38;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 8;"
        "V1@(pr0)	vr1 = shl.u32 vr3, r33;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20224;"
        "V0@(pr0)	vmsk1 = gteq.s32 vr10, r36;"
        "V1@(pr0)	vr1 = sel vmsk1 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20352;"
        "V0@(pr0)	vmsk1 = ls.s32 vr10, r36;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr0, vr1;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vr3", "vr0", "vmsk1"
        );
    return result0;
}

#endif // _FLOAT2UINT_RZ_H_
