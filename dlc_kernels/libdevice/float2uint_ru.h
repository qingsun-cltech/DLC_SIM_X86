#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FLOAT2UINT_RU_H_X86_
#define _FLOAT2UINT_RU_H_X86_

inline int8_128 __dlc_float2uint_ru(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vmsk1 = ls.f32 vr10, r46;"
        "V1@(pr0)	vr10 = sel vmsk1 vr10, r46;"
        "}"
        "{"
        "V0@(pr0)	vr0 = cvtftoint.s32 vr10, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 127;"
        "V0@(pr0)	vr3 = and.u32 vr10, r44;"
        "V1@(pr0)	vmsk1 = neq.s32 vr3, r46;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mov.u32 r46;"
        "V1@(pr0)	vr1 = sel vmsk1 vr2, r48;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = ls.f32 vr10, r49;"
        "V1@(pr0)	vr0 = sel vmsk1 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 128;"
        "pseudo@0	@pseudo imm_1 = 8;"
        "V0@(pr0)	vr3 = or.u32 vr3, r36;"
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
        :"vr2", "vr1", "vr0", "vr3", "vr10", "vmsk1"
        );
    return result0;
}

#endif // _FLOAT2UINT_RU_H_
