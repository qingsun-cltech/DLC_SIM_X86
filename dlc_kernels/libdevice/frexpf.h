#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FREXPF_H_X86_
#define _FREXPF_H_X86_

inline float2 __dlc_frexpf(float8_128 a)
{
    float8_128 result0;
    int8_128 result1;
    float2 res;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vr11 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 128;"
        "V0@(pr0)	vmsk1 = ls.s32 vr0, r36;"
        "V1@(pr0)	vr1 = count.u32 vr0;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "V0@(pr0)	vr1 = sub.s32 vr1, r32;"
        "V1@(pr0)	vr2 = shl.u32 vr0, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr3 = and.u32 vr10, r47;"
        "V1@(pr0)	vr2 = or.u32 vr2, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk1 vr10, vr2;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65411;"
        "V0@(pr0)	vr3 = exte.s32 vr0;"
        "V1@(pr0)	vr2 = mov.u32 r40;"
        "}"
        "{"
        "V0@(pr0)	vr1 = sub.s32 vr2, vr1;"
        "V1@(pr0)	vr3 = add.s32 vr3, r48;"
        "}"
        "{"
        "V0@(pr0)	vr3 = sel vmsk1 vr3, vr1;"
        "V1@(pr0)	vr11 = add.s32 vr11, vr3;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32895;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	vr0 = or.u32 vr0, r50;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = infnan.f32 vr10;"
        "V1@(pr0)	vmsk1 = eq.f32 vr10, r46;"
        "}"
        "{"
        "V0@(pr0)	vr0 = sel vmsk0 vr0, vr10;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr0, vr10;"
        "}"
        "{"
        "V0@(pr0)	vr11 = sel vmsk0 vr11, r46;"
        "V1@(pr0)	%[res1] = sel vmsk1 vr11, r46;"
        "}"
        : [res0] "=x" (result0), [res1] "=x" (result1)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr0", "vr11", "vr3", "vr10", "vmsk1", "vmsk0"
        );
    res.x = result0;
    res.y = result1;
    return res;
}

#endif // _FREXPF_H_
