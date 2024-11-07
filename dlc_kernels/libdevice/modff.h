#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _MODFF_H_X86_
#define _MODFF_H_X86_

inline float2 __dlc_modff(float8_128 a)
{
    float8_128 result0;
    float8_128 result1;
    float2 res;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vr2 = and.u32 vr10, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr10 = and.u32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	vr11 = cvtftoint.s32 vr10, r56;"
        "}"
        "{"
        "V0@(pr0)	vr13 = cvtinttof.f32 vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 19200;"
        "V0@(pr0)	vmsk0 = gt.s32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	vr11 = sel vmsk0 vr13, vr10;"
        "}"
        "{"
        "V1@(pr0)	vr12 = sub.f32 vr10, vr11;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = infnan.f32 vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vr1 = mov.u32 r36;"
        "}"
        "{"
        "V0@(pr0)	vr11 = sel vmsk0 vr11, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr12 = sel vmsk0 vr12, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, r36;"
        "}"
        "{"
        "V0@(pr0)	vr11 = sel vmsk0 vr11, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vr12 = sel vmsk0 vr12, r36;"
        "}"
        "{"
        "V0@(pr0)	%[res1] = or.u32 vr11, vr2;"
        "V1@(pr0)	%[res0] = or.u32 vr12, vr2;"
        "}"
        : [res0] "=x" (result0), [res1] "=x" (result1)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr11", "vr13", "vr10", "vr12", "vmsk0"
        );
    res.x = result0;
    res.y = result1;
    return res;
}

#endif // _MODFF_H_
