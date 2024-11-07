#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FLOAT2INT_RN_H_X86_
#define _FLOAT2INT_RN_H_X86_

inline int8_128 __dlc_float2int_rn(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V1@(pr0)	vr11 = mov.u32 vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 65535;"
        "pseudo@0	@pseudo imm_3 = 32767;"
        "V1@(pr0)	vr10 = cvtftoint.s32 vr11, r45;"
        "}"
        "{"
        "V1@(pr0)	vmsk7 = gteq.f32 vr11, r46;"
        "}"
        "{"
        "V0@(pr0)	vr12 = cvtftoint.s32 vr11, r46;"
        "V1@(pr0)	vr13 = cvtftoint.s32 vr11, r56;"
        "}"
        "{"
        "V1@(pr0)	vr14 = sel vmsk7 vr12, vr13;"
        "}"
        "{"
        "V0@(pr0)	vr14 = cvtinttof.f32 vr14;"
        "V1@(pr0)	vr15 = sub.f32 vr11, vr14;"
        "}"
        "{"
        "V0@(pr0)	vmsk6 = eq.f32 vr15, r50;"
        "V1@(pr0)	vmsk7 = eq.f32 vr15, r58;"
        "}"
        "{"
        "V0@(pr0)	vmsk5 = vor.f32 vmsk6, vmsk7;"
        "}"
        "{"
        "V0@(pr0)	vr16 = and.u32 vr12, r48;"
        "V1@(pr0)	vmsk3 = eq.s32 vr16, r46;"
        "}"
        "{"
        "V1@(pr0)	vr17 = sel vmsk3 vr13, vr12;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk5 vr10, vr17;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr17", "vr16", "vr11", "vr14", "vr15", "vr13", "vr10", "vr12", "vmsk5", "vmsk6", "vmsk3", "vmsk7"
        );
    return result0;
}

#endif // _FLOAT2INT_RN_H_
