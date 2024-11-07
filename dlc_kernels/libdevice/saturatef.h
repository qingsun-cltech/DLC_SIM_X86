#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _SATURATEF_H_X86_
#define _SATURATEF_H_X86_

inline float8_128 __dlc_saturatef(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr9 = relux.f32 vr10, r49;"
        "V1@(pr0)	vmsk0 = infnan.f32 vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 127;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vmsk1 = neq.s32 vr0, r46;"
        "}"
        "{"
        "V0@(pr0)	vr0 = sel vmsk0 vr9, r46;"
        "V1@(pr0)	vr1 = sel vmsk1 vr9, r46;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = or.u32 vr0, vr1;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vr0", "vr9", "vmsk0", "vmsk1"
        );
    return result0;
}

#endif // _SATURATEF_H_
