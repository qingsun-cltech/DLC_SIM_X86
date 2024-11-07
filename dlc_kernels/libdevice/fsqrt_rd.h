#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FSQRT_RD_H_X86_
#define _FSQRT_RD_H_X86_

inline float8_128 __dlc_fsqrt_rd(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr10;"
        "}"
        "{"
        "MTR@(pr0)	vr11 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rsqrt.f32 vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_1 = 0;"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, r44;"
        "V1@(pr0)	vr11 = sel vmsk0 vr11, r44;"
        "MTR@(pr0)	vr11 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, r46;"
        "V1@(pr0)	vr11 = sel vmsk0 vr11, r46;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, r47;"
        "V1@(pr0)	%[res0] = sel vmsk0 vr11, r47;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr11", "vmsk0"
        );
    return result0;
}

#endif // _FSQRT_RD_H_
