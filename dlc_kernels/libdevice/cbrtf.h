#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _CBRTF_H_X86_
#define _CBRTF_H_X86_

inline float8_128 __dlc_cbrtf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr5 = and.u32 vr10, r47;"
        "V1@(pr0)	vr10 = and.u32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	(urf) = log.f32 vr10;"
        "}"
        "{"
        "MTR@(pr0)      vr2 = pop urf;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 43691;"
        "pseudo@0	@pseudo imm_1 = 16042;"
        "V0@(pr0)	vr10 = mul.f32 vr2, r44;"
        "}"
        "{"
        "V0@(pr0)	(urf) = pow.f32 vr10;"
        "}"
        "{"
        "MTR@(pr0)      vr10 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = or.u32 vr10, vr5;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr10", "vr5"
        );
    return result0;
}

#endif // _CBRTF_H_
