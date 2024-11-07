#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _COPYSIGNF_H_X86_
#define _COPYSIGNF_H_X86_

inline float8_128 __dlc_copysignf(float8_128 a, float8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = and.u32 vr11, r47;"
        "V1@(pr0)	vr2 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32640;"
        "V0@(pr0)	vmsk0 = gt.s32 vr2, r37;"
        "V1@(pr0)	vr10 = or.u32 vr1, vr2;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32704;"
        "V0@(pr0)	%[res0] = sel vmsk0 vr10, r37;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr10", "vr1", "vr11", "vmsk0"
        );
    return result0;
}

#endif // _COPYSIGNF_H_
