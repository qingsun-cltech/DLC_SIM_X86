#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FABSF_H_X86_
#define _FABSF_H_X86_

inline float8_128 __dlc_fabsf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	%[res0] = and.u32 vr10, r44;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _FABSF_H_
