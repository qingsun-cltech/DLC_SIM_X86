#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ISINFF_H_X86_
#define _ISINFF_H_X86_

inline int8_128 __dlc_isinff(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr10 = and.u32 vr10, r44;"
        "V1@(pr0)	vr0 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_1 = 65535;"
        "V0@(pr0)	vmsk1 = eq.f32 vr10, r36;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr0, r41;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr0", "vmsk1"
        );
    return result0;
}

#endif // _ISINFF_H_
