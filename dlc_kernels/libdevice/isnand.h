#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ISNAND_H_X86_
#define _ISNAND_H_X86_

inline int8_128 __dlc_isnand(float8_128 a, float8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr10 = and.u32 vr10, r44;"
        "V1@(pr0)	vr0 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32752;"
        "V0@(pr0)	vmsk1 = eq.s32 vr10, r37;"
        "V1@(pr0)	vr1 = sel vmsk1 vr0, r48;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = neq.s32 vr11, r46;"
        "V1@(pr0)	vr2 = sel vmsk1 vr0, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32752;"
        "V0@(pr0)	vmsk1 = gt.s32 vr10, r37;"
        "V1@(pr0)	vr3 = sel vmsk1 vr0, r48;"
        "}"
        "{"
        "V0@(pr0)	vr4 = and.u32 vr1, vr2;"
        "V1@(pr0)	%[res0] = or.u32 vr4, vr3;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr4", "vr0", "vr11", "vr3", "vr10", "vmsk1"
        );
    return result0;
}

#endif // _ISNAND_H_
