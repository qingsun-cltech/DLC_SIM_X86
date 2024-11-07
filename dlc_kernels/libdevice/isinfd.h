#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ISINFD_H_X86_
#define _ISINFD_H_X86_

inline int8_128 __dlc_isinfd(float8_128 a, float8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32752;"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, r37;"
        "V1@(pr0)	vmsk2 = eq.s32 vr11, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 65520;"
        "V0@(pr0)	vmsk1 = eq.s32 vr10, r37;"
        "V1@(pr0)	vr1 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vmsk3 = vor.f32 vmsk0, vmsk1;"
        "V1@(pr0)	vr9 = sel vmsk3 vr1, r48;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk2 vr1, r48;"
        "V1@(pr0)	%[res0] = and.u32 vr9, vr10;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr10", "vr1", "vr11", "vr9", "vmsk0", "vmsk2", "vmsk1", "vmsk3"
        );
    return result0;
}

#endif // _ISINFD_H_
