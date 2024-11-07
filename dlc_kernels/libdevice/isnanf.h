#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ISNANF_H_X86_
#define _ISNANF_H_X86_

inline int8_128 __dlc_isnanf(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vr1 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = gt.s32 vr0, r36;"
        "V1@(pr0)	%[res0] = sel vmsk0 vr1, r48;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vr0", "vmsk0"
        );
    return result0;
}

#endif // _ISNANF_H_
