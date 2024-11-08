#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _H_X86ADD_H_X86_
#define _H_X86ADD_H_X86_

inline int8_128 __dlc_hadd(int8_128 a, int8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	vr0 = and.u32 vr10, r47;"
        "V1@(pr0)	vr1 = and.u32 vr11, r47;"
        "}"
        "{"
        "V0@(pr0)	vmsk7 = eq.s32 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr0 = mov.u32 r46;"
        "V1@(pr0)	vr1 = shl.u32 vr10, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr2 = shl.u32 vr11, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vmsk0 = carry.u32 vr1, vr2;"
        "V1@(pr0)	vr3 = shr.u32 vr10, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr1 = sel vmsk0 vr0, r48;"
        "V1@(pr0)	vr4 = shr.u32 vr11, r32;"
        "}"
        "{"
        "V0@(pr0)	vr3 = add.s32 vr3, vr4;"
        "V1@(pr0)	vr6 = and.u32 vr10, r54;"
        "}"
        "{"
        "V0@(pr0)	vr2 = and.u32 vr11, r54;"
        "V1@(pr0)	vr5 = add.s32 vr3, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 15;"
        "V0@(pr0)	vr1 = add.s32 vr6, vr2;"
        "V1@(pr0)	vr5 = shl.u32 vr5, r32;"
        "}"
        "{"
        "V0@(pr0)	vr1 = and.u32 vr1, r54;"
        "V1@(pr0)	vr1 = shr.u32 vr1, r48;"
        "}"
        "{"
        "V0@(pr0)	vr12 = or.u32 vr1, vr5;"
        "}"
        "{"
        "V1@(pr0)	vr13 = add.s32 vr10, vr11;"
        "}"
        "{"
        "V1@(pr0)	vr13 = shra.s32 vr13, r48;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk7 vr13, vr12;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr11", "vr3", "vr13", "vr10", "vr6", "vr12", "vmsk0", "vmsk7"
        );
    return result0;
}

#endif // _HADD_H_
