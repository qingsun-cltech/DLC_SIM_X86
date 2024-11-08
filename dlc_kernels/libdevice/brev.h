#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _BREV_H_X86_
#define _BREV_H_X86_

inline int8_128 __dlc_brev(int8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 43690;"
        "pseudo@0	@pseudo imm_1 = 43690;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vr0 = shr.u32 vr0, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 21845;"
        "pseudo@0	@pseudo imm_1 = 21845;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "V1@(pr0)	vr1 = shl.u32 vr1, r48;"
        "}"
        "{"
        "V0@(pr0)	vr10 = or.u32 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 52428;"
        "pseudo@0	@pseudo imm_1 = 52428;"
        "pseudo@0	@pseudo imm_2 = 2;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vr0 = shr.u32 vr0, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 13107;"
        "pseudo@0	@pseudo imm_1 = 13107;"
        "pseudo@0	@pseudo imm_2 = 2;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "V1@(pr0)	vr1 = shl.u32 vr1, r34;"
        "}"
        "{"
        "V0@(pr0)	vr10 = or.u32 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 61680;"
        "pseudo@0	@pseudo imm_1 = 61680;"
        "pseudo@0	@pseudo imm_2 = 4;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vr0 = shr.u32 vr0, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 3855;"
        "pseudo@0	@pseudo imm_1 = 3855;"
        "pseudo@0	@pseudo imm_2 = 4;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "V1@(pr0)	vr1 = shl.u32 vr1, r34;"
        "}"
        "{"
        "V0@(pr0)	vr10 = or.u32 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65280;"
        "pseudo@0	@pseudo imm_1 = 65280;"
        "pseudo@0	@pseudo imm_2 = 8;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vr0 = shr.u32 vr0, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 255;"
        "pseudo@0	@pseudo imm_1 = 255;"
        "pseudo@0	@pseudo imm_2 = 8;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "V1@(pr0)	vr1 = shl.u32 vr1, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 16;"
        "V0@(pr0)	vr10 = or.u32 vr0, vr1;"
        "V1@(pr0)	vr0 = shr.u32 vr10, r34;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 16;"
        "V1@(pr0)	vr1 = shl.u32 vr10, r34;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = or.u32 vr0, vr1;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vr0"
        );
    return result0;
}

#endif // _BREV_H_
