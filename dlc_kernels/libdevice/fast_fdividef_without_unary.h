#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FAST_FDIVIDEF_WITHOUT_UNARY_H_X86_
#define _FAST_FDIVIDEF_WITHOUT_UNARY_H_X86_

inline float8_128 __dlc_fast_fdividef_without_unary(float8_128 a, float8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vr13 = mov.u32 vr11;"
        "V1@(pr0)	vr31 = mov.u32 r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "pseudo@0	@pseudo imm_2 = 32640;"
        "V0@(pr0)	vr0 = and.u32 vr11, r44;"
        "V1@(pr0)	vmsk1 = eq.s32 vr0, r38;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr0, r50;"
        "V1@(pr0)	vr2 = mov.u32 vr0;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23007;"
        "pseudo@0	@pseudo imm_1 = 24375;"
        "V0@(pr0)	vr3 = mov.u32 r44;"
        "V1@(pr0)	vr2 = shr.u32 vr2, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16320;"
        "V0@(pr0)	vr2 = sub.s32 vr3, vr2;"
        "V1@(pr0)	vr4 = mov.u32 r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vr3 = mul.f32 vr2, vr2;"
        "V1@(pr0)	vr6 = and.u32 vr11, r36;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr1, vr3;"
        "V1@(pr0)	vr5 = sub.f32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr5, vr2;"
        "V1@(pr0)	vr2 = mov.u32 vr5;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr2, vr2;"
        "V1@(pr0)	vmsk0 = eq.f32 vr11, r46;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr1, vr3;"
        "V1@(pr0)	vr5 = sub.f32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr5, vr2;"
        "V1@(pr0)	vr2 = mov.u32 vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vr3 = mul.f32 vr2, vr2;"
        "V1@(pr0)	vr7 = or.u32 vr6, r36;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr1, vr3;"
        "V1@(pr0)	vr5 = sub.f32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr5, vr2;"
        "V1@(pr0)	vr2 = or.u32 vr6, r46;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr5, vr5;"
        "V1@(pr0)	vr1 = or.u32 vr6, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr11 = sel vmsk0 vr1, vr7;"
        "V1@(pr0)	vr11 = sel vmsk1 vr11, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mul.f32 vr10, vr11;"
        "V1@(pr0)	vr30 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 32384;"
        "pseudo@0	@pseudo imm_2 = 0;"
        "pseudo@0	@pseudo imm_3 = 32640;"
        "V0@(pr0)	vmsk0 = gt.f32 vr13, r45;"
        "V1@(pr0)	vmsk1 = ls.f32 vr13, r44;"
        "}"
        "{"
        "V0@(pr0)	vr3 = sel vmsk0 vr31, vr11;"
        "V1@(pr0)	vr1 = sel vmsk0 vr30, vr11;"
        "}"
        "{"
        "V0@(pr0)	vr2 = sel vmsk1 vr1, vr11;"
        "V1@(pr0)	vr4 = sel vmsk1 vr3, vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr5 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = eq.s32 vr5, r36;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk0 vr2, vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = gt.s32 vr0, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	%[res0] = sel vmsk0 vr10, r36;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr11", "vr3", "vr7", "vr30", "vr13", "vr31", "vr10", "vr6", "vmsk1", "vmsk0"
        );
    return result0;
}

#endif // _FAST_FDIVIDEF_WITHOUT_UNARY_H_
