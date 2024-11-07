#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _EXPM1F_WITHOUT_UNARY_H_X86_
#define _EXPM1F_WITHOUT_UNARY_H_X86_

inline float8_128 __dlc_expm1f_without_unary(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr0 = and.u32 vr10, r44;"
        "V1@(pr0)	vr1 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mov.u32 r46;"
        "V1@(pr0)	vr3 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vr4 = and.u32 vr10, r47;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 29056;"
        "pseudo@0	@pseudo imm_1 = 16177;"
        "V0@(pr0)	vmsk0 = eq.s32 vr4, r47;"
        "V1@(pr0)	vr5 = mov.u32 r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 29056;"
        "pseudo@0	@pseudo imm_1 = 48945;"
        "pseudo@0	@pseudo imm_2 = 63441;"
        "pseudo@0	@pseudo imm_3 = 14103;"
        "V0@(pr0)	vr5 = sel vmsk0 vr5, r44;"
        "V1@(pr0)	vr6 = mov.u32 r45;"
        "}"
        "{"
        "V1@(pr0)	vr5 = sub.f32 vr10, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 63441;"
        "pseudo@0	@pseudo imm_1 = 46871;"
        "V0@(pr0)	vr6 = sel vmsk0 vr6, r44;"
        "V1@(pr0)	vr7 = mov.u32 r48;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk0 vr7, r56;"
        "V1@(pr0)	vr14 = mov.u32 r50;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 43579;"
        "pseudo@0	@pseudo imm_1 = 16312;"
        "V0@(pr0)	vr13 = mul.f32 vr10, r44;"
        "V1@(pr0)	vr14 = sel vmsk0 vr14, r58;"
        "}"
        "{"
        "V1@(pr0)	vr13 = add.f32 vr13, vr14;"
        "}"
        "{"
        "V0@(pr0)	vr14 = cvtftoint.s32 vr13, r56;"
        "}"
        "{"
        "V0@(pr0)	vr13 = cvtinttof.f32 vr14;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 29056;"
        "pseudo@0	@pseudo imm_1 = 16177;"
        "V0@(pr0)	vr15 = mul.f32 vr13, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 63441;"
        "pseudo@0	@pseudo imm_1 = 14103;"
        "V0@(pr0)	vr16 = mul.f32 vr13, r44;"
        "V1@(pr0)	vr15 = sub.f32 vr10, vr15;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 29208;"
        "pseudo@0	@pseudo imm_1 = 16049;"
        "pseudo@0	@pseudo imm_2 = 5522;"
        "pseudo@0	@pseudo imm_3 = 16261;"
        "V0@(pr0)	vmsk0 = gt.s32 vr0, r44;"
        "V1@(pr0)	vmsk1 = ls.s32 vr0, r45;"
        "}"
        "{"
        "V0@(pr0)	vr5 = sel vmsk1 vr15, vr5;"
        "V1@(pr0)	vr6 = sel vmsk1 vr16, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk1 vr14, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr1 = sel vmsk0 vr1, vr5;"
        "V1@(pr0)	vr2 = sel vmsk0 vr2, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr3 = sel vmsk0 vr3, vr7;"
        "V1@(pr0)	vr5 = sub.f32 vr1, vr2;"
        "}"
        "{"
        "V1@(pr0)	vr6 = sub.f32 vr1, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr5 = sel vmsk0 vr10, vr5;"
        "V1@(pr0)	vr6 = sub.f32 vr6, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr13 = mul.f32 vr5, r50;"
        "}"
        "{"
        "V0@(pr0)	vr12 = mul.f32 vr5, vr13;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 60859;"
        "pseudo@0	@pseudo imm_1 = 46167;"
        "pseudo@0	@pseudo imm_2 = 32340;"
        "pseudo@0	@pseudo imm_3 = 13958;"
        "V0@(pr0)	vr14 = mul.f32 vr12, r44;"
        "V1@(pr0)	vr14 = add.f32 vr14, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 28877;"
        "pseudo@0	@pseudo imm_3 = 47270;"
        "V0@(pr0)	vr14 = mul.f32 vr14, vr12;"
        "V1@(pr0)	vr14 = add.f32 vr14, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 3329;"
        "pseudo@0	@pseudo imm_3 = 15056;"
        "V0@(pr0)	vr14 = mul.f32 vr14, vr12;"
        "V1@(pr0)	vr14 = add.f32 vr14, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 34953;"
        "pseudo@0	@pseudo imm_3 = 48392;"
        "V0@(pr0)	vr14 = mul.f32 vr14, vr12;"
        "V1@(pr0)	vr14 = add.f32 vr14, r45;"
        "}"
        "{"
        "V0@(pr0)	vr14 = mul.f32 vr14, vr12;"
        "V1@(pr0)	vr14 = add.f32 vr14, r49;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16448;"
        "V0@(pr0)	vr16 = mul.f32 vr14, vr13;"
        "V1@(pr0)	vr15 = mov.u32 r36;"
        "}"
        "{"
        "V1@(pr0)	vr15 = sub.f32 vr15, vr16;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16576;"
        "V0@(pr0)	vr16 = mul.f32 vr5, vr15;"
        "V1@(pr0)	vr7 = mov.u32 r36;"
        "}"
        "{"
        "V1@(pr0)	vr7 = sub.f32 vr7, vr16;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "pseudo@0	@pseudo imm_2 = 32640;"
        "V0@(pr0)	vr11 = and.u32 vr7, r44;"
        "V1@(pr0)	vmsk1 = eq.s32 vr11, r38;"
        "}"
        "{"
        "V0@(pr0)	vr13 = mul.f32 vr11, r50;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23007;"
        "pseudo@0	@pseudo imm_1 = 24375;"
        "V0@(pr0)	vr16 = mov.u32 r44;"
        "V1@(pr0)	vr11 = shr.u32 vr11, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16320;"
        "V0@(pr0)	vr11 = sub.s32 vr16, vr11;"
        "V1@(pr0)	vr28 = mov.u32 r36;"
        "}"
        "{"
        "V0@(pr0)	vr16 = mul.f32 vr11, vr11;"
        "V1@(pr0)	vr30 = and.u32 vr7, r47;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr13, vr16;"
        "V1@(pr0)	vr29 = sub.f32 vr28, vr29;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr29, vr11;"
        "V1@(pr0)	vr11 = mov.u32 vr29;"
        "}"
        "{"
        "V0@(pr0)	vr16 = mul.f32 vr11, vr11;"
        "V1@(pr0)	vmsk0 = eq.f32 vr7, r46;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr13, vr16;"
        "V1@(pr0)	vr29 = sub.f32 vr28, vr29;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr29, vr11;"
        "V1@(pr0)	vr11 = mov.u32 vr29;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vr16 = mul.f32 vr11, vr11;"
        "V1@(pr0)	vr31 = or.u32 vr30, r36;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr13, vr16;"
        "V1@(pr0)	vr29 = sub.f32 vr28, vr29;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr29, vr11;"
        "V1@(pr0)	vr11 = or.u32 vr30, r46;"
        "}"
        "{"
        "V0@(pr0)	vr13 = mul.f32 vr29, vr29;"
        "V1@(pr0)	vr13 = or.u32 vr30, vr13;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk0 vr13, vr31;"
        "V1@(pr0)	vr7 = sel vmsk1 vr7, vr11;"
        "}"
        "{"
        "V1@(pr0)	vr16 = sub.f32 vr14, vr15;"
        "}"
        "{"
        "V0@(pr0)	vr7 = mul.f32 vr16, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr7 = mul.f32 vr12, vr7;"
        "}"
        "{"
        "V1@(pr0)	vr14 = sub.f32 vr7, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr14 = mul.f32 vr14, vr5;"
        "V1@(pr0)	vr14 = sub.f32 vr14, vr6;"
        "}"
        "{"
        "V1@(pr0)	vr14 = sub.f32 vr14, vr12;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23;"
        "pseudo@0	@pseudo imm_1 = 256;"
        "V0@(pr0)	vmsk0 = ls.s32 vr3, r32;"
        "V1@(pr0)	vr15 = mov.u32 r37;"
        "}"
        "{"
        "V0@(pr0)	vr16 = mov.u32 r49;"
        "V1@(pr0)	vr15 = shr.u32 vr15, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr15 = sub.s32 vr16, vr15;"
        "V1@(pr0)	vr16 = sub.f32 vr14, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23;"
        "V1@(pr0)	vr17 = shl.u32 vr3, r32;"
        "}"
        "{"
        "V1@(pr0)	vr16 = sub.f32 vr15, vr16;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 127;"
        "V0@(pr0)	vr16 = add.s32 vr16, vr17;"
        "V1@(pr0)	vr15 = mov.u32 r32;"
        "}"
        "{"
        "V0@(pr0)	vr15 = sub.s32 vr15, vr3;"
        "}"
        "{"
        "V1@(pr0)	vr15 = shl.u32 vr15, r32;"
        "}"
        "{"
        "V1@(pr0)	vr15 = add.f32 vr14, vr15;"
        "}"
        "{"
        "V1@(pr0)	vr28 = sub.f32 vr5, vr15;"
        "}"
        "{"
        "V1@(pr0)	vr28 = add.f32 vr28, r49;"
        "}"
        "{"
        "V0@(pr0)	vr28 = add.s32 vr28, vr17;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr28, vr16;"
        "}"
        "{"
        "V0@(pr0)	vr16 = mov.u32 r49;"
        "V1@(pr0)	vr15 = sub.f32 vr14, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65534;"
        "V0@(pr0)	vmsk0 = gt.s32 vr3, r40;"
        "V1@(pr0)	vr16 = sub.f32 vr16, vr15;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 56;"
        "V0@(pr0)	vr16 = add.s32 vr16, vr17;"
        "V1@(pr0)	vmsk1 = lseq.s32 vr3, r32;"
        "}"
        "{"
        "V1@(pr0)	vr16 = sub.f32 vr16, r49;"
        "}"
        "{"
        "V0@(pr0)	vr28 = sel vmsk0 vr16, r46;"
        "V1@(pr0)	vr16 = sel vmsk1 vr16, r46;"
        "}"
        "{"
        "V0@(pr0)	vr16 = add.s32 vr16, vr28;"
        "V1@(pr0)	vr28 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr28, vr13;"
        "V1@(pr0)	vr13 = sel vmsk1 vr28, vr13;"
        "}"
        "{"
        "V0@(pr0)	vr13 = add.s32 vr13, vr16;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 48768;"
        "V0@(pr0)	vmsk0 = ls.f32 vr5, r36;"
        "V1@(pr0)	vr16 = add.f32 vr5, r50;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr3, r48;"
        "V1@(pr0)	vr16 = sub.f32 vr14, vr16;"
        "}"
        "{"
        "V0@(pr0)	vr16 = mul.f32 vr16, r59;"
        "V1@(pr0)	vr28 = sub.f32 vr5, vr14;"
        "}"
        "{"
        "V0@(pr0)	vr28 = mul.f32 vr28, r51;"
        "V1@(pr0)	vr28 = add.f32 vr28, r49;"
        "}"
        "{"
        "V0@(pr0)	vr16 = sel vmsk0 vr28, vr16;"
        "V1@(pr0)	vr13 = sel vmsk1 vr13, vr16;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr3, r56;"
        "V1@(pr0)	vr16 = sub.f32 vr5, vr14;"
        "}"
        "{"
        "V0@(pr0)	vr16 = mul.f32 vr16, r50;"
        "V1@(pr0)	vr16 = sub.f32 vr16, r50;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr13, vr16;"
        "}"
        "{"
        "V0@(pr0)	vr16 = mul.f32 vr5, vr7;"
        "V1@(pr0)	vmsk0 = eq.s32 vr3, r46;"
        "}"
        "{"
        "V1@(pr0)	vr16 = sub.f32 vr16, vr12;"
        "}"
        "{"
        "V1@(pr0)	vr16 = sub.f32 vr5, vr16;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr13, vr16;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 13056;"
        "V0@(pr0)	vmsk0 = ls.s32 vr0, r36;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr13, vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 29207;"
        "pseudo@0	@pseudo imm_3 = 17073;"
        "V0@(pr0)	vmsk3 = ls.s32 vr10, r46;"
        "V1@(pr0)	vmsk2 = gt.s32 vr10, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 47172;"
        "pseudo@0	@pseudo imm_1 = 16789;"
        "V0@(pr0)	vr12 = sel vmsk3 vr13, r57;"
        "V1@(pr0)	vmsk1 = gteq.s32 vr0, r44;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk1 vr13, vr12;"
        "V1@(pr0)	vmsk0 = infnan.f32 vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_1 = 65408;"
        "V0@(pr0)	vr13 = sel vmsk2 vr13, r36;"
        "V1@(pr0)	vmsk1 = eq.s32 vr10, r37;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vr6 = sel vmsk0 vr6, r36;"
        "V1@(pr0)	vr7 = and.u32 vr10, r47;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vr7 = or.u32 vr6, vr7;"
        "V1@(pr0)	vmsk2 = eq.s32 vr10, r36;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr13, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk1 vr13, r57;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	%[res0] = sel vmsk2 vr13, r36;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr4", "vr30", "vr10", "vr6", "vr12", "vr7", "vr15", "vr31", "vr1", "vr5", "vr0", "vr28", "vr13", "vr29", "vr2", "vr17", "vr16", "vr3", "vr14", "vr11", "vmsk0", "vmsk3", "vmsk2", "vmsk1"
        );
    return result0;
}

#endif // _EXPM1F_WITHOUT_UNARY_H_