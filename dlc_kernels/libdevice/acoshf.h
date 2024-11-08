#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ACOSHF_H_X86_
#define _ACOSHF_H_X86_

inline float8_128 __dlc_acoshf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr10, vr10;"
        "V1@(pr0)	vr2 = sub.f32 vr1, r49;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 19840;"
        "V0@(pr0)	(urf) = rcp.f32 vr2;"
        "V1@(pr0)	vmsk0 = gt.f32 vr10, r36;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = gt.f32 vr10, r51;"
        "V1@(pr0)	vmsk2 = ls.f32 vr10, r49;"
        "MTR@(pr0)	vr2 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rsqrt.f32 vr2;"
        "V1@(pr0)	vmsk3 = infnan.f32 vr10;"
        "}"
        "{"
        "V1@(pr0)	vmsk4 = eq.f32 vr10, r49;"
        "MTR@(pr0)	vr2 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vr4 = mul.f32 vr10, r51;"
        "V1@(pr0)	vr3 = add.f32 vr10, vr2;"
        "}"
        "{"
        "V1@(pr0)	vr2 = add.f32 vr2, vr10;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr2;"
        "}"
        "{"
        "MTR@(pr0)	vr2 = pop urf;"
        "}"
        "{"
        "V1@(pr0)	vr2 = sub.f32 vr4, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr17 = sel vmsk1 vr3, vr2;"
        "V1@(pr0)	vr17 = sel vmsk0 vr17, vr10;"
        "}"
        "{"
        "V0@(pr0)	(urf) = log.f32 vr17;"
        "}"
        "{"
        "MTR@(pr0)	vr1 = pop urf;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 29208;"
        "pseudo@0	@pseudo imm_1 = 16177;"
        "V0@(pr0)	vr17 = mul.f32 vr1, r44;"
        "V1@(pr0)	vr9 = add.f32 vr17, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65408;"
        "V0@(pr0)	vr11 = sel vmsk0 vr17, vr9;"
        "V1@(pr0)	vr11 = sel vmsk2 vr11, r36;"
        "}"
        "{"
        "V0@(pr0)	vr11 = sel vmsk3 vr11, vr10;"
        "V1@(pr0)	vr11 = sel vmsk4 vr11, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vmsk0 = ls.f32 vr10, r49;"
        "V1@(pr0)	%[res0] = sel vmsk0 vr11, r36;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr4", "vr1", "vr17", "vr3", "vr11", "vr10", "vr9", "vmsk3", "vmsk0", "vmsk4", "vmsk1", "vmsk2"
        );
    return result0;
}

#endif // _ACOSHF_H_
