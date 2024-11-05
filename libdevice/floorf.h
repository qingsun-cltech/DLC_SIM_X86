#ifndef _FLOORF_H_
#define _FLOORF_H_

inline float8_128 __dlc_floorf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vmsk0 = gt.f32 vr10, r46;"
        "V1@(pr0)	vr1 = cvtftoint.s32 vr10, r46;"
        "}"
        "{"
        "V0@(pr0)	vr2 = cvtftoint.s32 vr10, r56;"
        "V1@(pr0)	vr1 = sel vmsk0 vr1, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr0 = cvtinttof.f32 vr1;"
        "V1@(pr0)	vr3 = and.u32 vr10, r47;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_2 = 65535;"
        "pseudo@0	@pseudo imm_3 = 127;"
        "V0@(pr0)	vr1 = and.u32 vr10, r36;"
        "V1@(pr0)	vr2 = and.u32 vr10, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = eq.s32 vr1, r36;"
        "V1@(pr0)	vmsk1 = eq.s32 vr2, r46;"
        "}"
        "{"
        "V0@(pr0)	vmsk2 = eq.s32 vr3, r47;"
        "V1@(pr0)	vr4 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr5 = mov.u32 r44;"
        "V1@(pr0)	vr6 = sel vmsk0 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk1 vr5, vr4;"
        "V1@(pr0)	vr6 = and.u32 vr6, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 49024;"
        "V0@(pr0)	vmsk0 = eq.s32 vr1, r46;"
        "V1@(pr0)	vr5 = mov.u32 r36;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk0 vr4, vr5;"
        "V1@(pr0)	vr8 = sel vmsk1 vr5, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr9 = sel vmsk2 vr4, vr5;"
        "V1@(pr0)	vr7 = and.u32 vr7, vr8;"
        "}"
        "{"
        "V0@(pr0)	vr7 = and.u32 vr7, vr9;"
        "V1@(pr0)	vmsk2 = eq.s32 vr10, r47;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "pseudo@0	@pseudo imm_2 = 20224;"
        "V0@(pr0)	vr5 = and.u32 vr10, r44;"
        "V1@(pr0)	vmsk0 = gt.f32 vr5, r38;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr6, r46;"
        "V1@(pr0)	vr0 = sel vmsk1 vr6, vr0;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr7, r46;"
        "V1@(pr0)	vr0 = sel vmsk1 vr7, vr0;"
        "}"
        "{"
        "V0@(pr0)	vr0 = sel vmsk2 vr0, r47;"
        "V1@(pr0)	vr0 = sel vmsk0 vr0, vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 44928;"
        "V0@(pr0)	vmsk0 = gt.f32 vr10, r36;"
        "V1@(pr0)	vmsk1 = ls.f32 vr10, r47;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mov.u32 r47;"
        "V1@(pr0)	vr2 = sel vmsk0 vr1, r57;"
        "}"
        "{"
        "V0@(pr0)	vr3 = sel vmsk1 vr1, r57;"
        "V1@(pr0)	vr1 = and.u32 vr2, vr3;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr1, r47;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr1, vr0;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr3", "vr7", "vr8", "vr10", "vr6", "vr9", "vmsk0", "vmsk2", "vmsk1"
        );
    return result0;
}

#endif // _FLOORF_H_