#ifndef _FAST_COSF_H_
#define _FAST_COSF_H_

inline float8_128 __dlc_fast_cosf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 63875;"
        "pseudo@0	@pseudo imm_1 = 16162;"
        "V0@(pr0)	vr15 = mul.f32 vr1, r44;"
        "}"
        "{"
        "V0@(pr0)	vr15 = cvtftoint.s32 vr15, r56;"
        "}"
        "{"
        "V0@(pr0)	vr17 = cvtinttof.f32 vr15;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 4059;"
        "pseudo@0	@pseudo imm_1 = 16329;"
        "V0@(pr0)	vr16 = mul.f32 vr17, r44;"
        "}"
        "{"
        "V1@(pr0)	vr16 = sub.f32 vr1, vr16;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mov.u32 vr16;"
        "}"
        "{"
        "V0@(pr0)	vr12 = mov.u32 r46;"
        "V1@(pr0)	vr13 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 3;"
        "V0@(pr0)	vr15 = and.u32 vr15, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr0 = and.u32 vr11, r44;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr11, vr11;"
        "V1@(pr0)	vr7 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vr4 = mul.f32 vr3, vr11;"
        "V1@(pr0)	vmsk1 = eq.s32 vr13, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 51667;"
        "pseudo@0	@pseudo imm_1 = 12078;"
        "pseudo@0	@pseudo imm_2 = 12084;"
        "pseudo@0	@pseudo imm_3 = 45783;"
        "V0@(pr0)	vr1 = mul.f32 vr3, r44;"
        "V1@(pr0)	vr1 = add.f32 vr1, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 61211;"
        "pseudo@0	@pseudo imm_1 = 13880;"
        "V0@(pr0)	vr1 = mul.f32 vr3, vr1;"
        "V1@(pr0)	vr1 = add.f32 vr1, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 3329;"
        "pseudo@0	@pseudo imm_1 = 47440;"
        "V0@(pr0)	vr1 = mul.f32 vr3, vr1;"
        "V1@(pr0)	vr1 = add.f32 vr1, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 34953;"
        "pseudo@0	@pseudo imm_1 = 15368;"
        "V0@(pr0)	vr1 = mul.f32 vr3, vr1;"
        "V1@(pr0)	vr5 = add.f32 vr1, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 43691;"
        "pseudo@0	@pseudo imm_1 = 48682;"
        "V0@(pr0)	vr1 = mul.f32 vr3, vr5;"
        "V1@(pr0)	vr1 = add.f32 vr1, r44;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr4, vr1;"
        "V1@(pr0)	vr1 = add.f32 vr11, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr12, r50;"
        "V1@(pr0)	vr6 = sel vmsk1 vr6, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr1 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr3, vr1;"
        "V1@(pr0)	vr1 = sub.f32 vr1, vr12;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 43691;"
        "pseudo@0	@pseudo imm_1 = 48682;"
        "V0@(pr0)	vr2 = mul.f32 vr4, r44;"
        "V1@(pr0)	vr1 = sub.f32 vr1, vr2;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr2 = cvtftoint.s32 vr11, r44;"
        "V1@(pr0)	vr1 = sub.f32 vr11, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 12800;"
        "V0@(pr0)	vr6 = sel vmsk1 vr1, vr6;"
        "V1@(pr0)	vmsk1 = ls.s32 vr0, r36;"
        "}"
        "{"
        "V0@(pr0)	vr1 = sel vmsk1 vr6, vr11;"
        "V1@(pr0)	vmsk1 = eq.s32 vr2, r46;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk1 vr6, vr1;"
        "V1@(pr0)	vr11 = mov.u32 vr6;"
        "}"
        "{"
        "V0@(pr0)	vr17 = mul.f32 vr11, r57;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 1;"
        "V1@(pr0)	vmsk1 = eq.s32 vr15, r32;"
        "}"
        "{"
        "V0@(pr0)	vr14 = sel vmsk1 vr11, vr17;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mov.u32 vr16;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr4 = mul.f32 vr11, vr11;"
        "V1@(pr0)	vr7 = and.u32 vr11, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 39322;"
        "pseudo@0	@pseudo imm_1 = 16025;"
        "V0@(pr0)	vr0 = mov.u32 r46;"
        "V1@(pr0)	vmsk1 = ls.s32 vr7, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 55118;"
        "pseudo@0	@pseudo imm_1 = 44359;"
        "pseudo@0	@pseudo imm_2 = 29942;"
        "pseudo@0	@pseudo imm_3 = 12559;"
        "V0@(pr0)	vr5 = mul.f32 vr4, r44;"
        "V1@(pr0)	vr5 = add.f32 vr5, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 62076;"
        "pseudo@0	@pseudo imm_1 = 46227;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 3329;"
        "pseudo@0	@pseudo imm_1 = 14288;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 2913;"
        "pseudo@0	@pseudo imm_1 = 47798;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 43691;"
        "pseudo@0	@pseudo imm_1 = 15658;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr11, vr12;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr1 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr4, r50;"
        "V1@(pr0)	vr1 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mov.u32 r49;"
        "V1@(pr0)	vr1 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk1 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16200;"
        "pseudo@0	@pseudo imm_1 = 256;"
        "V0@(pr0)	vmsk2 = gt.s32 vr7, r36;"
        "V1@(pr0)	vr1 = sub.s32 vr7, r37;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 16016;"
        "V0@(pr0)	vr2 = mul.f32 vr4, r50;"
        "V1@(pr0)	vr1 = sel vmsk2 vr1, r44;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mov.u32 r49;"
        "V1@(pr0)	vr2 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V1@(pr0)	vr1 = sub.f32 vr3, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr11, vr12;"
        "V1@(pr0)	vr3 = add.f32 vr3, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr3 = sub.f32 vr3, vr2;"
        "}"
        "{"
        "V1@(pr0)	vr1 = sub.f32 vr1, vr3;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 12800;"
        "V0@(pr0)	vr6 = sel vmsk1 vr1, vr6;"
        "V1@(pr0)	vmsk1 = ls.s32 vr7, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr2 = cvtftoint.s32 vr11, r44;"
        "}"
        "{"
        "V0@(pr0)	vr1 = sel vmsk1 vr6, r49;"
        "V1@(pr0)	vmsk1 = eq.s32 vr2, r46;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk1 vr6, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mov.u32 vr6;"
        "}"
        "{"
        "V0@(pr0)	vr17 = mul.f32 vr11, r57;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "V1@(pr0)	vmsk1 = eq.s32 vr15, r32;"
        "}"
        "{"
        "V0@(pr0)	vr14 = sel vmsk1 vr14, vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 2;"
        "V1@(pr0)	vmsk1 = eq.s32 vr15, r32;"
        "}"
        "{"
        "V0@(pr0)	vr14 = sel vmsk1 vr14, vr17;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mov.u32 vr1;"
        "}"
        "{"
        "V0@(pr0)	vr12 = mov.u32 r46;"
        "V1@(pr0)	vr13 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr4 = mul.f32 vr11, vr11;"
        "V1@(pr0)	vr7 = and.u32 vr11, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 39322;"
        "pseudo@0	@pseudo imm_1 = 16025;"
        "V0@(pr0)	vr0 = mov.u32 r46;"
        "V1@(pr0)	vmsk1 = ls.s32 vr7, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 55118;"
        "pseudo@0	@pseudo imm_1 = 44359;"
        "pseudo@0	@pseudo imm_2 = 29942;"
        "pseudo@0	@pseudo imm_3 = 12559;"
        "V0@(pr0)	vr5 = mul.f32 vr4, r44;"
        "V1@(pr0)	vr5 = add.f32 vr5, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 62076;"
        "pseudo@0	@pseudo imm_1 = 46227;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 3329;"
        "pseudo@0	@pseudo imm_1 = 14288;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 2913;"
        "pseudo@0	@pseudo imm_1 = 47798;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 43691;"
        "pseudo@0	@pseudo imm_1 = 15658;"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r44;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr1 = mul.f32 vr11, vr12;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr1 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr4, r50;"
        "V1@(pr0)	vr1 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mov.u32 r49;"
        "V1@(pr0)	vr1 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk1 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16200;"
        "pseudo@0	@pseudo imm_1 = 256;"
        "V0@(pr0)	vmsk2 = gt.s32 vr7, r36;"
        "V1@(pr0)	vr1 = sub.s32 vr7, r37;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 16016;"
        "V0@(pr0)	vr2 = mul.f32 vr4, r50;"
        "V1@(pr0)	vr1 = sel vmsk2 vr1, r44;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mov.u32 r49;"
        "V1@(pr0)	vr2 = sub.f32 vr2, vr1;"
        "}"
        "{"
        "V1@(pr0)	vr1 = sub.f32 vr3, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr11, vr12;"
        "V1@(pr0)	vr3 = add.f32 vr3, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr4, vr5;"
        "V1@(pr0)	vr3 = sub.f32 vr3, vr2;"
        "}"
        "{"
        "V1@(pr0)	vr1 = sub.f32 vr1, vr3;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 12800;"
        "V0@(pr0)	vr6 = sel vmsk1 vr1, vr6;"
        "V1@(pr0)	vmsk1 = ls.s32 vr7, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr2 = cvtftoint.s32 vr11, r44;"
        "}"
        "{"
        "V0@(pr0)	vr1 = sel vmsk1 vr6, r49;"
        "V1@(pr0)	vmsk1 = eq.s32 vr2, r46;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk1 vr6, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mov.u32 vr6;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 4056;"
        "pseudo@0	@pseudo imm_1 = 16201;"
        "V0@(pr0)	vmsk1 = ls.f32 vr1, r44;"
        "}"
        "{"
        "V1@(pr0)	vr14 = sel vmsk1 vr14, vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 32640;"
        "V0@(pr0)	vmsk1 = ls.f32 vr1, r44;"
        "V1@(pr0)	vr15 = sub.f32 vr10, vr10;"
        "}"
        "{"
        "V1@(pr0)	vr14 = sel vmsk1 vr15, vr14;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = mov.u32 vr14;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr17", "vr4", "vr5", "vr16", "vr0", "vr11", "vr3", "vr15", "vr7", "vr14", "vr13", "vr10", "vr6", "vr12", "vmsk1", "vmsk2"
        );
    return result0;
}

#endif // _FAST_COSF_H_