#ifndef _HALF2FLOAT_H_
#define _HALF2FLOAT_H_

inline float8_128 __dlc_half2float(short8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr10 = and.u32 vr10, r54;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 1023;"
        "pseudo@0	@pseudo imm_1 = 31744;"
        "V0@(pr0)	vr0 = and.u32 vr10, r32;"
        "V1@(pr0)	vr1 = and.u32 vr10, r33;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr0, r46;"
        "V1@(pr0)	vmsk1 = eq.s32 vr1, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 31744;"
        "pseudo@0	@pseudo imm_1 = 32768;"
        "V0@(pr0)	vmsk2 = eq.s32 vr1, r32;"
        "V1@(pr0)	vr2 = and.u32 vr10, r33;"
        "}"
        "{"
        "V0@(pr0)	vmsk3 = eq.s32 vr2, r46;"
        "V1@(pr0)	vr3 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_1 = 32704;"
        "V0@(pr0)	vr4 = mov.u32 r37;"
        "V1@(pr0)	vr5 = mov.u32 r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "pseudo@0	@pseudo imm_1 = 16;"
        "V0@(pr0)	vr6 = and.u32 vr10, r32;"
        "V1@(pr0)	vr6 = shl.u32 vr6, r33;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32767;"
        "pseudo@0	@pseudo imm_1 = 13;"
        "V0@(pr0)	vr7 = and.u32 vr10, r32;"
        "V1@(pr0)	vr7 = shl.u32 vr7, r33;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 14336;"
        "V0@(pr0)	vr7 = add.s32 vr7, r44;"
        "V1@(pr0)	vr6 = or.u32 vr6, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr7 = count.u32 vr0;"
        "V1@(pr0)	vr7 = add.s32 vr7, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 22;"
        "V0@(pr0)	vr8 = sub.s32 vr7, r32;"
        "V1@(pr0)	vr7 = shl.u32 vr0, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 113;"
        "V0@(pr0)	vr9 = mov.u32 r32;"
        "V1@(pr0)	vr8 = sub.s32 vr9, vr8;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 9;"
        "pseudo@0	@pseudo imm_1 = 32768;"
        "V0@(pr0)	vmsk7 = eq.s32 vr10, r33;"
        "V1@(pr0)	vr7 = shr.u32 vr7, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23;"
        "V1@(pr0)	vr8 = shl.u32 vr8, r32;"
        "}"
        "{"
        "V0@(pr0)	vr0 = or.u32 vr7, vr8;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk0 vr4, vr3;"
        "V1@(pr0)	vr8 = sel vmsk2 vr3, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr7 = and.u32 vr7, vr8;"
        "V1@(pr0)	vmsk4 = eq.s32 vr7, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr8 = sel vmsk0 vr3, vr5;"
        "V1@(pr0)	vr9 = sel vmsk2 vr3, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr8 = and.u32 vr8, vr9;"
        "V1@(pr0)	vmsk5 = eq.s32 vr8, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 16;"
        "V0@(pr0)	vr1 = sel vmsk5 vr3, vr2;"
        "V1@(pr0)	vr1 = shl.u32 vr1, r33;"
        "}"
        "{"
        "V0@(pr0)	vr8 = or.u32 vr8, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr9 = sel vmsk1 vr6, vr3;"
        "V1@(pr0)	vr10 = sel vmsk2 vr6, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr9 = and.u32 vr9, vr10;"
        "V1@(pr0)	vmsk6 = eq.s32 vr9, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk0 vr0, vr3;"
        "V1@(pr0)	vr11 = sel vmsk1 vr3, vr0;"
        "}"
        "{"
        "V0@(pr0)	vr10 = and.u32 vr10, vr11;"
        "V1@(pr0)	vmsk6 = eq.s32 vr10, vr0;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 16;"
        "V0@(pr0)	vr1 = sel vmsk6 vr3, vr2;"
        "V1@(pr0)	vr1 = shl.u32 vr1, r33;"
        "}"
        "{"
        "V0@(pr0)	vr10 = or.u32 vr10, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr11 = or.u32 vr7, vr8;"
        "V1@(pr0)	vr11 = or.u32 vr11, vr9;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vr10 = or.u32 vr11, vr10;"
        "V1@(pr0)	%[res0] = sel vmsk7 vr10, r36;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr3", "vr7", "vr11", "vr8", "vr10", "vr6", "vr9", "vmsk3", "vmsk0", "vmsk5", "vmsk4", "vmsk7", "vmsk1", "vmsk2", "vmsk6"
        );
    return result0;
}

#endif // _HALF2FLOAT_H_
