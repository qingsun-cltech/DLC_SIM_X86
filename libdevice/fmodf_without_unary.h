#ifndef _FMODF_WITHOUT_UNARY_H_
#define _FMODF_WITHOUT_UNARY_H_

inline float8_128 __dlc_fmodf_without_unary(float8_128 a, float8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	vmsk2 = eq.f32 vr11, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_1 = 65408;"
        "V0@(pr0)	vmsk3 = eq.f32 vr10, r36;"
        "V1@(pr0)	vmsk4 = eq.f32 vr10, r37;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "pseudo@0	@pseudo imm_1 = 65408;"
        "V0@(pr0)	vmsk5 = eq.f32 vr11, r36;"
        "V1@(pr0)	vmsk6 = eq.f32 vr11, r37;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "V1@(pr0)	vr31 = and.u32 vr11, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32640;"
        "V0@(pr0)	vmsk7 = gt.s32 vr1, r37;"
        "}"
        "{"
        "V0@(pr0)	vr12 = mov.u32 vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "pseudo@0	@pseudo imm_2 = 32640;"
        "V0@(pr0)	vr0 = and.u32 vr12, r44;"
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
        "V1@(pr0)	vr6 = and.u32 vr12, r36;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr1, vr3;"
        "V1@(pr0)	vr5 = sub.f32 vr4, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vr2 = mul.f32 vr5, vr2;"
        "V1@(pr0)	vr7 = or.u32 vr6, r36;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr2, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr1, vr3;"
        "V1@(pr0)	vr5 = sub.f32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr5, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr2, vr2;"
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
        "V0@(pr0)	vr12 = sel vmsk2 vr1, vr7;"
        "V1@(pr0)	vr12 = sel vmsk1 vr12, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr2 = mul.f32 vr10, vr12;"
        "V1@(pr0)	vr5 = mov.u32 vr10;"
        "}"
        "{"
        "V0@(pr0)	vr3 = cvtftoint.s32 vr2, r56;"
        "V1@(pr0)	vr4 = cvtinttof.f32 vr3;"
        "}"
        "{"
        "V0@(pr0)	vr6 = mul.f32 vr4, vr11;"
        "V1@(pr0)	vr10 = sub.f32 vr5, vr6;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr14 = and.u32 vr10, r44;"
        "V1@(pr0)	vr15 = and.u32 vr11, r44;"
        "}"
        "{"
        "V0@(pr0)	vr17 = and.u32 vr5, r47;"
        "V1@(pr0)	vmsk0 = gteq.f32 vr14, vr15;"
        "}"
        "{"
        "V1@(pr0)	vr16 = sub.f32 vr14, vr15;"
        "}"
        "{"
        "V0@(pr0)	vr16 = or.u32 vr16, vr17;"
        "V1@(pr0)	vr10 = sel vmsk0 vr10, vr16;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr5, r46;"
        "V1@(pr0)	vmsk1 = eq.s32 vr5, r47;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk0 vr10, r46;"
        "V1@(pr0)	vr10 = sel vmsk1 vr10, r47;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vr10 = sel vmsk2 vr10, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vr10 = sel vmsk3 vr10, r36;"
        "V1@(pr0)	vr10 = sel vmsk4 vr10, r36;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk5 vr10, vr5;"
        "V1@(pr0)	vr10 = sel vmsk6 vr10, vr5;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.f32 vr11, vr5;"
        "V1@(pr0)	vr16 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vr16 = or.u32 vr16, vr17;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32640;"
        "V0@(pr0)	vr11 = sel vmsk1 vr10, vr16;"
        "V1@(pr0)	vmsk0 = gt.s32 vr2, r37;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32704;"
        "V0@(pr0)	vr10 = sel vmsk0 vr10, r37;"
        "V1@(pr0)	%[res0] = sel vmsk7 vr10, r37;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr4", "vr5", "vr17", "vr16", "vr0", "vr11", "vr3", "vr7", "vr14", "vr15", "vr31", "vr10", "vr6", "vr12", "vmsk3", "vmsk0", "vmsk5", "vmsk4", "vmsk7", "vmsk1", "vmsk2", "vmsk6"
        );
    return result0;
}

#endif // _FMODF_WITHOUT_UNARY_H_
