#ifndef _UMULHI_H_
#define _UMULHI_H_

inline int8_128 __dlc_umulhi(int8_128 a, int8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 24;"
        "pseudo@0	@pseudo imm_1 = 255;"
        "V0@(pr0)	vr3 = and.u32 vr10, r33;"
        "V1@(pr0)	vr0 = shr.u32 vr10, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "pseudo@0	@pseudo imm_1 = 65280;"
        "V0@(pr0)	vr2 = and.u32 vr10, r33;"
        "V1@(pr0)	vr1 = shr.u32 vr10, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "pseudo@0	@pseudo imm_1 = 255;"
        "V0@(pr0)	vr1 = and.u32 vr1, r33;"
        "V1@(pr0)	vr2 = shr.u32 vr2, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 24;"
        "pseudo@0	@pseudo imm_1 = 255;"
        "V0@(pr0)	vr7 = and.u32 vr11, r33;"
        "V1@(pr0)	vr4 = shr.u32 vr11, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "pseudo@0	@pseudo imm_1 = 65280;"
        "V0@(pr0)	vr6 = and.u32 vr11, r33;"
        "V1@(pr0)	vr5 = shr.u32 vr11, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "pseudo@0	@pseudo imm_1 = 255;"
        "V0@(pr0)	vr5 = and.u32 vr5, r33;"
        "V1@(pr0)	vr6 = shr.u32 vr6, r32;"
        "}"
        "{"
        "V0@(pr0)	vr0 = cvtinttof.f32 vr0;"
        "V1@(pr0)	vr1 = cvtinttof.f32 vr1;"
        "}"
        "{"
        "V0@(pr0)	vr2 = cvtinttof.f32 vr2;"
        "V1@(pr0)	vr3 = cvtinttof.f32 vr3;"
        "}"
        "{"
        "V0@(pr0)	vr4 = cvtinttof.f32 vr4;"
        "V1@(pr0)	vr5 = cvtinttof.f32 vr5;"
        "}"
        "{"
        "V0@(pr0)	vr6 = cvtinttof.f32 vr6;"
        "V1@(pr0)	vr7 = cvtinttof.f32 vr7;"
        "}"
        "{"
        "V0@(pr0)	vr28 = mul.f32 vr0, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr30 = mul.f32 vr1, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr31 = mul.f32 vr0, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr1, vr7;"
        "V1@(pr0)	vr31 = cvtftoint.s32 vr31, r56;"
        "}"
        "{"
        "V0@(pr0)	vr30 = cvtftoint.s32 vr30, r56;"
        "V1@(pr0)	vr29 = cvtftoint.s32 vr29, r56;"
        "}"
        "{"
        "V1@(pr0)	vr30 = add.s32 vr31, vr30;"
        "}"
        "{"
        "V0@(pr0)	vr28 = cvtftoint.s32 vr28, r56;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr28 = shl.u32 vr28, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "V0@(pr0)	vr28 = add.s32 vr28, vr29;"
        "V1@(pr0)	vr30 = shl.u32 vr30, r32;"
        "}"
        "{"
        "V0@(pr0)	vr12 = add.s32 vr28, vr30;"
        "}"
        "{"
        "V0@(pr0)	vr28 = mul.f32 vr2, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr30 = mul.f32 vr2, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr31 = mul.f32 vr3, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr3, vr5;"
        "V1@(pr0)	vr31 = cvtftoint.s32 vr31, r56;"
        "}"
        "{"
        "V0@(pr0)	vr30 = cvtftoint.s32 vr30, r56;"
        "V1@(pr0)	vr29 = cvtftoint.s32 vr29, r56;"
        "}"
        "{"
        "V1@(pr0)	vr30 = add.s32 vr31, vr30;"
        "}"
        "{"
        "V0@(pr0)	vr28 = cvtftoint.s32 vr28, r56;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr28 = shl.u32 vr28, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "V0@(pr0)	vr28 = add.s32 vr28, vr29;"
        "V1@(pr0)	vr30 = shl.u32 vr30, r32;"
        "}"
        "{"
        "V0@(pr0)	vr13 = add.s32 vr28, vr30;"
        "}"
        "{"
        "V0@(pr0)	vr28 = mul.f32 vr0, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr30 = mul.f32 vr0, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr31 = mul.f32 vr1, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr1, vr5;"
        "V1@(pr0)	vr31 = cvtftoint.s32 vr31, r56;"
        "}"
        "{"
        "V0@(pr0)	vr30 = cvtftoint.s32 vr30, r56;"
        "V1@(pr0)	vr29 = cvtftoint.s32 vr29, r56;"
        "}"
        "{"
        "V1@(pr0)	vr30 = add.s32 vr31, vr30;"
        "}"
        "{"
        "V0@(pr0)	vr28 = cvtftoint.s32 vr28, r56;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr28 = shl.u32 vr28, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "V0@(pr0)	vr28 = add.s32 vr28, vr29;"
        "V1@(pr0)	vr30 = shl.u32 vr30, r32;"
        "}"
        "{"
        "V0@(pr0)	vr14 = add.s32 vr28, vr30;"
        "}"
        "{"
        "V0@(pr0)	vr28 = mul.f32 vr2, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr30 = mul.f32 vr2, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr31 = mul.f32 vr3, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr29 = mul.f32 vr3, vr7;"
        "V1@(pr0)	vr31 = cvtftoint.s32 vr31, r56;"
        "}"
        "{"
        "V0@(pr0)	vr30 = cvtftoint.s32 vr30, r56;"
        "V1@(pr0)	vr29 = cvtftoint.s32 vr29, r56;"
        "}"
        "{"
        "V1@(pr0)	vr30 = add.s32 vr31, vr30;"
        "}"
        "{"
        "V0@(pr0)	vr28 = cvtftoint.s32 vr28, r56;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr28 = shl.u32 vr28, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "V0@(pr0)	vr28 = add.s32 vr28, vr29;"
        "V1@(pr0)	vr30 = shl.u32 vr30, r32;"
        "}"
        "{"
        "V0@(pr0)	vr15 = add.s32 vr28, vr30;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr0 = mov.u32 r46;"
        "V1@(pr0)	vr1 = shl.u32 vr12, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr2 = shl.u32 vr13, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vmsk0 = carry.u32 vr1, vr2;"
        "V1@(pr0)	vr3 = shr.u32 vr12, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr1 = sel vmsk0 vr0, r48;"
        "V1@(pr0)	vr4 = shr.u32 vr13, r32;"
        "}"
        "{"
        "V0@(pr0)	vr3 = add.s32 vr3, vr4;"
        "V1@(pr0)	vr6 = and.u32 vr12, r54;"
        "}"
        "{"
        "V0@(pr0)	vr2 = and.u32 vr13, r54;"
        "V1@(pr0)	vr5 = add.s32 vr3, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr1 = add.s32 vr6, vr2;"
        "V1@(pr0)	vr5 = shl.u32 vr5, r32;"
        "}"
        "{"
        "V0@(pr0)	vr1 = and.u32 vr1, r54;"
        "}"
        "{"
        "V0@(pr0)	vr16 = or.u32 vr1, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr17 = shl.u32 vr16, r32;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = carry.u32 vr17, vr15;"
        "V1@(pr0)	vmsk2 = carry.u32 vr12, vr13;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr17 = sel vmsk0 vr0, r48;"
        "V1@(pr0)	vr16 = shr.u32 vr16, r32;"
        "}"
        "{"
        "V0@(pr0)	vr16 = add.s32 vr16, vr17;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr1 = shl.u32 vr14, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 16;"
        "V0@(pr0)	vr2 = and.u32 vr14, r54;"
        "V1@(pr0)	vr28 = shl.u32 vr16, r33;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vmsk1 = carry.u32 vr1, vr28;"
        "V1@(pr0)	vr3 = shr.u32 vr14, r32;"
        "}"
        "{"
        "V0@(pr0)	vr4 = sel vmsk1 vr0, r48;"
        "V1@(pr0)	vr7 = sel vmsk2 vr0, r48;"
        "}"
        "{"
        "V0@(pr0)	vr5 = add.s32 vr2, vr16;"
        "V1@(pr0)	vr6 = add.s32 vr3, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr6 = add.s32 vr6, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 16;"
        "V0@(pr0)	vr5 = and.u32 vr5, r54;"
        "V1@(pr0)	vr6 = shl.u32 vr6, r33;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = or.u32 vr6, vr5;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr4", "vr30", "vr10", "vr6", "vr12", "vr7", "vr15", "vr31", "vr1", "vr5", "vr0", "vr28", "vr13", "vr29", "vr2", "vr17", "vr16", "vr3", "vr11", "vr14", "vmsk0", "vmsk2", "vmsk1"
        );
    return result0;
}

#endif // _UMULHI_H_
