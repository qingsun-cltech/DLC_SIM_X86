#ifndef _LL2DOUBLE_RD_H_
#define _LL2DOUBLE_RD_H_

inline float2 __dlc_ll2double_rd(int8_128 a, int8_128 b)
{
    float8_128 result0;
    float8_128 result1;
    float2 res;
    asm (
        "{V0@(pr0)  vr16 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr17 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 32768;"
        "V0@(pr0)	vr12 = mov.u32 r44;"
        "V1@(pr0)	vmsk7 = gteq.f32 vr16, r46;"
        "}"
        "{"
        "V1@(pr0)	vr12 = sel vmsk7 vr12, r46;"
        "}"
        "{"
        "V0@(pr0)	vmsk3 = eq.s32 vr17, r46;"
        "V1@(pr0)	vmsk2 = eq.s32 vr16, r46;"
        "}"
        "{"
        "V0@(pr0)	vr6 = mov.u32 r46;"
        "V1@(pr0)	vr6 = sel vmsk2 vr6, r48;"
        "}"
        "{"
        "V0@(pr0)	vr7 = add.s32 vr6, r48;"
        "V1@(pr0)	vr6 = sel vmsk3 vr6, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 2;"
        "V1@(pr0)	vmsk2 = eq.s32 vr6, r32;"
        "}"
        "{"
        "V1@(pr0)	vr6 = sub.s32 vr17, r48;"
        "}"
        "{"
        "V1@(pr0)	vr7 = sub.s32 vr16, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 65535;"
        "V0@(pr0)	vr6 = sel vmsk3 vr6, r44;"
        "V1@(pr0)	vr7 = sel vmsk3 vr16, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 65535;"
        "V0@(pr0)	vr6 = xor.u32 vr6, r44;"
        "V1@(pr0)	vr7 = xor.u32 vr7, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 65535;"
        "V0@(pr0)	vr17 = sel vmsk7 vr6, vr17;"
        "V1@(pr0)	vr16 = sel vmsk7 vr7, vr16;"
        "}"
        "{"
        "V0@(pr0)	vr2 = count.u32 vr16;"
        "V1@(pr0)	vr1 = count.u32 vr17;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32;"
        "V0@(pr0)	vmsk6 = ls.s32 vr2, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 31;"
        "V0@(pr0)	vr6 = mov.u32 r32;"
        "V1@(pr0)	vr1 = sub.s32 vr6, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 63;"
        "V0@(pr0)	vr6 = mov.u32 r32;"
        "V1@(pr0)	vr2 = sub.s32 vr6, vr2;"
        "}"
        "{"
        "V1@(pr0)	vr13 = sel vmsk6 vr1, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr0 = mov.u32 vr13;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 1023;"
        "V0@(pr0)	vr13 = add.s32 vr13, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20;"
        "V1@(pr0)	vr13 = shl.u32 vr13, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 52;"
        "pseudo@0	@pseudo imm_1 = 20;"
        "V0@(pr0)	vmsk5 = lseq.s32 vr0, r32;"
        "V1@(pr0)	vmsk4 = lseq.s32 vr0, r33;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32;"
        "V0@(pr0)	vr6 = sub.s32 vr0, r32;"
        "V1@(pr0)	vr7 = mov.u32 r48;"
        "}"
        "{"
        "V1@(pr0)	vr6 = shl.u32 vr7, vr6;"
        "}"
        "{"
        "V1@(pr0)	vr6 = sub.s32 vr16, vr6;"
        "}"
        "{"
        "V1@(pr0)	vr7 = shl.u32 vr7, vr0;"
        "}"
        "{"
        "V1@(pr0)	vr7 = sub.s32 vr17, vr7;"
        "}"
        "{"
        "V1@(pr0)	vr5 = sel vmsk4 vr6, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 52;"
        "V0@(pr0)	vr6 = sub.s32 vr0, r32;"
        "V1@(pr0)	vr3 = shr.u32 vr5, vr6;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 84;"
        "V0@(pr0)	vr6 = mov.u32 r32;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sub.s32 vr6, vr0;"
        "V1@(pr0)	vr4 = shl.u32 vr5, vr6;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 52;"
        "V0@(pr0)	vr6 = sub.s32 vr0, r32;"
        "V1@(pr0)	vr6 = shr.u32 vr17, vr6;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20;"
        "V0@(pr0)	vr4 = add.s32 vr4, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr15 = mov.u32 vr3;"
        "V1@(pr0)	vr14 = mov.u32 vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 52;"
        "V0@(pr0)	vr6 = mov.u32 r32;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sub.s32 vr6, vr0;"
        "V1@(pr0)	vr3 = shl.u32 vr5, vr6;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20;"
        "V0@(pr0)	vr6 = sub.s32 vr0, r32;"
        "V1@(pr0)	vr6 = shr.u32 vr17, vr6;"
        "}"
        "{"
        "V1@(pr0)	vr3 = add.s32 vr3, vr6;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 52;"
        "V0@(pr0)	vr7 = mov.u32 r32;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sub.s32 vr7, vr0;"
        "V1@(pr0)	vr4 = shl.u32 vr17, vr7;"
        "}"
        "{"
        "V0@(pr0)	vr15 = sel vmsk5 vr15, vr3;"
        "V1@(pr0)	vr14 = sel vmsk5 vr14, vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20;"
        "V0@(pr0)	vr6 = mov.u32 r32;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sub.s32 vr6, vr0;"
        "V1@(pr0)	vr3 = shl.u32 vr5, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr15 = sel vmsk4 vr15, vr3;"
        "V1@(pr0)	vr14 = sel vmsk4 vr14, r46;"
        "}"
        "{"
        "V0@(pr0)	vr10 = or.u32 vr12, vr13;"
        "V1@(pr0)	vr10 = or.u32 vr10, vr15;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mov.u32 vr14;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk2 vr10, r46;"
        "V1@(pr0)	%[res1] = sel vmsk2 vr11, r46;"
        "}"
        : [res0] "=x" (result0), [res1] "=x" (result1)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr17", "vr5", "vr4", "vr16", "vr0", "vr3", "vr7", "vr15", "vr14", "vr11", "vr13", "vr10", "vr6", "vr12", "vmsk2", "vmsk5", "vmsk4", "vmsk7", "vmsk3", "vmsk6"
        );
    res.x = result0;
    res.y = result1;
    return res;
}

#endif // _LL2DOUBLE_RD_H_
