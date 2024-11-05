#ifndef _UHADD_H_
#define _UHADD_H_

inline int8_128 __dlc_uhadd(int8_128 a, int8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr0 = mov.u32 r46;"
        "V1@(pr0)	vr1 = shl.u32 vr10, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V1@(pr0)	vr2 = shl.u32 vr11, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vmsk0 = carry.u32 vr1, vr2;"
        "V1@(pr0)	vr3 = shr.u32 vr10, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 16;"
        "V0@(pr0)	vr1 = sel vmsk0 vr0, r48;"
        "V1@(pr0)	vr4 = shr.u32 vr11, r32;"
        "}"
        "{"
        "V0@(pr0)	vr3 = add.s32 vr3, vr4;"
        "V1@(pr0)	vr6 = and.u32 vr10, r54;"
        "}"
        "{"
        "V0@(pr0)	vr2 = and.u32 vr11, r54;"
        "V1@(pr0)	vr5 = add.s32 vr3, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 15;"
        "V0@(pr0)	vr1 = add.s32 vr6, vr2;"
        "V1@(pr0)	vr5 = shl.u32 vr5, r32;"
        "}"
        "{"
        "V0@(pr0)	vr1 = and.u32 vr1, r54;"
        "V1@(pr0)	vr1 = shr.u32 vr1, r48;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = or.u32 vr1, vr5;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr11", "vr3", "vr10", "vr6", "vmsk0"
        );
    return result0;
}

#endif // _UHADD_H_
