#ifndef _UINT2FLOAT_RU_H_
#define _UINT2FLOAT_RU_H_

inline float8_128 __dlc_uint2float_ru(int8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "V0@(pr0)	vr1 = count.u32 vr10;"
        "V1@(pr0)	vr2 = mov.u32 r32;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mov.u32 r48;"
        "V1@(pr0)	vr4 = sub.s32 vr2, vr1;"
        "}"
        "{"
        "V0@(pr0)	vr14 = cvtinttof.f32 vr10;"
        "V1@(pr0)	vr2 = shl.u32 vr3, vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20352;"
        "V0@(pr0)	vmsk0 = ls.f32 vr14, r46;"
        "V1@(pr0)	vr16 = add.f32 vr14, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20352;"
        "V0@(pr0)	vr14 = sel vmsk0 vr14, vr16;"
        "V1@(pr0)	vr15 = sub.f32 vr14, r36;"
        "}"
        "{"
        "V0@(pr0)	vr13 = cvtftoint.s32 vr15, vr15;"
        "V1@(pr0)	vr12 = cvtftoint.s32 vr14, vr14;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vmsk0 = neq.s32 vr13, r36;"
        "V1@(pr0)	vr13 = sel vmsk0 vr12, vr13;"
        "}"
        "{"
        "V0@(pr0)	vr7 = mov.u32 r46;"
        "V1@(pr0)	vr11 = add.s32 vr14, r48;"
        "}"
        "{"
        "V0@(pr0)	vr15 = sub.s32 vr13, vr10;"
        "V1@(pr0)	vmsk0 = ls.s32 vr15, r46;"
        "}"
        "{"
        "V0@(pr0)	vr4 = and.u32 vr14, r48;"
        "V1@(pr0)	vr5 = shr.u32 vr2, r48;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sub.s32 vr5, vr2;"
        "V1@(pr0)	vr7 = sub.s32 vr7, vr15;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8;"
        "V0@(pr0)	vr7 = sel vmsk0 vr15, vr7;"
        "V1@(pr0)	vmsk0 = ls.s32 vr1, r32;"
        "}"
        "{"
        "V0@(pr0)	vr11 = sel vmsk0 vr14, vr11;"
        "V1@(pr0)	vmsk0 = eq.s32 vr15, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr14, vr11;"
        "V1@(pr0)	vmsk0 = eq.s32 vr4, r46;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr13, vr14;"
        "V1@(pr0)	vmsk0 = eq.s32 vr15, vr6;"
        "}"
        "{"
        "V0@(pr0)	vr16 = sel vmsk0 vr14, vr11;"
        "V1@(pr0)	vmsk0 = eq.s32 vr4, r48;"
        "}"
        "{"
        "V0@(pr0)	vr16 = sel vmsk0 vr16, vr14;"
        "V1@(pr0)	vmsk0 = gteq.s32 vr15, r46;"
        "}"
        "{"
        "V0@(pr0)	vr12 = sel vmsk0 vr16, vr13;"
        "V1@(pr0)	vr15 = sel vmsk0 vr11, vr14;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr7, vr5;"
        "V1@(pr0)	vr15 = sel vmsk0 vr15, vr14;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk0 vr15, vr12;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr4", "vr1", "vr5", "vr16", "vr3", "vr14", "vr15", "vr7", "vr11", "vr13", "vr10", "vr6", "vr12", "vmsk0"
        );
    return result0;
}

#endif // _UINT2FLOAT_RU_H_
