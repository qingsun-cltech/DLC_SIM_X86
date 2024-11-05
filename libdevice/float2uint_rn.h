#ifndef _FLOAT2UINT_RN_H_
#define _FLOAT2UINT_RN_H_

inline int8_128 __dlc_float2uint_rn(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vmsk1 = ls.f32 vr10, r46;"
        "V1@(pr0)	vr10 = sel vmsk1 vr10, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = cvtftoint.s32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	vr2 = cvtinttof.f32 vr1;"
        "V1@(pr0)	vr3 = and.u32 vr1, r48;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr3, r48;"
        "V1@(pr0)	vr3 = sub.s32 vr1, r48;"
        "}"
        "{"
        "V0@(pr0)	vr0 = sel vmsk1 vr1, vr3;"
        "V1@(pr0)	vr2 = sub.f32 vr2, vr10;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.f32 vr2, r50;"
        "V1@(pr0)	vr5 = sel vmsk1 vr1, vr0;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 127;"
        "pseudo@0	@pseudo imm_2 = 128;"
        "V0@(pr0)	vr4 = and.u32 vr10, r44;"
        "V1@(pr0)	vr4 = or.u32 vr4, r38;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 8;"
        "V1@(pr0)	vr1 = shl.u32 vr4, r33;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20224;"
        "V0@(pr0)	vmsk1 = gteq.s32 vr10, r36;"
        "V1@(pr0)	vr1 = sel vmsk1 vr5, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20352;"
        "V0@(pr0)	vmsk1 = ls.s32 vr10, r36;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr5, vr1;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr5", "vr4", "vr0", "vr3", "vr10", "vmsk1"
        );
    return result0;
}

#endif // _FLOAT2UINT_RN_H_
