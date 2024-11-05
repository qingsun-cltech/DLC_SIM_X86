#ifndef _FLOAT2INT_RD_H_
#define _FLOAT2INT_RD_H_

inline int8_128 __dlc_float2int_rd(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vr3 = and.u32 vr10, r47;"
        "V1@(pr0)	vr0 = and.u32 vr10, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 12160;"
        "V0@(pr0)	vmsk1 = gteq.s32 vr0, r37;"
        "V1@(pr0)	vmsk2 = ls.f32 vr10, r46;"
        "}"
        "{"
        "V0@(pr0)	vr2 = sel vmsk2 vr10, r49;"
        "V1@(pr0)	vr10 = sel vmsk1 vr2, vr10;"
        "}"
        "{"
        "V0@(pr0)	vr10 = or.u32 vr10, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mov.u32 vr10;"
        "V1@(pr0)	vmsk7 = gteq.f32 vr11, r46;"
        "}"
        "{"
        "V0@(pr0)	vr12 = cvtftoint.s32 vr11, r46;"
        "V1@(pr0)	vr13 = cvtftoint.s32 vr11, r56;"
        "}"
        "{"
        "V1@(pr0)	%[res0] = sel vmsk7 vr12, vr13;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr0", "vr11", "vr3", "vr13", "vr10", "vr12", "vmsk1", "vmsk2", "vmsk7"
        );
    return result0;
}

#endif // _FLOAT2INT_RD_H_
