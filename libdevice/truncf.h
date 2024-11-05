#ifndef _TRUNCF_H_
#define _TRUNCF_H_

inline float8_128 __dlc_truncf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr5 = and.u32 vr10, r47;"
        "V1@(pr0)	vr0 = cvtftoint.s32 vr10, r56;"
        "}"
        "{"
        "V0@(pr0)	vr0 = cvtinttof.f32 vr0;"
        "V1@(pr0)	vr0 = or.u32 vr0, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "pseudo@0	@pseudo imm_2 = 20224;"
        "V0@(pr0)	vr1 = and.u32 vr10, r44;"
        "V1@(pr0)	vmsk0 = gt.f32 vr1, r38;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr1 = mov.u32 r44;"
        "V1@(pr0)	vmsk1 = infnan.f32 vr10;"
        "}"
        "{"
        "V0@(pr0)	vr0 = sel vmsk1 vr0, vr1;"
        "V1@(pr0)	%[res0] = sel vmsk0 vr0, vr10;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr5", "vr0", "vr1", "vmsk0", "vmsk1"
        );
    return result0;
}

#endif // _TRUNCF_H_
