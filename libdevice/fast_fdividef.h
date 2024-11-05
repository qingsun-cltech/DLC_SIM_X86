#ifndef _FAST_FDIVIDEF_H_
#define _FAST_FDIVIDEF_H_

inline float8_128 __dlc_fast_fdividef(float8_128 a, float8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr11;"
        "V1@(pr0)	vr13 = mov.u32 vr11;"
        "}"
        "{"
        "MTR@(pr0)	vr11 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vr11 = mul.f32 vr10, vr11;"
        "V1@(pr0)	vr30 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 32384;"
        "pseudo@0	@pseudo imm_2 = 0;"
        "pseudo@0	@pseudo imm_3 = 32640;"
        "V0@(pr0)	vmsk0 = gt.f32 vr13, r45;"
        "V1@(pr0)	vmsk1 = ls.f32 vr13, r44;"
        "}"
        "{"
        "V0@(pr0)	vr3 = sel vmsk0 vr31, vr11;"
        "V1@(pr0)	vr1 = sel vmsk0 vr30, vr11;"
        "}"
        "{"
        "V0@(pr0)	vr2 = sel vmsk1 vr1, vr11;"
        "V1@(pr0)	vr4 = sel vmsk1 vr3, vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr5 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = eq.s32 vr5, r36;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk0 vr2, vr4;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr4", "vr5", "vr3", "vr11", "vr30", "vr13", "vr31", "vr10", "vmsk0", "vmsk1"
        );
    return result0;
}

#endif // _FAST_FDIVIDEF_H_
