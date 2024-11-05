#ifndef _FAST_LOG10F_H_
#define _FAST_LOG10F_H_

inline float8_128 __dlc_fast_log10f(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	(urf) = log.f32 vr10;"
        "MTR@(pr0)	vr1 = pop urf;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 8347;"
        "pseudo@0	@pseudo imm_1 = 16026;"
        "V0@(pr0)	vr1 = mul.f32 vr1, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65408;"
        "V0@(pr0)	vmsk1 = eq.f32 vr10, r47;"
        "V1@(pr0)	vr1 = sel vmsk1 vr1, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65408;"
        "V0@(pr0)	vmsk1 = eq.f32 vr10, r46;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr1, r36;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vmsk1"
        );
    return result0;
}

#endif // _FAST_LOG10F_H_
