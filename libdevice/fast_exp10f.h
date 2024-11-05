#ifndef _FAST_EXP10F_H_
#define _FAST_EXP10F_H_

inline float8_128 __dlc_fast_exp10f(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 39544;"
        "pseudo@0	@pseudo imm_1 = 16468;"
        "V0@(pr0)	vr10 = mul.f32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	(urf) = pow.f32 vr10;"
        "}"
        "{"
        "MTR@(pr0)	%[res0] = pop urf;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _FAST_EXP10F_H_
