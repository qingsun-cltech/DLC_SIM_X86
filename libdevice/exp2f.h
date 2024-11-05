#ifndef _EXP2F_H_
#define _EXP2F_H_

inline float8_128 __dlc_exp2f(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
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

#endif // _EXP2F_H_
