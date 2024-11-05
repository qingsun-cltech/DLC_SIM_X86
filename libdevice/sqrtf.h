#ifndef _SQRTF_H_
#define _SQRTF_H_

inline float8_128 __dlc_sqrtf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	(urf) = rsqrt.f32 vr10;"
        "V1@(pr0)	vmsk1 = eq.s32 vr10, r47;"
        "}"
        "{"
        "MTR@(pr0)      vr10 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr10;"
        "}"
        "{"
        "MTR@(pr0)      vr10 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk1 vr10, r47;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vmsk1"
        );
    return result0;
}

#endif // _SQRTF_H_
