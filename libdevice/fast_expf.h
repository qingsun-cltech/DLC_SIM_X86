#ifndef _FAST_EXPF_H_
#define _FAST_EXPF_H_

inline float8_128 __dlc_fast_expf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	(urf) = exp.f32 vr10;"
        "V1@(pr0)	vmsk0 = ls.f32 vr10, r46;"
        "}"
        "{"
        "MTR@(pr0)	vr0 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr0;"
        "}"
        "{"
        "MTR@(pr0)	vr1 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk0 vr0, vr1;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vr0", "vmsk0"
        );
    return result0;
}

#endif // _FAST_EXPF_H_
