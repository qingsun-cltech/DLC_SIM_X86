#ifndef _FINITEF_H_
#define _FINITEF_H_

inline float8_128 __dlc_finitef(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vmsk0 = infnan.f32 vr10;"
        "V1@(pr0)	vr1 = mov.u32 r48;"
        "}"
        "{"
        "V1@(pr0)	%[res0] = sel vmsk0 vr1, r46;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vmsk0"
        );
    return result0;
}

#endif // _FINITEF_H_
