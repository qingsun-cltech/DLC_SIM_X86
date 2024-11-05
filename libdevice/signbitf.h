#ifndef _SIGNBITF_H_
#define _SIGNBITF_H_

inline int8_128 __dlc_signbitf(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr10 = and.u32 vr10, r47;"
        "V1@(pr0)	vr1 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = neq.s32 vr10, r46;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr1, r48;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vmsk1"
        );
    return result0;
}

#endif // _SIGNBITF_H_
