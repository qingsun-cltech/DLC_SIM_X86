#ifndef _FLOAT_AS_INT_H_
#define _FLOAT_AS_INT_H_

inline int8_128 __dlc_float_as_int(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V1@(pr0)	%[res0] = mov.u32 vr10;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _FLOAT_AS_INT_H_
