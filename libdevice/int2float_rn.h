#ifndef _INT2FLOAT_RN_H_
#define _INT2FLOAT_RN_H_

inline float8_128 __dlc_int2float_rn(int8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V1@(pr0)	%[res0] = cvtinttof.f32 vr10;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _INT2FLOAT_RN_H_
