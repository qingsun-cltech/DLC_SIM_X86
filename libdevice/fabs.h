#ifndef _FABS_H_
#define _FABS_H_

inline float2 __dlc_fabs(float8_128 a, float8_128 b)
{
    float8_128 result0;
    float8_128 result1;
    float2 res;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	%[res0] = and.u32 vr10, r44;"
        "}"
        ""
        : [res0] "=x" (result0), [res1] "=x" (result1)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr10", "vr11"
        );
    res.x = result0;
    res.y = result1;
    return res;
}

#endif // _FABS_H_
