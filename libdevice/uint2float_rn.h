#ifndef _UINT2FLOAT_RN_H_
#define _UINT2FLOAT_RN_H_

inline float8_128 __dlc_uint2float_rn(int8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vmsk0 = ls.s32 vr10, r46;"
        "V1@(pr0)	vr10 = cvtinttof.f32 vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 20352;"
        "V1@(pr0)	vr11 = add.f32 vr10, r36;"
        "}"
        "{"
        "V1@(pr0)	%[res0] = sel vmsk0 vr10, vr11;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr11", "vmsk0"
        );
    return result0;
}

#endif // _UINT2FLOAT_RN_H_
