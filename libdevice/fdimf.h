#ifndef _FDIMF_H_
#define _FDIMF_H_

inline float8_128 __dlc_fdimf(float8_128 a, float8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	vr2 = mov.u32 r46;"
        "V1@(pr0)	vr1 = sub.f32 vr10, vr11;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = gt.f32 vr1, r46;"
        "V1@(pr0)	vr12 = sel vmsk0 vr2, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr10 = and.u32 vr10, r44;"
        "V1@(pr0)	vr11 = and.u32 vr11, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = gt.s32 vr10, r36;"
        "V1@(pr0)	vmsk1 = gt.s32 vr11, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vr13 = sel vmsk0 vr12, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr13, r36;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr11", "vr13", "vr10", "vr12", "vmsk0", "vmsk1"
        );
    return result0;
}

#endif // _FDIMF_H_
