#ifndef _POWIF_H_
#define _POWIF_H_

inline float8_128 __dlc_powif(float8_128 a, int8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vr16 = and.u32 vr10, r36;"
        "V1@(pr0)	vmsk2 = eq.s32 vr11, r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "pseudo@0	@pseudo imm_2 = 0;"
        "pseudo@0	@pseudo imm_3 = 32768;"
        "V0@(pr0)	vmsk0 = eq.s32 vr16, r45;"
        "V1@(pr0)	vr10 = and.u32 vr10, r44;"
        "}"
        "{"
        "V0@(pr0)	(urf) = log.f32 vr10;"
        "V1@(pr0)	vr15 = and.u32 vr11, r48;"
        "}"
        "{"
        "V1@(pr0)	vr11 = cvtinttof.f32 vr11;"
        "MTR@(pr0)	vr12 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vr13 = mul.f32 vr11, vr12;"
        "V1@(pr0)	vmsk1 = eq.s32 vr15, r48;"
        "}"
        "{"
        "V0@(pr0)	(urf) = pow.f32 vr13;"
        "MISC@(pr0)	vmsk1 = vmsk.and vmsk0;"
        "}"
        "{"
        "MTR@(pr0)	vr10 = pop urf;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vr11 = or.u32 vr10, r36;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk1 vr10, vr11;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk2 vr10, r46;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr16", "vr11", "vr15", "vr13", "vr10", "vr12", "vmsk0", "vmsk2", "vmsk1"
        );
    return result0;
}

#endif // _POWIF_H_
