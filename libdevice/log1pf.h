#ifndef _LOG1PF_H_
#define _LOG1PF_H_

inline float8_128 __dlc_log1pf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr2 = mov.u32 vr10;"
        "V1@(pr0)	vr10 = add.f32 vr10, r49;"
        "}"
        "{"
        "V0@(pr0)	(urf) = log.f32 vr10;"
        "MTR@(pr0)	vr1 = pop urf;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 29208;"
        "pseudo@0	@pseudo imm_1 = 16177;"
        "V0@(pr0)	vr10 = mul.f32 vr1, r44;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.f32 vr10, r46;"
        "V1@(pr0)	vr10 = sel vmsk1 vr10, vr2;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65408;"
        "V0@(pr0)	vmsk1 = eq.f32 vr2, r57;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr10, r36;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr10", "vr1", "vmsk1"
        );
    return result0;
}

#endif // _LOG1PF_H_
