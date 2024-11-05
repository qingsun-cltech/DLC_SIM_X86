#ifndef _LOGBF_H_
#define _LOGBF_H_

inline float8_128 __dlc_logbf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr2 = and.u32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk1 = lseq.f32 vr2, r36;"
        "V1@(pr0)	vr1 = sel vmsk1 vr10, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk1 = eq.f32 vr2, r36;"
        "V1@(pr0)	vr1 = sel vmsk1 vr1, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 23;"
        "V1@(pr0)	vr4 = shr.u32 vr2, r32;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 127;"
        "V0@(pr0)	vr4 = sub.s32 vr4, r32;"
        "V1@(pr0)	vr5 = cvtinttof.f32 vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk1 = ls.f32 vr2, r36;"
        "V1@(pr0)	vr1 = sel vmsk1 vr1, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr4 = count.u32 vr2;"
        "V1@(pr0)	vr5 = cvtinttof.f32 vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 0;"
        "pseudo@0	@pseudo imm_1 = 49900;"
        "V0@(pr0)	vr4 = mov.u32 r44;"
        "V1@(pr0)	vr4 = sub.f32 vr4, vr5;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 128;"
        "V0@(pr0)	vmsk1 = ls.f32 vr2, r36;"
        "V1@(pr0)	vr1 = sel vmsk1 vr1, vr4;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65408;"
        "V0@(pr0)	vmsk1 = eq.f32 vr2, r46;"
        "V1@(pr0)	vr1 = sel vmsk1 vr1, r36;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = mov.u32 vr1;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr4", "vr1", "vr5", "vr10", "vmsk1"
        );
    return result0;
}

#endif // _LOGBF_H_
