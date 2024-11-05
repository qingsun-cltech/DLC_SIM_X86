#ifndef _NEXTAFTERF_H_
#define _NEXTAFTERF_H_

inline float8_128 __dlc_nextafterf(float8_128 a, float8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	vr14 = mov.u32 r46;"
        "V1@(pr0)	vr1 = add.s32 vr10, r48;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mov.u32 r46;"
        "V1@(pr0)	vr2 = sub.s32 vr10, r48;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr4 = and.u32 vr10, r44;"
        "V1@(pr0)	vr5 = and.u32 vr11, r44;"
        "}"
        "{"
        "V0@(pr0)	vr6 = xor.u32 vr10, vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vr6 = and.u32 vr6, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32768;"
        "V0@(pr0)	vmsk7 = eq.s32 vr6, r36;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = gt.s32 vr4, vr5;"
        "V1@(pr0)	vmsk1 = gteq.s32 vr5, r46;"
        "}"
        "{"
        "V0@(pr0)	vr31 = sel vmsk0 vr14, vr2;"
        "V1@(pr0)	vr14 = sel vmsk1 vr14, vr31;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = gt.s32 vr5, vr4;"
        "V1@(pr0)	vmsk1 = gteq.s32 vr4, r46;"
        "}"
        "{"
        "V0@(pr0)	vr31 = sel vmsk0 vr14, vr1;"
        "V1@(pr0)	vr14 = sel vmsk1 vr14, vr31;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk7 vr14, vr2;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, r46;"
        "V1@(pr0)	vmsk1 = gt.s32 vr11, r46;"
        "}"
        "{"
        "V0@(pr0)	vmsk2 = eq.s32 vr10, r47;"
        "V1@(pr0)	vmsk3 = eq.s32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vmsk4 = eq.s32 vr4, r46;"
        "}"
        "{"
        "V0@(pr0)	vr7 = xor.u32 vr1, r56;"
        "}"
        "{"
        "V0@(pr0)	vr7 = and.u32 vr7, r47;"
        "}"
        "{"
        "V0@(pr0)	vr7 = add.s32 vr7, r48;"
        "}"
        "{"
        "V0@(pr0)	vr31 = sel vmsk1 vr7, vr1;"
        "V1@(pr0)	vr13 = sel vmsk0 vr13, vr31;"
        "}"
        "{"
        "V0@(pr0)	vr31 = sel vmsk1 vr1, vr7;"
        "V1@(pr0)	vr13 = sel vmsk2 vr13, vr31;"
        "}"
        "{"
        "V0@(pr0)	vr31 = sel vmsk3 vr13, vr11;"
        "V1@(pr0)	vr13 = sel vmsk4 vr13, vr31;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, vr11;"
        "}"
        "{"
        "V1@(pr0)	vr13 = sel vmsk0 vr13, vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = eq.s32 vr4, r36;"
        "}"
        "{"
        "V0@(pr0)	vr13 = sel vmsk0 vr13, vr10;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32640;"
        "V0@(pr0)	vmsk0 = gt.s32 vr4, r36;"
        "V1@(pr0)	vmsk1 = gt.s32 vr5, r36;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	vr13 = sel vmsk0 vr13, r36;"
        "V1@(pr0)	vr13 = sel vmsk1 vr13, r36;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = mov.u32 vr13;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr4", "vr5", "vr3", "vr11", "vr14", "vr7", "vr13", "vr31", "vr10", "vr6", "vmsk3", "vmsk0", "vmsk4", "vmsk7", "vmsk1", "vmsk2"
        );
    return result0;
}

#endif // _NEXTAFTERF_H_
