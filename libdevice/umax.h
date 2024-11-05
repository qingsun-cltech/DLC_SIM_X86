#ifndef _UMAX_H_
#define _UMAX_H_

inline int8_128 __dlc_umax(int8_128 a, int8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	vr0 = and.u32 vr10, r47;"
        "V1@(pr0)	vr1 = and.u32 vr11, r47;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr2 = and.u32 vr10, r44;"
        "V1@(pr0)	vr3 = and.u32 vr11, r44;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = neq.s32 vr0, vr1;"
        "V1@(pr0)	vmsk1 = ls.s32 vr10, vr11;"
        "}"
        "{"
        "V0@(pr0)	vmsk2 = ls.s32 vr2, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr2 = sel vmsk0 vr10, r46;"
        "V1@(pr0)	vr3 = sel vmsk0 vr11, r46;"
        "}"
        "{"
        "V0@(pr0)	vr2 = sel vmsk2 vr2, vr3;"
        "V1@(pr0)	vr3 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vr0 = sel vmsk0 vr3, vr10;"
        "V1@(pr0)	vr1 = sel vmsk0 vr3, vr11;"
        "}"
        "{"
        "V0@(pr0)	vr0 = sel vmsk1 vr1, vr0;"
        "V1@(pr0)	%[res0] = add.s32 vr0, vr2;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr2", "vr1", "vr0", "vr11", "vr3", "vr10", "vmsk0", "vmsk2", "vmsk1"
        );
    return result0;
}

#endif // _UMAX_H_
