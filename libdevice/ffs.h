#ifndef _FFS_H_
#define _FFS_H_

inline int8_128 __dlc_ffs(int8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr1 = mov.u32 r46;"
        "V1@(pr0)	vr1 = sub.s32 vr1, vr10;"
        "}"
        "{"
        "V0@(pr0)	vr1 = and.u32 vr10, vr1;"
        "V1@(pr0)	vr1 = count.u32 vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32;"
        "V0@(pr0)	vr0 = mov.u32 r32;"
        "V1@(pr0)	vr1 = sub.s32 vr0, vr1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 32;"
        "V0@(pr0)	vmsk0 = eq.s32 vr10, r47;"
        "V1@(pr0)	%[res0] = sel vmsk0 vr1, r32;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vr0", "vmsk0"
        );
    return result0;
}

#endif // _FFS_H_
