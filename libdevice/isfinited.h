#ifndef _ISFINITED_H_
#define _ISFINITED_H_

inline int8_128 __dlc_isfinited(float8_128 a, float8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65535;"
        "pseudo@0	@pseudo imm_1 = 32767;"
        "V0@(pr0)	vr10 = and.u32 vr10, r44;"
        "V1@(pr0)	vr0 = mov.u32 r46;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_1 = 32752;"
        "V0@(pr0)	vmsk1 = ls.s32 vr10, r37;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr0, r48;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr10", "vr11", "vr0", "vmsk1"
        );
    return result0;
}

#endif // _ISFINITED_H_
