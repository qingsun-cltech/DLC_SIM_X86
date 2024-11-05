#ifndef _SAD_H_
#define _SAD_H_

inline int8_128 __dlc_sad(int8_128 a, int8_128 b, int8_128 c)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{V0@(pr0)  vr12 = mov.u32 %[input2];}"
        "{"
        "V0@(pr0)	vmsk1 = gteq.s32 vr10, vr11;"
        "V1@(pr0)	vr2 = sub.s32 vr10, vr11;"
        "}"
        "{"
        "V0@(pr0)	vr3 = sub.s32 vr11, vr10;"
        "V1@(pr0)	vr10 = sel vmsk1 vr3, vr2;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = add.s32 vr10, vr12;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b),  [input2] "x" (c)
        :"vr2", "vr11", "vr3", "vr10", "vr12", "vmsk1"
        );
    return result0;
}

#endif // _SAD_H_
