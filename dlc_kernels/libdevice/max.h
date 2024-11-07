#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _MAX_H_X86_
#define _MAX_H_X86_

inline int8_128 __dlc_max(int8_128 a, int8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	vmsk0 = gt.s32 vr10, vr11;"
        "}"
        "{"
        "V0@(pr0)	vr12 = sel vmsk0 vr11, %[res0];"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr10", "vr11", "vr12", "vmsk0"
        );
    return result0;
}

#endif // _MAX_H_
