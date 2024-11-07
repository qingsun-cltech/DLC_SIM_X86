#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _MIN_H_X86_
#define _MIN_H_X86_

inline int8_128 __dlc_min(int8_128 a, int8_128 b)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	vmsk0 = ls.s32 vr10, vr11;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk0 vr11, vr10;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr10", "vr11", "vmsk0"
        );
    return result0;
}

#endif // _MIN_H_
