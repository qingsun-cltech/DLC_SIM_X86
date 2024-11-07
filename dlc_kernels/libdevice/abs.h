#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _ABS_H_X86_
#define _ABS_H_X86_

inline int8_128 __dlc_abs(int8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vr1 = mov.u32 r46;"
        "V1@(pr0)	vr1 = sub.s32 vr1, vr10;"
        "}"
        "{"
        "V0@(pr0)	vmsk0 = ls.s32 vr1, vr10;"
        "V1@(pr0)	vr0 = sel vmsk0 vr1, vr10;"
        "}"
        "{"
        "V0@(pr0)	vmsk1 = eq.s32 vr10, r47;"
        "V1@(pr0)	%[res0] = sel vmsk1 vr0, vr10;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10", "vr1", "vr0", "vmsk0", "vmsk1"
        );
    return result0;
}

#endif // _ABS_H_
