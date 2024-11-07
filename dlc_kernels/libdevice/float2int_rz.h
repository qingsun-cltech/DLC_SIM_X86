#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FLOAT2INT_RZ_H_X86_
#define _FLOAT2INT_RZ_H_X86_

inline int8_128 __dlc_float2int_rz(float8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V1@(pr0)	%[res0] = cvtftoint.s32 vr10, r56;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _FLOAT2INT_RZ_H_
