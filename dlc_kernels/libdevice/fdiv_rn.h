#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FDIV_RN_H_X86_
#define _FDIV_RN_H_X86_

inline float8_128 __dlc_fdiv_rn(float8_128 a, float8_128 b)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{V0@(pr0)  vr11 = mov.u32 %[input1];}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr11;"
        "}"
        "{"
        "MTR@(pr0)	vr11 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = mul.f32 vr10, vr11;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a),  [input1] "x" (b)
        :"vr10", "vr11"
        );
    return result0;
}

#endif // _FDIV_RN_H_