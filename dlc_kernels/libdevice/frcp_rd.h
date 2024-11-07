#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FRCP_RD_H_X86_
#define _FRCP_RD_H_X86_

inline float8_128 __dlc_frcp_rd(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr10;"
        "}"
        "{"
        "MTR@(pr0)	%[res0] = pop urf;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _FRCP_RD_H_
