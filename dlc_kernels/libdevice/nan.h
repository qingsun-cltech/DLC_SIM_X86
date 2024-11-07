#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _NAN_H_X86_
#define _NAN_H_X86_

inline float2 __dlc_nan()
{
    float8_128 result0;
    float8_128 result1;
    float2 res;
    asm (
        "{"
        "pseudo@0	@pseudo imm_1 = 32760;"
        "V0@(pr0)	%[res0] = mov.u32 r37;"
        "V1@(pr0)	%[res1] = mov.u32 r46;"
        "}"
        ""
        : [res0] "=x" (result0), [res1] "=x" (result1)
        :
        :
        );
    res.x = result0;
    res.y = result1;
    return res;
}

#endif // _NAN_H_
