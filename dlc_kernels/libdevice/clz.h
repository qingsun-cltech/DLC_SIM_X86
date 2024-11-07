#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _CLZ_H_X86_
#define _CLZ_H_X86_

inline int8_128 __dlc_clz(int8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	%[res0] = count.u32 vr10;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _CLZ_H_
