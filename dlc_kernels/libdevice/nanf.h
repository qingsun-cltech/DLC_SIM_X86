#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _NANF_H_X86_
#define _NANF_H_X86_

inline float8_128 __dlc_nanf()
{
    float8_128 result0;
    asm (
        "{"
        "pseudo@0	@pseudo imm_0 = 32704;"
        "V0@(pr0)	%[res0] = mov.u32 r36;"
        "}"
        : [res0] "=x" (result0)
        :
        :
        );
    return result0;
}

#endif // _NANF_H_
