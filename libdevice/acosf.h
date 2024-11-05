#ifndef _ACOSF_H_X86_
#define _ACOSF_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline float8_128 __dlc_acosf(float8_128 a) {
    return process1024(a, [](float x){ return acosf(x); });

    return a;
}

#endif
