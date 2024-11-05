#ifndef _ASINF_H_X86_
#define _ASINF_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline float8_128 __dlc_asinf(float8_128 a) {
    return process1024(a, [](float x){ return asinf(x); });
}

#endif
