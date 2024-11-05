#ifndef _ASINHF_H_X86_
#define _ASINHF_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline float8_128 __dlc_asinhf(float8_128 a) {
    return process1024(a, [](float x){ return std::asinhf(x); });
}

#endif
