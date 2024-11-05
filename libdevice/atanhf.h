#ifndef _ATANHF_H_X86_
#define _ATANHF_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline float8_128 __dlc_atanhf(float8_128 a) {
    return process1024(a, [](float x){ return std::atanhf(x); });
}

#endif