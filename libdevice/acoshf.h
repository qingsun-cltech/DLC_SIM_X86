#ifndef _ACOSHF_H_X86_
#define _ACOSHF_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline float8_128 __dlc_acoshf(float8_128 a) {
    return process1024(a, [](float x){ return std::acoshf(x); });
}

#endif
