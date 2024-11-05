#ifndef _ATAN2F_H_X86_
#define _ATAN2F_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline float8_128 __dlc_atan2f(float8_128 x, float8_128 y) {
    float8_128 res;

    for (int i = 0; i < 1024; ++i) {
        res[i] = atan2f(x[i], y[i]);
    }

    return res;
}

#endif