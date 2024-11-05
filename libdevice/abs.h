#ifndef _ABS_H_X86_
#define _ABS_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline int8_128 __dlc_abs(int8_128 a){
    return process1024(a, [](int x){ return std::abs(x); });
}

#endif
