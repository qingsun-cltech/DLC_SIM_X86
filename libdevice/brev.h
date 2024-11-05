#ifndef _BREV_H_X86_
#define _BREV_H_X86_

#include "../dlc-intrinsics.h"
#include "../typehint.h"

inline int8_128 __dlc_brev(int8_128 a) {
    return process1024(a, [](float x){ return acos(x); });
}

#endif
