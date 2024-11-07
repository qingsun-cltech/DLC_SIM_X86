#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _INT_AS_FLOAT_H_X86_
#define _INT_AS_FLOAT_H_X86_

#include "../ops/bitcast.h"

inline float8_128 __dlc_int_as_float(int8_128 a)
{
    return int_as_float(a);
}

#endif // _INT_AS_FLOAT_H_
