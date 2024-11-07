#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FLOAT_AS_INT_H_X86_
#define _FLOAT_AS_INT_H_X86_

#include "../ops/bitcast.h"

inline int8_128 __dlc_float_as_int(float8_128 a)
{
    return float_as_int(a);
}

#endif // _FLOAT_AS_INT_H_
