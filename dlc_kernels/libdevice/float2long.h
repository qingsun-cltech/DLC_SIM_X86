#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef _FLOAT2LONG_H_X86_
#define _FLOAT2LONG_H_X86_
// #include "../uint2float_rd.h"
// const float _2_32 = 4294967296.0f;
/*
    high: a
    low: b
*/
// inline float8_128 __dlc_long2float(int8_128 a, int8_128 b)
inline void __dlc_float2long(float8_128 a, int8_128 *lo, int8_128 *hi)
{
    float8_128 a_float = a / 4294967296.0f;
    // Print("a_float: %f\n", a_float);
    // Print("b_float: %f\n", b_float);

    *hi = v_cvt_ftoi(a_float, 0xff);
    a_float = v_cvt_itof(*hi);
    float8_128 b_float = a - a_float * 4294967296.0f;

    *lo = v_cvt_ftoi(b_float, 0xff);
}

#endif 