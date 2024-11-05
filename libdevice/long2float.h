#ifndef _LONG2FLOAT_H_
#define _LONG2FLOAT_H_
#include "uint2float_rd.h"
// const float _2_32 = 4294967296.0f;
/*
    high: a
    low: b
*/
inline float8_128 __dlc_long2float(int8_128 a, int8_128 b)
{
    // float8_128 result0;
    float8_128 a_float = v_cvt_itof(a);
    float8_128 b_float = __dlc_uint2float_rd(b);
    return a_float + b_float * 4294967296.0f;
}

#endif 
