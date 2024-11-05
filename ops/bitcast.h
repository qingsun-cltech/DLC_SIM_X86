#ifndef __BITCAST_OPS_H__
#define __BITCAST_OPS_H__

#include "typehint.h"

inline int8_128 float_as_int(float8_128 a) { return *(int8_128 *)(&a); }
inline float8_128 int_as_float(int8_128 a) { return *(float8_128 *)(&a); }

#endif
