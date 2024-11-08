#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef __BINARY_OPERATOR_OPS_H_X86__
#define __BINARY_OPERATOR_OPS_H_X86__

// #include "typehint.h"

inline int8_128 v_u32_or(int8_128 a, int8_128 b) { return a | b; }
inline int8_128 v_s32_sub(int8_128 a, int8_128 b) { return a - b; }

#endif
