#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #pragma once

#define MIN_FLT -3.40282e+38f
#define MAX_FLT 3.40282e+38f
#define INF_BIN 0x7f800000
#define NEG_INF_BIN 0xff800000
#define NAN_BIN 0x7fffffff
#define NEG_NAN_BIN 0xffffffff

#ifndef INFINITY
#define INFINITY (__builtin_huge_valf())
#endif
#ifndef QUIET_NAN
#define QUIET_NAN (__builtin_nanf (""))
#endif

enum {
    LOSS_REDUCTION_NONE = 0,
    LOSS_REDUCTION_MEAN = 1,
    LOSS_REDUCTION_SUM = 2
};
