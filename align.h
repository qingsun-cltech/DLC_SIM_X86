#ifndef _ALIGN_H_X86_
#define _ALIGN_H_X86_

#include "math.h"

#include "dlc-intrinsics.h"
#include "typehint.h"

//向上取整对应的倍数
#define ALIGN1024(a)  (((a) + 1023) & 0xfffffc00)
#define ALIGN512(a)   (((a) + 511) & 0xfffffe00)
#define ALIGN128(a)   (((a) + 127) & 0xffffff80)
#define ALIGN256(a)   (((a) + 255) & 0xffffff00)
#define SIZE256(sz)   ((((sz)[0] + 255) & 0xffffff00) * (sz)[1] * (sz)[2] * (sz)[3] * (sz)[4])
#define SIZE(sz)      ((((sz)[0] + 127) & 0xffffff80) * (sz)[1] * (sz)[2] * (sz)[3] * (sz)[4])

//计算不同数据类型在dlc中占多少个4B
inline int bf16len(int len, int d0) { return soft_sdiv(len, ALIGN128(d0)) * ALIGN256(d0) / 2; }
inline int uint8len(int len, int d0) { return soft_sdiv(len, ALIGN128(d0)) * ALIGN512(d0) / 4; }

#endif