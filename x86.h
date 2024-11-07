#ifndef _X86_H_
#define _X86_H_

#include "dlc_kernels/align.h"
#include "dlc_kernels/bf16.h"
// #include "dlc_kernels/chunk.h"
#include "dlc_kernels/dma.h"
#include "dlc_kernels/ldst.h"
#include "dlc_kernels/math.h"
#include "dlc_kernels/matmul_t.h"
// #include "dlc_kernels/mm_t.h"
#include "dlc_kernels/nn.h"
#include "dlc_kernels/libdevice.h"
#include "dlc_kernels/permute.h"
#include "dlc_kernels/pingpong.h"

#include "dlc-intrinsics.h"
#include "typehint.h"

template<typename Func, typename... Args>
void invokeFunction(Func&& func, Args&&... args) {
  func(std::forward<Args>(args)...);
}

#endif