#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "math.h"
// 1. unary f32算子不需要分块逻辑，直接使用SIM_X86::DLCMem中的size就行
//    unary bf16算子，因为目前没法处理非最后一维的整数倍的情况，所以需要计算一下一次性可以做多少行
inline int CalcVMemBlockSizeBF16Evenly(int height, int width, int vmemSize, int TensorNum) {
    int k = height * width * TensorNum;
    if (k > vmemSize) {
        return (vmemSize / width) / TensorNum;
    }
    return height;
}

inline int CalcVMemBlockSizeBF16Evenly_sdiv(int height, int width, int vmemSize, int TensorNum) {
    int k = height * width * TensorNum;
    if (k > vmemSize) {
        return soft_sdiv(vmemSize, width) / TensorNum;
    }
    return height;
}

// 2.
inline int CalcVMemBlockSizeEvenly(int len, int vmemSize, int TensorNum) {
    int k = len * TensorNum;
    if (k > vmemSize) {
        return (vmemSize / TensorNum) & 0xffffff80;
    }
    return len;
}
inline int CalcVMemBlockSizeEvenly_sdiv(int len, int vmemSize, int TensorNum) {
    int k = len * TensorNum;
    if (k > vmemSize) {
        return soft_sdiv(vmemSize, TensorNum) & 0xffffff80;
    }
    return len;
}

// tril
inline int CalcVMemBlockSizeMatrix(int MatrixNum, int MatrixSize, int VmemSize) {
    int l = 1, r = MatrixNum;
    while (l < r) {
        int mid = (l + r + 1) >> 1;
        if ((mid * MatrixSize) > VmemSize)
            r = mid - 1;
        else
            l = mid;
    }
    return l;
}

inline int calcVMemBlockSize(int height, int width, int vmemSize) {
    int k = height * width;
    if (k > vmemSize) {
        return (soft_sdiv(vmemSize, width));
    }
    return height;
}