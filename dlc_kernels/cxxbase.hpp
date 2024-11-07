#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"


#include "../cxx/types.hpp"
#include "../cxx/uint31.hpp"


struct CxxTensor {
    Uint31 addr;

    constexpr CxxTensor(unsigned address) : addr(address) {}
    constexpr CxxTensor(Uint31 address) : addr(address) {}
    constexpr CxxTensor(SIM_X86::tensor address) : addr((unsigned)address) {}

    constexpr inline CxxTensor operator+(Uint31 off) { return CxxTensor(addr + off); }

    constexpr inline operator SIM_X86::tensor() { return (SIM_X86::tensor)addr.uval; }
};

template <class O, class I> inline O bitAs(I);
template <> inline float8_128 bitAs(int8_128 a) { return *(float8_128 *)(&a); }
template <> inline float8_128 bitAs(float8_128 a) { return a; }
template <> inline int8_128 bitAs(int8_128 a) { return a; }
template <> inline int8_128 bitAs(float8_128 a) { return *(int8_128 *)(&a); }
