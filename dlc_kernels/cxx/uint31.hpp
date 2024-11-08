#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

// #pragma once
// #include "typehint.h"

struct Uint31 {
    union {
        int sval;
        unsigned uval;
    };
    constexpr Uint31(int v) : sval(v) {}
    constexpr Uint31(unsigned v) : uval(v) {}
    constexpr inline operator int() { return sval; }
    constexpr inline operator unsigned() { return uval; }

    constexpr inline Uint31 operator+(const Uint31 b) const { return Uint31(sval + b.sval); }
    constexpr inline Uint31 operator+(const int b) const { return Uint31(sval + b); }
    constexpr inline Uint31 operator+(const unsigned b) const { return Uint31(sval + *(int *)(&b)); }

    constexpr inline Uint31 operator-(const Uint31 b) const { return Uint31(sval - b.sval); }
    constexpr inline Uint31 operator-(const int b) const { return Uint31(sval - b); }
    constexpr inline Uint31 operator-(const unsigned b) const { return Uint31(sval - *(int *)(&b)); }

    constexpr inline void operator+=(const Uint31 b) { sval += b.sval; }
    constexpr inline void operator+=(const int b) { sval += b; }
    constexpr inline void operator+=(const unsigned b) { sval += *(int *)(&b); }

    constexpr inline void operator++() { sval += 1; }

    constexpr inline void operator-=(const Uint31 b) { sval -= b.sval; }
    constexpr inline void operator-=(const int b) { sval -= b; }
    constexpr inline void operator-=(const unsigned b) { sval -= *(int *)(&b); }

    constexpr inline void operator--() { sval -= 1; }

    constexpr inline Uint31 operator*(const Uint31 b) const { return Uint31(uval * b.uval); }
    constexpr inline Uint31 operator*(const int b) const { return Uint31(uval * *(unsigned *)(&b)); }
    constexpr inline Uint31 operator*(const unsigned b) const { return Uint31(uval * b); }

    constexpr inline void operator*=(const Uint31 b) { uval *= b.uval; }
    constexpr inline void operator*=(const int b) { uval *= *(unsigned *)(&b); }
    constexpr inline void operator*=(const unsigned b) { uval *= b; }

    constexpr inline Uint31 operator/(const Uint31 b) const { return Uint31(uval / b.uval); }
    constexpr inline Uint31 operator/(const int b) const { return Uint31(uval / *(unsigned *)(&b)); }
    constexpr inline Uint31 operator/(const unsigned b) const { return Uint31(uval / b); }

    constexpr inline Uint31 operator%(const Uint31 b) const { return Uint31(uval % b.uval); }
    constexpr inline Uint31 operator%(const int b) const { return Uint31(uval % *(unsigned *)(&b)); }
    constexpr inline Uint31 operator%(const unsigned b) const { return Uint31(uval % b); }

    constexpr inline bool operator<(const Uint31 b) const { return sval < b.sval; }
    constexpr inline bool operator<(const int b) const { return sval < b; }
    constexpr inline bool operator<(const unsigned b) const { return uval < *(int *)(&b); }

    constexpr inline bool operator<=(const Uint31 b) const { return sval <= b.sval; }
    constexpr inline bool operator<=(const int b) const { return sval <= b; }
    constexpr inline bool operator<=(const unsigned b) const { return sval <= *(int *)(&b); }

    constexpr inline bool operator>(const Uint31 b) const { return sval > b.sval; }
    constexpr inline bool operator>(const int b) const { return sval > b; }
    constexpr inline bool operator>(const unsigned b) const { return sval > *(int *)(&b); }

    constexpr inline bool operator>=(const Uint31 b) const { return sval >= b.sval; }
    constexpr inline bool operator>=(const int b) const { return sval >= b; }
    constexpr inline bool operator>=(const unsigned b) const { return sval >= *(int *)(&b); }

    constexpr inline bool operator==(const Uint31 b) const { return sval == b.sval; }
    constexpr inline bool operator==(const int b) const { return sval == b; }
    constexpr inline bool operator==(const unsigned b) const { return sval == *(int *)(&b); }

    constexpr inline bool operator!=(const Uint31 b) const { return sval != b.sval; }
    constexpr inline bool operator!=(const int b) const { return sval != b; }
    constexpr inline bool operator!=(const unsigned b) const { return sval != *(int *)(&b); }

    constexpr inline Uint31 operator<<(const Uint31 b) const { return Uint31(uval << b.uval); }
    constexpr inline Uint31 operator<<(const int b) const { return Uint31(uval << b); }
    constexpr inline Uint31 operator<<(const unsigned b) const { return Uint31(uval << b); }

    constexpr inline Uint31 operator>>(const Uint31 b) const { return Uint31(uval >> b.uval); }
    constexpr inline Uint31 operator>>(const int b) const { return Uint31(uval >> b); }
    constexpr inline Uint31 operator>>(const unsigned b) const { return Uint31(uval >> b); }

    constexpr inline Uint31 operator|(const Uint31 b) const { return Uint31(uval | b.uval); }
    constexpr inline Uint31 operator|(const int b) const { return Uint31(uval | b); }
    constexpr inline Uint31 operator|(const unsigned b) const { return Uint31(uval | b); }

    constexpr inline Uint31 operator&(const Uint31 b) const { return Uint31(uval & b.uval); }
    constexpr inline Uint31 operator&(const int b) const { return Uint31(uval & b); }
    constexpr inline Uint31 operator&(const unsigned b) const { return Uint31(uval & b); }
};

constexpr inline Uint31 operator<<(const int a, const Uint31 b) { return Uint31(a << b.uval); }
constexpr inline Uint31 operator<<(const unsigned a, const Uint31 b) { return Uint31(a << b.uval); }

constexpr inline Uint31 operator*(const int b, const Uint31 a) { return Uint31(a.uval * *(unsigned *)(&b)); }
constexpr inline Uint31 operator*(const unsigned b, const Uint31 a) { return Uint31(a.uval * b); }

constexpr inline bool operator<(const int a, const Uint31 b) { return a < b.sval; }
constexpr inline bool operator<(const unsigned a, const Uint31 b) { return *(int *)(&a) < b.uval; }

constexpr inline bool operator>(const int a, const Uint31 b) { return a > b.sval; }
constexpr inline bool operator>(const unsigned a, const Uint31 b) { return *(int *)(&a) > b.uval; }

constexpr inline Uint31 operator-(const int a, const Uint31 b) { return Uint31(a - b.sval); }
constexpr inline Uint31 operator-(const unsigned a, const Uint31 b) { return Uint31(*(int *)(&a) - b.sval); }
