#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"



namespace Types {
struct false_type {
    constexpr static bool value = false;
};

struct true_type {
    constexpr static bool value = true;
};

template <class T, class U> struct is_same : false_type {};

template <class T> struct is_same<T, T> : true_type {};

template <class T, class U>
concept same_as = is_same<T, U>::value && is_same<T, U>::value;
} // namespace Types
