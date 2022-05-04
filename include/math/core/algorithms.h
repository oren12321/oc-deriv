#ifndef MATH_ALGORITHMS_H
#define MATH_ALGORITHMS_H

#include <type_traits>
#include <cmath>
#include <limits>

#include <math/core/utils.h>

namespace math::algorithms {
    template <typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    template <Arithmetic T>
    bool is_equal(T a, T b, T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()))
    {
        return std::abs(a - b) <= epsilon;
    }
}

#endif // MATH_ALGORITHMS_H