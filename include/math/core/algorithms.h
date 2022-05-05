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

namespace math::algorithms::derivatives {

    class Const {
    public:
        Const(double value)
            : value_(value) {}

        double compute()
        {
            return value_;
        }

        double backward(std::size_t)
        {
            return 0.0;
        }
    private:
        double value_;
    };

    class Var {
    public:
        Var(double value, std::size_t index)
            : value_(value), index_(index) {}

        double compute()
        {
            return value_;
        }

        double backward(std::size_t index)
        {
            return index_ == index ? 1.0 : 0.0;
        }
    private:
        double value_;
        std::size_t index_;
    };

    template <typename T, typename U>
    class Add {
    public:
        Add(T x, U y)
            : x_(x), y_(y) {}

        double compute()
        {
            return x_.compute() + y_.compute();
        }

        double backward(std::size_t index)
        {
            return x_.backward(index) + y_.backward(index);
        }
    private:
        T x_;
        U y_;
    };

    template <typename T, typename U>
    class Mul {
    public:
        Mul(T x, U y)
            : x_(x), y_(y) {}

        double compute()
        {
            return x_.compute() * y_.compute();
        }

        double backward(std::size_t index)
        {
            return x_.backward(index) * y_.compute() + x_.compute() * y_.backward(index);
        }
    private:
        T x_;
        U y_;
    };
}

#endif // MATH_ALGORITHMS_H