#ifndef MATH_ALGORITHMS_H
#define MATH_ALGORITHMS_H

#include <type_traits>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <math/core/allocators.h>
#include <math/core/pointers.h>
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

namespace math::algorithms::derivatives::backward {
    template <typename T>
    concept Decimal = std::is_floating_point_v<T>;

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    struct Node {
        virtual ~Node() {}
        virtual F compute() const = 0;
        virtual math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const = 0;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Const : public Node<F, Internal_allocator> {
    public:
        Const(F value)
            : value_(value) {}

        F compute() const override
        {
            return value_;
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(F{ 0 });
        }

    private:
        F value_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Var : public Node<F, Internal_allocator> {
    public:
        Var(std::size_t id, F value)
            : id_(id), value_(value) {}

        F compute() const override
        {
            return value_;
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return id_ == id ? 
                math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(F{ 1 }) :
                math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(F{ 0 });
        }

    private:
        std::size_t id_;
        double value_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Add : public Node<F, Internal_allocator> {
    public:
        Add(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
            : n1_(n1), n2_(n2) {}

        F compute() const override
        {
            return n1_->compute() + n2_->compute();
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Add<F, Internal_allocator>, Internal_allocator>::make_shared(n1_->backward(id), n2_->backward(id));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Sub : public Node<F, Internal_allocator> {
    public:
        Sub(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
            : n1_(n1), n2_(n2) {}

        F compute() const override
        {
            return n1_->compute() - n2_->compute();
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Sub<F, Internal_allocator>, Internal_allocator>::make_shared(n1_->backward(id), n2_->backward(id));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Neg : public Node<F, Internal_allocator> {
    public:
        Neg(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
            : n_(n) {}

        F compute() const override
        {
            return -n_->compute();
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Neg<F, Internal_allocator>, Internal_allocator>::make_shared(n_->backward(id));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Mul : public Node<F, Internal_allocator> {
    public:
        Mul(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
            : n1_(n1), n2_(n2) {}

        F compute() const override
        {
            return n1_->compute() * n2_->compute();
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Add<F, Internal_allocator>, Internal_allocator>::make_shared(
                math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(n1_->backward(id), n2_),
                math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(n1_, n2_->backward(id)));
        }
    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Div : public Node<F, Internal_allocator> {
    public:
        Div(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
            : n1_(n1), n2_(n2) {}

        F compute() const override
        {
            F n2_value{ n2_->compute() };
            CORE_EXPECT(n2_value != 0, std::overflow_error, "division by zero");

            return n1_->compute() / n2_value;
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Div<F, Internal_allocator>>::make_shared(
                math::core::pointers::Shared_ptr<Sub<F, Internal_allocator>>::make_shared(
                    math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>>::make_shared(n1_->backward(id), n2_),
                    math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>>::make_shared(n1_, n2_->backward(id))),
                math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>>::make_shared(n2_, n2_));
        }
    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Sin : public Node<F, Internal_allocator> {
    public:
        Sin(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
            : n_(n) {}

        F compute() const override
        {
            return std::sin(n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Cos<F, Internal_allocator>, Internal_allocator>::make_shared(n_));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Cos : public Node<F, Internal_allocator> {
    public:
        Cos(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
            : n_(v) {}

        F compute() const override
        {
            return std::cos(n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Neg<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Sin<F, Internal_allocator>, Internal_allocator>::make_shared(n_)));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Tan : public Node<F, Internal_allocator> {
    public:
        Tan(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
            : n_(v) {}

        F compute() const override
        {
            return std::tan(n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Sec<F, Internal_allocator>, Internal_allocator>::make_shared(n_),
                    math::core::pointers::Shared_ptr<Sec<F, Internal_allocator>, Internal_allocator>::make_shared(n_)));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Sec : public Node<F, Internal_allocator> {
    public:
        Sec(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
            : n_(v) {}

        F compute() const override
        {
            F d{ std::cos(n_->compute()) };
            CORE_EXPECT(d != 0, std::overflow_error, "division by zero");

            return F{ 1 } / d;
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Sec<F, Internal_allocator>, Internal_allocator>::make_shared(n_),
                    math::core::pointers::Shared_ptr<Tan<F, Internal_allocator>, Internal_allocator>::make_shared(n_)));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Cot : public Node<F, Internal_allocator> {
    public:
        Cot(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
            : n_(v) {}

        F compute() const override
        {
            F d{ std::tan(n_->compute()) };
            CORE_EXPECT(d != 0, std::overflow_error, "division by zero");

            return F{ 1 } / d;
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Neg<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Csc<F, Internal_allocator>, Internal_allocator>::make_shared(n_),
                        math::core::pointers::Shared_ptr<Csc<F, Internal_allocator>, Internal_allocator>::make_shared(n_))));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Csc : public Node<F, Internal_allocator> {
    public:
        Csc(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
            : n_(v) {}

        F compute() const override
        {
            F d{ std::sin(n_->compute()) };
            CORE_EXPECT(d != 0, std::overflow_error, "division by zero");

            return F{ 1 } / d;
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Neg<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Csc<F, Internal_allocator>, Internal_allocator>::make_shared(n_),
                        math::core::pointers::Shared_ptr<Cot<F, Internal_allocator>, Internal_allocator>::make_shared(n_))));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Exp : public Node<F, Internal_allocator> {
    public:
        Exp(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
            : n_(v) {}

        F compute() const override
        {
            return std::exp(n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Exp<F, Internal_allocator>, Internal_allocator>::make_shared(n_));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Ln : public Node<F, Internal_allocator> {
    public:
        Ln(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
            : n_(v) {}

        F compute() const override
        {
            F d{ n_->compute() };
            CORE_EXPECT(d > F{ 0 }, std::overflow_error, "log of non-positive number");

            return std::log(d);
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Div<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                n_);
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Pow_fn : public Node<F, Internal_allocator> {
    public:
        Pow_fn(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f, F n)
            : f_(f), n_(n) {}

        F compute() const override
        {
            return std::pow(f_->compute(), n_);
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                f_->backward(id),
                math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(n_),
                    math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(f_, n_ - F{ 1 })));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> f_;
        F n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Pow_af : public Node<F, Internal_allocator> {
    public:
        Pow_af(F a, const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f)
            : a_(a), f_(f) {}

        F compute() const override
        {
            return std::pow(a_, f_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                f_->backward(id),
                math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Pow_af<F, Internal_allocator>, Internal_allocator>::make_shared(a_, f_),
                    math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(std::log(a_))));
        }

    private:
        F a_;
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> f_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Pow_fg : public Node<F, Internal_allocator> {
    public:
        Pow_fg(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
            : n1_(n1), n2_(n2) {}

        F compute() const override
        {
            return std::pow(n1_->compute(), n2_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                math::core::pointers::Shared_ptr<Pow_fg<F, Internal_allocator>, Internal_allocator>::make_shared(n1_, n2_),
                math::core::pointers::Shared_ptr<Add<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Div<F, Internal_allocator>, Internal_allocator>::make_shared(n2_, n1_),
                        n1_->backward(0)),
                    math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Ln<F, Internal_allocator>, Internal_allocator>::make_shared(n1_),
                        n2_->backward(0))));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Asin : public Node<F, Internal_allocator> {
    public:
        Asin(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
            : n_(n) {}

        F compute() const override
        {
            return std::asin(n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Sub<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(F{ 1 }),
                        math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(n_, F{ 2 })),
                    F{ -0.5 }));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Acos : public Node<F, Internal_allocator> {
    public:
        Acos(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
            : n_(n) {}

        F compute() const override
        {
            return std::acos(n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Neg<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Sub<F, Internal_allocator>, Internal_allocator>::make_shared(
                            math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(F{ 1 }),
                            math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(n_, F{ 2 })),
                        F{ -0.5 })));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Atan : public Node<F, Internal_allocator> {
    public:
        Atan(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
            : n_(n) {}

        F compute() const override
        {
            return std::atan(n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Add<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(F{ 1 }),
                        math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(n_, F{ 2 })),
                    F{ -1 }));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Acot : public Node<F, Internal_allocator> {
    public:
        Acot(const math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
            : n_(n) {}

        F compute() const override
        {
            F d{ n_->compute() };
            CORE_EXPECT(d != 0, std::overflow_error, "division by zero");

            return std::atan(d / n_->compute());
        }

        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
        {
            return math::core::pointers::Shared_ptr<Mul<F, Internal_allocator>, Internal_allocator>::make_shared(
                n_->backward(id),
                math::core::pointers::Shared_ptr<Neg<F, Internal_allocator>, Internal_allocator>::make_shared(
                    math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(
                        math::core::pointers::Shared_ptr<Add<F, Internal_allocator>, Internal_allocator>::make_shared(
                            math::core::pointers::Shared_ptr<Const<F, Internal_allocator>, Internal_allocator>::make_shared(F{ 1 }),
                            math::core::pointers::Shared_ptr<Pow_fn<F, Internal_allocator>, Internal_allocator>::make_shared(n_, F{ 2 })),
                        F{ -1 })));
        }

    private:
        math::core::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
    };
}

#endif // MATH_ALGORITHMS_H