#ifndef MATH_ALGORITHMS_H
#define MATH_ALGORITHMS_H

#include <type_traits>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <memoc/allocators.h>
#include <memoc/pointers.h>
#include <computoc/errors.h>

namespace computoc::algorithms {
    namespace details {
        template <typename T>
        concept Arithmetic = std::is_arithmetic_v<T>;

        template <Arithmetic T>
        bool is_equal(T a, T b, T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()))
        {
            return std::abs(a - b) <= epsilon;
        }
    }

    using details::is_equal;
}

namespace computoc::algorithms::derivatives {
    namespace details {
        template <typename T>
        concept Decimal = std::is_floating_point_v<T>;

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        struct Node {
            virtual ~Node() {}
            virtual F compute() const = 0;
            virtual memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const = 0;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Const : public Node<F, Internal_allocator> {
        public:
            Const(F value)
                : value_(value) {}

            F compute() const override
            {
                return value_;
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 0 });
            }

        private:
            F value_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Var : public Node<F, Internal_allocator> {
        public:
            explicit Var(std::size_t id, F value = F{ 0 })
                : id_(id), value_(value) {}

            void set(F value)
            {
                value_ = value;
            }

            F compute() const override
            {
                return value_;
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return id_ == id ?
                    memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }) :
                    memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 0 });
            }

        private:
            std::size_t id_;
            double value_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Add : public Node<F, Internal_allocator> {
        public:
            Add(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return n1_->compute() + n2_->compute();
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Add<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Sub : public Node<F, Internal_allocator> {
        public:
            Sub(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return n1_->compute() - n2_->compute();
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Neg : public Node<F, Internal_allocator> {
        public:
            Neg(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return -n_->compute();
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(n_->backward(id));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Mul : public Node<F, Internal_allocator> {
        public:
            Mul(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return n1_->compute() * n2_->compute();
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                    memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                    memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id)));
            }
        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Div : public Node<F, Internal_allocator> {
        public:
            Div(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                F n2_value{ n2_->compute() };
                COMPUTOC_THROW_IF_FALSE(n2_value != 0, std::overflow_error, "division by zero");

                return n1_->compute() / n2_value;
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Div<F, Internal_allocator>, Internal_allocator>(
                    memoc::pointers::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                        memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id))),
                    memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n2_, n2_));
            }
        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Cos;

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Sin : public Node<F, Internal_allocator> {
        public:
            Sin(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return std::sin(n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Cos<F, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Cos : public Node<F, Internal_allocator> {
        public:
            Cos(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                return std::cos(n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Sin<F, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Sec;

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Tan : public Node<F, Internal_allocator> {
        public:
            Tan(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                return std::tan(n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Sec<F, Internal_allocator>, Internal_allocator>(n_),
                        memoc::pointers::make_shared<Sec<F, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Sec : public Node<F, Internal_allocator> {
        public:
            Sec(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ std::cos(n_->compute()) };
                COMPUTOC_THROW_IF_FALSE(d != 0, std::overflow_error, "division by zero");

                return F{ 1 } / d;
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Sec<F, Internal_allocator>, Internal_allocator>(n_),
                        memoc::pointers::make_shared<Tan<F, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Csc;

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Cot : public Node<F, Internal_allocator> {
        public:
            Cot(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ std::tan(n_->compute()) };
                COMPUTOC_THROW_IF_FALSE(d != 0, std::overflow_error, "division by zero");

                return F{ 1 } / d;
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Csc<F, Internal_allocator>, Internal_allocator>(n_),
                            memoc::pointers::make_shared<Csc<F, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Csc : public Node<F, Internal_allocator> {
        public:
            Csc(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ std::sin(n_->compute()) };
                COMPUTOC_THROW_IF_FALSE(d != 0, std::overflow_error, "division by zero");

                return F{ 1 } / d;
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Csc<F, Internal_allocator>, Internal_allocator>(n_),
                            memoc::pointers::make_shared<Cot<F, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Exp : public Node<F, Internal_allocator> {
        public:
            Exp(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                return std::exp(n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Exp<F, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Ln : public Node<F, Internal_allocator> {
        public:
            Ln(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ n_->compute() };
                COMPUTOC_THROW_IF_FALSE(d > F{ 0 }, std::overflow_error, "log of non-positive number");

                return std::log(d);
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Div<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    n_);
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Pow_fn : public Node<F, Internal_allocator> {
        public:
            Pow_fn(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f, F n)
                : f_(f), n_(n) {}

            F compute() const override
            {
                return std::pow(f_->compute(), n_);
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(n_),
                        memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(f_, n_ - F{ 1 })));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> f_;
            F n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Pow_af : public Node<F, Internal_allocator> {
        public:
            Pow_af(F a, const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f)
                : a_(a), f_(f) {}

            F compute() const override
            {
                return std::pow(a_, f_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Pow_af<F, Internal_allocator>, Internal_allocator>(a_, f_),
                        memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(std::log(a_))));
            }

        private:
            F a_;
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> f_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Pow_fg : public Node<F, Internal_allocator> {
        public:
            Pow_fg(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return std::pow(n1_->compute(), n2_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    memoc::pointers::make_shared<Pow_fg<F, Internal_allocator>, Internal_allocator>(n1_, n2_),
                    memoc::pointers::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Div<F, Internal_allocator>, Internal_allocator>(n2_, n1_),
                            n1_->backward(0)),
                        memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Ln<F, Internal_allocator>, Internal_allocator>(n1_),
                            n2_->backward(0))));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Asin : public Node<F, Internal_allocator> {
        public:
            Asin(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return std::asin(n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                            memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                        F{ -0.5 }));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Acos : public Node<F, Internal_allocator> {
        public:
            Acos(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return std::acos(n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(
                                memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                                memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                            F{ -0.5 })));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Atan : public Node<F, Internal_allocator> {
        public:
            Atan(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return std::atan(n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                            memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                        F{ -1 }));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };

        template <Decimal F, memoc::allocators::Allocator Internal_allocator>
        class Acot : public Node<F, Internal_allocator> {
        public:
            Acot(const memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                F d{ n_->compute() };
                COMPUTOC_THROW_IF_FALSE(d != 0, std::overflow_error, "division by zero");

                return std::atan(d / n_->compute());
            }

            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::pointers::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::pointers::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                            memoc::pointers::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                                memoc::pointers::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                                memoc::pointers::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                            F{ -1 })));
            }

        private:
            memoc::pointers::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
    }

    using details::Acos;
    using details::Acot;
    using details::Add;
    using details::Asin;
    using details::Atan;
    using details::Const;
    using details::Cos;
    using details::Cot;
    using details::Csc;
    using details::Div;
    using details::Exp;
    using details::Ln;
    using details::Mul;
    using details::Neg;
    using details::Node;
    using details::Pow_af;
    using details::Pow_fg;
    using details::Pow_fn;
    using details::Sec;
    using details::Sin;
    using details::Sub;
    using details::Tan;
    using details::Var;
}

#endif // MATH_ALGORITHMS_H
