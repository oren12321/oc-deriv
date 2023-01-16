#ifndef COMPUTOC_DERIVATIVES_H
#define COMPUTOC_DERIVATIVES_H

#include <stdexcept>
#include <cstdint>

#include <memoc/allocators.h>
#include <memoc/pointers.h>
#include <erroc/errors.h>
#include <computoc/concepts.h>
#include <computoc/math.h>

namespace computoc {
    namespace details {
        template <Numeric T, memoc::Allocator Internal_allocator>
        struct Node {
            virtual ~Node() {}
            [[nodiscard]] virtual T compute() const = 0;
            [[nodiscard]] virtual memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const = 0;
            virtual void set(T value) {}
        };

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Const : public Node<T, Internal_allocator> {
        public:
            Const(T value)
                : value_(value) {}

            T compute() const override
            {
                return value_;
            }

            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(T{ 0 });
            }

        private:
            T value_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> constant(T value)
        {
            return memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Var : public Node<T, Internal_allocator> {
        public:
            explicit Var(std::int64_t id, T value = T{ 0 })
                : id_(id), value_(value) {}

            void set(T value) override
            {
                value_ = value;
            }

            [[nodiscard]] T compute() const override
            {
                return value_;
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return id_ == id ?
                    memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }) :
                    memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(T{ 0 });
            }

        private:
            std::int64_t id_;
            T value_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> variable(std::int64_t id, T value = T{ 0 })
        {
            return memoc::make_shared<Var<T, Internal_allocator>, Internal_allocator>(id, value);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Add : public Node<T, Internal_allocator> {
        public:
            Add(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return n1_->compute() + n2_->compute();
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> add(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator+(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator+(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, T value)
        {
            return memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator+(T value, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Sub : public Node<T, Internal_allocator> {
        public:
            Sub(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return n1_->compute() - n2_->compute();
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> subtract(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator-(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator-(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, T value)
        {
            return memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator-(T value, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Neg : public Node<T, Internal_allocator> {
        public:
            Neg(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return -n_->compute();
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(n_->backward(id));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> negate(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(n);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator-(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Mul : public Node<T, Internal_allocator> {
        public:
            Mul(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return n1_->compute() * n2_->compute();
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(
                    memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                    memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id)));
            }
        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> multiply(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator*(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator*(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, T value)
        {
            return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator*(T value, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Div : public Node<T, Internal_allocator> {
        public:
            Div(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                T n2_value{ n2_->compute() };
                ERROC_EXPECT(n2_value != T{}, std::overflow_error, "division by zero");

                return n1_->compute() / n2_value;
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Div<T, Internal_allocator>, Internal_allocator>(
                    memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                        memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id))),
                    memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(n2_, n2_));
            }
        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> divide(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Div<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator/(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Div<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator/(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, T value)
        {
            return memoc::make_shared<Div<T, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator/(T value, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Div<T, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Cos;

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Sin : public Node<T, Internal_allocator> {
        public:
            Sin(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return sin(n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Cos<T, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> sin(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Sin<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Cos : public Node<T, Internal_allocator> {
        public:
            Cos(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                return cos(n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sin<T, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> cos(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Cos<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Sec;

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Tan : public Node<T, Internal_allocator> {
        public:
            Tan(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                return tan(n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sec<T, Internal_allocator>, Internal_allocator>(n_),
                        memoc::make_shared<Sec<T, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> tan(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Tan<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Sec : public Node<T, Internal_allocator> {
        public:
            Sec(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ cos(n_->compute()) };
                ERROC_EXPECT(d != T{}, std::overflow_error, "division by zero");

                return T{ 1 } / d;
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sec<T, Internal_allocator>, Internal_allocator>(n_),
                        memoc::make_shared<Tan<T, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> sec(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Sec<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Csc;

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Cot : public Node<T, Internal_allocator> {
        public:
            Cot(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ tan(n_->compute()) };
                ERROC_EXPECT(d != T{}, std::overflow_error, "division by zero");

                return T{ 1 } / d;
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Csc<T, Internal_allocator>, Internal_allocator>(n_),
                            memoc::make_shared<Csc<T, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> cot(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Cot<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Csc : public Node<T, Internal_allocator> {
        public:
            Csc(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ sin(n_->compute()) };
                ERROC_EXPECT(d != T{}, std::overflow_error, "division by zero");

                return T{ 1 } / d;
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Csc<T, Internal_allocator>, Internal_allocator>(n_),
                            memoc::make_shared<Cot<T, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> csc(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Csc<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Exp : public Node<T, Internal_allocator> {
        public:
            Exp(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                return exp(n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Exp<T, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> exp(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Exp<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Ln : public Node<T, Internal_allocator> {
        public:
            Ln(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ n_->compute() };
                ERROC_EXPECT(d > T{}, std::overflow_error, "log of non-positive number");

                return log(d);
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Div<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    n_);
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> ln(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Ln<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Pow_fn : public Node<T, Internal_allocator> {
        public:
            Pow_fn(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& f, T n)
                : f_(f), n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return pow(f_->compute(), n_);
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(n_),
                        memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(f_, n_ - T{ 1 })));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> f_;
            T n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> pow(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& f, T n)
        {
            return memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(f, n);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator^(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& f, T n)
        {
            return memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(f, n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Pow_af : public Node<T, Internal_allocator> {
        public:
            Pow_af(T a, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& f)
                : a_(a), f_(f) {}

            [[nodiscard]] T compute() const override
            {
                return pow(a_, f_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Pow_af<T, Internal_allocator>, Internal_allocator>(a_, f_),
                        memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(log(a_))));
            }

        private:
            T a_;
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> f_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> pow(T a, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& f)
        {
            return memoc::make_shared<Pow_af<T, Internal_allocator>, Internal_allocator>(a, f);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator^(T a, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& f)
        {
            return memoc::make_shared<Pow_af<T, Internal_allocator>, Internal_allocator>(a, f);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Pow_fg : public Node<T, Internal_allocator> {
        public:
            Pow_fg(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return pow(n1_->compute(), n2_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    memoc::make_shared<Pow_fg<T, Internal_allocator>, Internal_allocator>(n1_, n2_),
                    memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Div<T, Internal_allocator>, Internal_allocator>(n2_, n1_),
                            n1_->backward(0)),
                        memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Ln<T, Internal_allocator>, Internal_allocator>(n1_),
                            n2_->backward(0))));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> pow(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Pow_fg<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> operator^(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Pow_fg<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Asin : public Node<T, Internal_allocator> {
        public:
            Asin(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return asin(n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                            memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                        T{ -0.5 }));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> asin(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Asin<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Acos : public Node<T, Internal_allocator> {
        public:
            Acos(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return acos(n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Sub<T, Internal_allocator>, Internal_allocator>(
                                memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                                memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                            T{ -0.5 })));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> acos(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Acos<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Atan : public Node<T, Internal_allocator> {
        public:
            Atan(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return atan(n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                            memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                        T{ -1 }));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> atan(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Atan<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <Numeric T, memoc::Allocator Internal_allocator>
        class Acot : public Node<T, Internal_allocator> {
        public:
            Acot(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                T d{ n_->compute() };
                ERROC_EXPECT(d != T{}, std::overflow_error, "division by zero");

                return atan(d / n_->compute());
            }

            [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> backward(std::int64_t id) const override
            {
                return memoc::make_shared<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<T, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Add<T, Internal_allocator>, Internal_allocator>(
                                memoc::make_shared<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                                memoc::make_shared<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                            T{ -1 })));
            }

        private:
            memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Numeric T>
        [[nodiscard]] memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator> acot(const memoc::Shared_ptr<Node<T, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Acot<T, Internal_allocator>, Internal_allocator>(n);
        }
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

    using details::acos;
    using details::acot;
    using details::add;
    using details::asin;
    using details::atan;
    using details::constant;
    using details::cos;
    using details::cot;
    using details::csc;
    using details::divide;
    using details::exp;
    using details::ln;
    using details::multiply;
    using details::negate;
    using details::pow;
    using details::sec;
    using details::sin;
    using details::subtract;
    using details::tan;
    using details::variable;
}

#endif // COMPUTOC_ALGORITHMS_H
