#ifndef COMPUTOC_DERIVATIVES_H
#define COMPUTOC_DERIVATIVES_H

#include <stdexcept>

#include <memoc/allocators.h>
#include <memoc/pointers.h>
#include <computoc/errors.h>
#include <computoc/concepts.h>
#include <computoc/math.h>

namespace computoc {
    namespace details {
        template <Arithmetic F, memoc::Allocator Internal_allocator>
        struct Node {
            virtual ~Node() {}
            virtual F compute() const = 0;
            virtual memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const = 0;
            virtual void set(F value) {}
        };

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Const : public Node<F, Internal_allocator> {
        public:
            Const(F value)
                : value_(value) {}

            F compute() const override
            {
                return value_;
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 0 });
            }

        private:
            F value_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> constant(F value)
        {
            return memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Var : public Node<F, Internal_allocator> {
        public:
            explicit Var(std::size_t id, F value = F{ 0 })
                : id_(id), value_(value) {}

            void set(F value) override
            {
                value_ = value;
            }

            F compute() const override
            {
                return value_;
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return id_ == id ?
                    memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }) :
                    memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 0 });
            }

        private:
            std::size_t id_;
            F value_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> variable(std::size_t id, F value = F{ 0 })
        {
            return memoc::make_shared<Var<F, Internal_allocator>, Internal_allocator>(id, value);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Add : public Node<F, Internal_allocator> {
        public:
            Add(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return n1_->compute() + n2_->compute();
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> add(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator+(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator+(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, F value)
        {
            return memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator+(F value, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Sub : public Node<F, Internal_allocator> {
        public:
            Sub(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return n1_->compute() - n2_->compute();
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> subtract(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator-(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator-(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, F value)
        {
            return memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator-(F value, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Neg : public Node<F, Internal_allocator> {
        public:
            Neg(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return -n_->compute();
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(n_->backward(id));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> negate(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(n);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator-(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Mul : public Node<F, Internal_allocator> {
        public:
            Mul(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return n1_->compute() * n2_->compute();
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                    memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                    memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id)));
            }
        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> multiply(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator*(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator*(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, F value)
        {
            return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator*(F value, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Div : public Node<F, Internal_allocator> {
        public:
            Div(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                F n2_value{ n2_->compute() };
                COMPUTOC_THROW_IF_FALSE(n2_value != F{}, std::overflow_error, "division by zero");

                return n1_->compute() / n2_value;
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Div<F, Internal_allocator>, Internal_allocator>(
                    memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                        memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id))),
                    memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(n2_, n2_));
            }
        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> divide(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Div<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator/(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Div<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator/(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, F value)
        {
            return memoc::make_shared<Div<F, Internal_allocator>, Internal_allocator>(n1, memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value));
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator/(F value, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Div<F, Internal_allocator>, Internal_allocator>(memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Cos;

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Sin : public Node<F, Internal_allocator> {
        public:
            Sin(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return sin(n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Cos<F, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> sin(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Sin<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Cos : public Node<F, Internal_allocator> {
        public:
            Cos(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                return cos(n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sin<F, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> cos(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Cos<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Sec;

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Tan : public Node<F, Internal_allocator> {
        public:
            Tan(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                return tan(n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sec<F, Internal_allocator>, Internal_allocator>(n_),
                        memoc::make_shared<Sec<F, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> tan(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Tan<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Sec : public Node<F, Internal_allocator> {
        public:
            Sec(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ cos(n_->compute()) };
                COMPUTOC_THROW_IF_FALSE(d != F{}, std::overflow_error, "division by zero");

                return F{ 1 } / d;
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sec<F, Internal_allocator>, Internal_allocator>(n_),
                        memoc::make_shared<Tan<F, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> sec(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Sec<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Csc;

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Cot : public Node<F, Internal_allocator> {
        public:
            Cot(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ tan(n_->compute()) };
                COMPUTOC_THROW_IF_FALSE(d != F{}, std::overflow_error, "division by zero");

                return F{ 1 } / d;
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Csc<F, Internal_allocator>, Internal_allocator>(n_),
                            memoc::make_shared<Csc<F, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> cot(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Cot<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Csc : public Node<F, Internal_allocator> {
        public:
            Csc(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ sin(n_->compute()) };
                COMPUTOC_THROW_IF_FALSE(d != F{}, std::overflow_error, "division by zero");

                return F{ 1 } / d;
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Csc<F, Internal_allocator>, Internal_allocator>(n_),
                            memoc::make_shared<Cot<F, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> csc(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Csc<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Exp : public Node<F, Internal_allocator> {
        public:
            Exp(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                return exp(n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Exp<F, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> exp(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Exp<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Ln : public Node<F, Internal_allocator> {
        public:
            Ln(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& v)
                : n_(v) {}

            F compute() const override
            {
                F d{ n_->compute() };
                COMPUTOC_THROW_IF_FALSE(d > F{}, std::overflow_error, "log of non-positive number");

                return log(d);
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Div<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    n_);
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> ln(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Ln<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Pow_fn : public Node<F, Internal_allocator> {
        public:
            Pow_fn(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f, F n)
                : f_(f), n_(n) {}

            F compute() const override
            {
                return pow(f_->compute(), n_);
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(n_),
                        memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(f_, n_ - F{ 1 })));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> f_;
            F n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> pow(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f, F n)
        {
            return memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(f, n);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator^(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f, F n)
        {
            return memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(f, n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Pow_af : public Node<F, Internal_allocator> {
        public:
            Pow_af(F a, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f)
                : a_(a), f_(f) {}

            F compute() const override
            {
                return pow(a_, f_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Pow_af<F, Internal_allocator>, Internal_allocator>(a_, f_),
                        memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(log(a_))));
            }

        private:
            F a_;
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> f_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> pow(F a, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f)
        {
            return memoc::make_shared<Pow_af<F, Internal_allocator>, Internal_allocator>(a, f);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator^(F a, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& f)
        {
            return memoc::make_shared<Pow_af<F, Internal_allocator>, Internal_allocator>(a, f);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Pow_fg : public Node<F, Internal_allocator> {
        public:
            Pow_fg(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
                : n1_(n1), n2_(n2) {}

            F compute() const override
            {
                return pow(n1_->compute(), n2_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    memoc::make_shared<Pow_fg<F, Internal_allocator>, Internal_allocator>(n1_, n2_),
                    memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Div<F, Internal_allocator>, Internal_allocator>(n2_, n1_),
                            n1_->backward(0)),
                        memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Ln<F, Internal_allocator>, Internal_allocator>(n1_),
                            n2_->backward(0))));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n1_;
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n2_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> pow(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Pow_fg<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> operator^(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n1, const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n2)
        {
            return memoc::make_shared<Pow_fg<F, Internal_allocator>, Internal_allocator>(n1, n2);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Asin : public Node<F, Internal_allocator> {
        public:
            Asin(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return asin(n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                            memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                        F{ -0.5 }));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> asin(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Asin<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Acos : public Node<F, Internal_allocator> {
        public:
            Acos(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return acos(n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Sub<F, Internal_allocator>, Internal_allocator>(
                                memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                                memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                            F{ -0.5 })));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> acos(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Acos<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Atan : public Node<F, Internal_allocator> {
        public:
            Atan(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                return atan(n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                            memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                        F{ -1 }));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> atan(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Atan<F, Internal_allocator>, Internal_allocator>(n);
        }

        template <Arithmetic F, memoc::Allocator Internal_allocator>
        class Acot : public Node<F, Internal_allocator> {
        public:
            Acot(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
                : n_(n) {}

            F compute() const override
            {
                F d{ n_->compute() };
                COMPUTOC_THROW_IF_FALSE(d != F{}, std::overflow_error, "division by zero");

                return atan(d / n_->compute());
            }

            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> backward(std::size_t id) const override
            {
                return memoc::make_shared<Mul<F, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    memoc::make_shared<Neg<F, Internal_allocator>, Internal_allocator>(
                        memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(
                            memoc::make_shared<Add<F, Internal_allocator>, Internal_allocator>(
                                memoc::make_shared<Const<F, Internal_allocator>, Internal_allocator>(F{ 1 }),
                                memoc::make_shared<Pow_fn<F, Internal_allocator>, Internal_allocator>(n_, F{ 2 })),
                            F{ -1 })));
            }

        private:
            memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> n_;
        };
        template <memoc::Allocator Internal_allocator = memoc::Malloc_allocator, Arithmetic F>
        memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator> acot(const memoc::Shared_ptr<Node<F, Internal_allocator>, Internal_allocator>& n)
        {
            return memoc::make_shared<Acot<F, Internal_allocator>, Internal_allocator>(n);
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
