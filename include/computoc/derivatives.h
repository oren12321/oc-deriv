#ifndef COMPUTOC_DERIVATIVES_H
#define COMPUTOC_DERIVATIVES_H

#include <stdexcept>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <cmath>

namespace computoc {
    namespace details {
        template <typename T, template<typename> typename Allocator, typename ...Args>
        std::shared_ptr<T> make_node(Args&&... args)
        {
            return std::allocate_shared<T>(Allocator<T>{}, std::forward<Args>(args)...);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        struct Node {
            virtual ~Node() {}
            [[nodiscard]] virtual T compute() const = 0;
            [[nodiscard]] virtual std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const = 0;
            virtual void set(T value) {}
        };

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Const : public Node<T, Internal_allocator>{
        public:
            Const(T value)
                : value_(value) {}

            T compute() const override
            {
                return value_;
            }

            std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Const<T, Internal_allocator>, Internal_allocator>(T{ 0 });
            }

        private:
            T value_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        std::shared_ptr<Node<T, Internal_allocator>> constant(T value)
        {
            return make_node<Const<T, Internal_allocator>, Internal_allocator>(value);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Var : public Node<T, Internal_allocator>{
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

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return id_ == id ?
                    make_node<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }) :
                    make_node<Const<T, Internal_allocator>, Internal_allocator>(T{ 0 });
            }

        private:
            std::int64_t id_;
            T value_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> variable(std::int64_t id, T value = T{ 0 })
        {
            return make_node<Var<T, Internal_allocator>, Internal_allocator>(id, value);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Add : public Node<T, Internal_allocator>{
        public:
            Add(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return n1_->compute() + n2_->compute();
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Add<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n1_;
            std::shared_ptr<Node<T, Internal_allocator>> n2_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>add(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Add<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator+(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Add<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator+(const std::shared_ptr<Node<T, Internal_allocator>>& n1, T value)
        {
            return make_node<Add<T, Internal_allocator>, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator+(T value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Add<T, Internal_allocator>, Internal_allocator>(make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Sub : public Node<T, Internal_allocator>{
        public:
            Sub(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return n1_->compute() - n2_->compute();
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Sub<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_->backward(id));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n1_;
            std::shared_ptr<Node<T, Internal_allocator>> n2_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>subtract(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Sub<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator-(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Sub<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator-(const std::shared_ptr<Node<T, Internal_allocator>>& n1, T value)
        {
            return make_node<Sub<T, Internal_allocator>, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator-(T value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Sub<T, Internal_allocator>, Internal_allocator>(make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Neg : public Node<T, Internal_allocator>{
        public:
            Neg(const std::shared_ptr<Node<T, Internal_allocator>>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return -n_->compute();
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Neg<T, Internal_allocator>, Internal_allocator>(n_->backward(id));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>negate(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Neg<T, Internal_allocator>, Internal_allocator>(n);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator-(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Neg<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Mul : public Node<T, Internal_allocator>{
        public:
            Mul(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return n1_->compute() * n2_->compute();
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Add<T, Internal_allocator>, Internal_allocator>(
                    make_node<Mul<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                    make_node<Mul<T, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id)));
            }
        private:
            std::shared_ptr<Node<T, Internal_allocator>> n1_;
            std::shared_ptr<Node<T, Internal_allocator>> n2_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>multiply(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Mul<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator*(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Mul<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator*(const std::shared_ptr<Node<T, Internal_allocator>>& n1, T value)
        {
            return make_node<Mul<T, Internal_allocator>, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator*(T value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Mul<T, Internal_allocator>, Internal_allocator>(make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Div : public Node<T, Internal_allocator>{
        public:
            Div(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                T n2_value{ n2_->compute() };
                if (n2_value == T{}) {
                    throw std::overflow_error{ "division by zero" };
                }

                return n1_->compute() / n2_value;
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Div<T, Internal_allocator>, Internal_allocator>(
                    make_node<Sub<T, Internal_allocator>, Internal_allocator>(
                        make_node<Mul<T, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_),
                        make_node<Mul<T, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id))),
                    make_node<Mul<T, Internal_allocator>, Internal_allocator>(n2_, n2_));
            }
        private:
            std::shared_ptr<Node<T, Internal_allocator>> n1_;
            std::shared_ptr<Node<T, Internal_allocator>> n2_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>divide(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Div<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator/(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Div<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator/(const std::shared_ptr<Node<T, Internal_allocator>>& n1, T value)
        {
            return make_node<Div<T, Internal_allocator>, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator/(T value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Div<T, Internal_allocator>, Internal_allocator>(make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <typename T, template<typename> typename Internal_allocator>
        class Cos;

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Sin : public Node<T, Internal_allocator>{
        public:
            Sin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return std::sin(n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Cos<T, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>sin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Sin<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Cos : public Node<T, Internal_allocator>{
        public:
            Cos(const std::shared_ptr<Node<T, Internal_allocator>>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                return std::cos(n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Neg<T, Internal_allocator>, Internal_allocator>(
                        make_node<Sin<T, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>cos(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Cos<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator>
        class Sec;

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Tan : public Node<T, Internal_allocator>{
        public:
            Tan(const std::shared_ptr<Node<T, Internal_allocator>>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                return std::tan(n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                        make_node<Sec<T, Internal_allocator>, Internal_allocator>(n_),
                        make_node<Sec<T, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>tan(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Tan<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Sec : public Node<T, Internal_allocator>{
        public:
            Sec(const std::shared_ptr<Node<T, Internal_allocator>>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ std::cos(n_->compute()) };
                if (d == T{}) {
                    throw std::overflow_error{ "division by zero" };
                }

                return T{ 1 } / d;
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                        make_node<Sec<T, Internal_allocator>, Internal_allocator>(n_),
                        make_node<Tan<T, Internal_allocator>, Internal_allocator>(n_)));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>sec(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Sec<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator>
        class Csc;

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Cot : public Node<T, Internal_allocator>{
        public:
            Cot(const std::shared_ptr<Node<T, Internal_allocator>>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ std::tan(n_->compute()) };
                if (d == T{}) {
                    throw std::overflow_error{ "division by zero" };
                }

                return T{ 1 } / d;
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Neg<T, Internal_allocator>, Internal_allocator>(
                        make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                            make_node<Csc<T, Internal_allocator>, Internal_allocator>(n_),
                            make_node<Csc<T, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>cot(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Cot<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Csc : public Node<T, Internal_allocator>{
        public:
            Csc(const std::shared_ptr<Node<T, Internal_allocator>>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ std::sin(n_->compute()) };
                if (d == T{}) {
                    throw std::overflow_error{ "division by zero" };
                }

                return T{ 1 } / d;
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Neg<T, Internal_allocator>, Internal_allocator>(
                        make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                            make_node<Csc<T, Internal_allocator>, Internal_allocator>(n_),
                            make_node<Cot<T, Internal_allocator>, Internal_allocator>(n_))));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        std::shared_ptr<Node<T, Internal_allocator>>csc(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Csc<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Exp : public Node<T, Internal_allocator>{
        public:
            Exp(const std::shared_ptr<Node<T, Internal_allocator>>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                return std::exp(n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Exp<T, Internal_allocator>, Internal_allocator>(n_));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>exp(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Exp<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Ln : public Node<T, Internal_allocator>{
        public:
            Ln(const std::shared_ptr<Node<T, Internal_allocator>>& v)
                : n_(v) {}

            [[nodiscard]] T compute() const override
            {
                T d{ n_->compute() };
                if (d <= T{}) {
                    throw std::overflow_error{ "log of non-positive number" };
                }

                return std::log(d);
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Div<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    n_);
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>ln(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Ln<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Pow_fn : public Node<T, Internal_allocator>{
        public:
            Pow_fn(const std::shared_ptr<Node<T, Internal_allocator>>& f, T n)
                : f_(f), n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return std::pow(f_->compute(), n_);
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                        make_node<Const<T, Internal_allocator>, Internal_allocator>(n_),
                        make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(f_, n_ - T{ 1 })));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> f_;
            T n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>pow(const std::shared_ptr<Node<T, Internal_allocator>>& f, T n)
        {
            return make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(f, n);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator^(const std::shared_ptr<Node<T, Internal_allocator>>& f, T n)
        {
            return make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(f, n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Pow_af : public Node<T, Internal_allocator>{
        public:
            Pow_af(T a, const std::shared_ptr<Node<T, Internal_allocator>>& f)
                : a_(a), f_(f) {}

            [[nodiscard]] T compute() const override
            {
                return std::pow(a_, f_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    f_->backward(id),
                    make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                        make_node<Pow_af<T, Internal_allocator>, Internal_allocator>(a_, f_),
                        make_node<Const<T, Internal_allocator>, Internal_allocator>(log(a_))));
            }

        private:
            T a_;
            std::shared_ptr<Node<T, Internal_allocator>> f_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>pow(T a, const std::shared_ptr<Node<T, Internal_allocator>>& f)
        {
            return make_node<Pow_af<T, Internal_allocator>, Internal_allocator>(a, f);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator^(T a, const std::shared_ptr<Node<T, Internal_allocator>>& f)
        {
            return make_node<Pow_af<T, Internal_allocator>, Internal_allocator>(a, f);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Pow_fg : public Node<T, Internal_allocator>{
        public:
            Pow_fg(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] T compute() const override
            {
                return std::pow(n1_->compute(), n2_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    make_node<Pow_fg<T, Internal_allocator>, Internal_allocator>(n1_, n2_),
                    make_node<Add<T, Internal_allocator>, Internal_allocator>(
                        make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                            make_node<Div<T, Internal_allocator>, Internal_allocator>(n2_, n1_),
                            n1_->backward(0)),
                        make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                            make_node<Ln<T, Internal_allocator>, Internal_allocator>(n1_),
                            n2_->backward(0))));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n1_;
            std::shared_ptr<Node<T, Internal_allocator>> n2_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>pow(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Pow_fg<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>operator^(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
        {
            return make_node<Pow_fg<T, Internal_allocator>, Internal_allocator>(n1, n2);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Asin : public Node<T, Internal_allocator>{
        public:
            Asin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return std::asin(n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                        make_node<Sub<T, Internal_allocator>, Internal_allocator>(
                            make_node<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                            make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                        T{ -0.5 }));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>asin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Asin<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Acos : public Node<T, Internal_allocator>{
        public:
            Acos(const std::shared_ptr<Node<T, Internal_allocator>>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return std::acos(n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Neg<T, Internal_allocator>, Internal_allocator>(
                        make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                            make_node<Sub<T, Internal_allocator>, Internal_allocator>(
                                make_node<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                                make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                            T{ -0.5 })));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>acos(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Acos<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Atan : public Node<T, Internal_allocator>{
        public:
            Atan(const std::shared_ptr<Node<T, Internal_allocator>>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                return std::atan(n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                        make_node<Add<T, Internal_allocator>, Internal_allocator>(
                            make_node<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                            make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                        T{ -1 }));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>atan(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Atan<T, Internal_allocator>, Internal_allocator>(n);
        }

        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        class Acot : public Node<T, Internal_allocator>{
        public:
            Acot(const std::shared_ptr<Node<T, Internal_allocator>>& n)
                : n_(n) {}

            [[nodiscard]] T compute() const override
            {
                T d{ n_->compute() };
                if (d == T{}) {
                    throw std::overflow_error{ "division by zero" };
                }

                return std::atan(d / n_->compute());
            }

            [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const override
            {
                return make_node<Mul<T, Internal_allocator>, Internal_allocator>(
                    n_->backward(id),
                    make_node<Neg<T, Internal_allocator>, Internal_allocator>(
                        make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(
                            make_node<Add<T, Internal_allocator>, Internal_allocator>(
                                make_node<Const<T, Internal_allocator>, Internal_allocator>(T{ 1 }),
                                make_node<Pow_fn<T, Internal_allocator>, Internal_allocator>(n_, T{ 2 })),
                            T{ -1 })));
            }

        private:
            std::shared_ptr<Node<T, Internal_allocator>> n_;
        };
        template <template<typename> typename Internal_allocator = std::allocator , typename T>
        [[nodiscard]] std::shared_ptr<Node<T, Internal_allocator>>acot(const std::shared_ptr<Node<T, Internal_allocator>>& n)
        {
            return make_node<Acot<T, Internal_allocator>, Internal_allocator>(n);
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
