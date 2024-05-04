#ifndef OC_DERIV_H
#define OC_DERIV_H

#include <stdexcept>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <cmath>
#include <ostream>

namespace oc::deriv {
    namespace details {
        template <typename T, template<typename> typename Allocator, typename ...Args>
        [[nodiscard]] std::shared_ptr<T> make_node(Args&&... args)
        {
            return std::allocate_shared<T>(Allocator<T>{}, std::forward<Args>(args)...);
        }

        template <typename T>
        [[nodiscard]] auto zero_value()
        {
            return T{0};
        }

        template <typename T>
        [[nodiscard]] auto unit_value()
        {
            return T{1};
        }

        template <typename T>
        [[nodiscard]] auto full_value(const T& value)
        {
            return T{value};
        }

        struct node_tag { };

        template <typename T>
        concept node_type = std::is_same_v<typename std::remove_cvref_t<T>::tag, node_tag>;

        template <node_type N>
        std::ostream& operator<<(std::ostream& os, const std::shared_ptr<N>& n)
        {
            return n->print(os);
        }

        template <typename T, template <typename> typename Internal_allocator = std::allocator>
        class Const {
        public:
            using tag = node_tag;

            explicit Const(const T& value)
                : value_(value) {}

            auto compute() const
            {
                return value_;
            }

            auto backward(std::int64_t id) const
            {
                return make_node<Const<decltype(zero_value<T>()), Internal_allocator>, Internal_allocator>(
                    zero_value<T>());
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << value_ << ')';
                return os;
            }

        private:
            T value_;
        };
        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto constant(const T& value)
        {
            return make_node<Const<T, Internal_allocator>, Internal_allocator>(value);
        }

        template <typename T, template <typename> typename Internal_allocator = std::allocator>
        class Var {
        public:
            using tag = node_tag;

            explicit Var(std::int64_t id, const T& value = zero_value<T>())
                : id_(id), value_(value) {}

            void set(const T& value)
            {
                value_ = value;
            }

            auto compute() const
            {
                return value_;
            }

            auto backward(std::int64_t id) const
            {
                static_assert(std::is_same_v<decltype(unit_value<T>()), decltype(zero_value<T>())>);

                return id_ == id ? make_node<Const<decltype(unit_value<T>()), Internal_allocator>, Internal_allocator>(
                           unit_value<T>())
                                 : make_node<Const<decltype(zero_value<T>()), Internal_allocator>, Internal_allocator>(
                                     zero_value<T>());
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "x_" << id_;
                return os;
            }

        private:
            std::int64_t id_;
            T value_;
        };
        template <typename T, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto variable(std::int64_t id, const T& value = zero_value<T>())
        {
            return make_node<Var<T, Internal_allocator>, Internal_allocator>(id, value);
        }

        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        class Add {
        public:
            using tag = node_tag;

            Add(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] auto compute() const
            {
                return n1_->compute() + n2_->compute();
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                using n1_type = typename decltype(n1_->backward(id))::element_type;
                using n2_type = typename decltype(n2_->backward(id))::element_type;

                return make_node<Add<n1_type, n2_type, Internal_allocator>, Internal_allocator>(
                    n1_->backward(id), n2_->backward(id));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << n1_ << '+' << n2_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N1> n1_;
            std::shared_ptr<N2> n2_;
        };
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto add(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Add<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator+(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Add<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N, typename T, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator+(const std::shared_ptr<N>& n1, const T& value)
        {
            return make_node<Add<N, Const<T, Internal_allocator>, Internal_allocator>, Internal_allocator>(
                n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <typename T, node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator+(const T& value, const std::shared_ptr<N>& n2)
        {
            return make_node<Add<Const<T, Internal_allocator>, N, Internal_allocator>, Internal_allocator>(
                make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        class Sub {
        public:
            using tag = node_tag;

            Sub(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] auto compute() const
            {
                return n1_->compute() - n2_->compute();
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                using n1_type = typename decltype(n1_->backward(id))::element_type;
                using n2_type = typename decltype(n2_->backward(id))::element_type;

                return make_node<Sub<n1_type, n2_type, Internal_allocator>, Internal_allocator>(
                    n1_->backward(id), n2_->backward(id));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << n1_ << '-' << n2_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N1> n1_;
            std::shared_ptr<N2> n2_;
        };
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto subtract(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Sub<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator-(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Sub<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N, typename T, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator-(const std::shared_ptr<N>& n1, const T& value)
        {
            return make_node<Sub<N, Const<T, Internal_allocator>, Internal_allocator>, Internal_allocator>(
                n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <typename T, node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator-(const T& value, const std::shared_ptr<N>& n2)
        {
            return make_node<Sub<Const<T, Internal_allocator>, N, Internal_allocator>, Internal_allocator>(
                make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Neg {
        public:
            using tag = node_tag;

            Neg(const std::shared_ptr<N>& n)
                : n_(n) {}

            [[nodiscard]] auto compute() const
            {
                return -n_->compute();
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                using n_type = typename decltype(n_->backward(id))::element_type;

                return make_node<Neg<n_type, Internal_allocator>, Internal_allocator>(n_->backward(id));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << '-' << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto negate(const std::shared_ptr<N>& n)
        {
            return make_node<Neg<N, Internal_allocator>, Internal_allocator>(n);
        }
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator-(const std::shared_ptr<N>& n)
        {
            return make_node<Neg<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        class Mul {
        public:
            using tag = node_tag;

            Mul(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] auto compute() const
            {
                return n1_->compute() * n2_->compute();
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                using n1_type = typename decltype(n1_->backward(id))::element_type;
                using n2_type = typename decltype(n2_->backward(id))::element_type;

                return make_node<Mul<n1_type, N2, Internal_allocator>, Internal_allocator>(n1_->backward(id), n2_)
                    + make_node<Mul<N1, n2_type, Internal_allocator>, Internal_allocator>(n1_, n2_->backward(id));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << n1_ << '*' << n2_ << ')';
                return os;
            }
        private:
            std::shared_ptr<N1> n1_;
            std::shared_ptr<N2> n2_;
        };
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto multiply(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Mul<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator*(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Mul<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N, typename T, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator*(const std::shared_ptr<N>& n1, const T& value)
        {
            return make_node<Mul<N, Const<T, Internal_allocator>, Internal_allocator>, Internal_allocator>(
                n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <typename T, node_type N, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator*(const T& value, const std::shared_ptr<N>& n2)
        {
            return make_node<Mul<Const<T, Internal_allocator>, N, Internal_allocator>, Internal_allocator>(
                make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        class Div {
        public:
            using tag = node_tag;

            Div(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] auto compute() const
            {
                auto n2_value = n2_->compute();
                if (n2_value == zero_value<decltype(n2_value)>()) {
                    throw std::overflow_error{ "division by zero" };
                }

                return n1_->compute() / n2_value;
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                using n1_type = typename decltype(n1_->backward(id) * n2_ - n1_ * n2_->backward(id))::element_type;
                using n2_type = typename decltype(n2_ * n2_)::element_type;

                return make_node<Div<n1_type, n2_type, Internal_allocator>, Internal_allocator>(
                    n1_->backward(id) * n2_ - n1_ * n2_->backward(id), n2_ * n2_);
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << n1_ << '/' << n2_ << ')';
                return os;
            }
        private:
            std::shared_ptr<N1> n1_;
            std::shared_ptr<N2> n2_;
        };
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto divide(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Div<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator/(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Div<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N, typename T, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator/(const std::shared_ptr<N>& n1, const T& value)
        {
            return make_node<Div<N, Const<T, Internal_allocator>, Internal_allocator>, Internal_allocator>(
                n1, make_node<Const<T, Internal_allocator>, Internal_allocator>(value));
        }
        template <typename T, node_type N, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator/(const T& value, const std::shared_ptr<N>& n2)
        {
            return make_node<Div<Const<T, Internal_allocator>, N, Internal_allocator>, Internal_allocator>(
                make_node<Const<T, Internal_allocator>, Internal_allocator>(value), n2);
        }

        template <node_type N, template <typename> typename Internal_allocator>
        class Cos;

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Sin {
        public:
            using tag = node_tag;

            Sin(const std::shared_ptr<N>& n)
                : n_(n) {}

            [[nodiscard]] auto compute() const
            {
                using std::sin;
                return sin(n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) * cos(n_);
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "sin(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto sin(const std::shared_ptr<N>& n)
        {
            return make_node<Sin<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Cos {
        public:
            using tag = node_tag;

            Cos(const std::shared_ptr<N>& v)
                : n_(v) {}

            [[nodiscard]] auto compute() const
            {
                using std::cos;
                return cos(n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) * (-sin(n_));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "cos(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto cos(const std::shared_ptr<N>& n)
        {
            return make_node<Cos<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator>
        class Sec;

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Tan {
        public:
            using tag = node_tag;

            Tan(const std::shared_ptr<N>& v)
                : n_(v) {}

            [[nodiscard]] auto compute() const
            {
                using std::tan;
                return tan(n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) * (sec(n_) * sec(n_));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "tan(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto tan(const std::shared_ptr<N>& n)
        {
            return make_node<Tan<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Sec {
        public:
            using tag = node_tag;

            Sec(const std::shared_ptr<N>& v)
                : n_(v) {}

            [[nodiscard]] auto compute() const
            {
                using std::cos;
                auto d = full_value<decltype(cos(n_->compute()))>(cos(n_->compute()));
                if (d == zero_value<decltype(d)>()) {
                    throw std::overflow_error{ "division by zero" };
                }

                return unit_value<decltype(d)>() / d;
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) * (sec(n_) * tan(n_));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "sec(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto sec(const std::shared_ptr<N>& n)
        {
            return make_node<Sec<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template<typename> typename Internal_allocator>
        class Csc;

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Cot {
        public:
            using tag = node_tag;

            Cot(const std::shared_ptr<N>& v)
                : n_(v) {}

            [[nodiscard]] auto compute() const
            {
                using std::tan;
                auto d = full_value<decltype(tan(n_->compute()))>(tan(n_->compute()));
                if (d == zero_value<decltype(d)>()) {
                    throw std::overflow_error{ "division by zero" };
                }

                return unit_value<decltype(d)>() / d;
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) * (-(csc(n_) * csc(n_)));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "cot(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto cot(const std::shared_ptr<N>& n)
        {
            return make_node<Cot<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Csc {
        public:
            using tag = node_tag;

            Csc(const std::shared_ptr<N>& v)
                : n_(v) {}

            [[nodiscard]] auto compute() const
            {
                using std::sin;
                auto d = full_value<decltype(sin(n_->compute()))>(sin(n_->compute()));
                if (d == zero_value<decltype(d)>()) {
                    throw std::overflow_error{ "division by zero" };
                }

                return unit_value<decltype(d)>() / d;
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) * (-(csc(n_) * cot(n_)));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "csc(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        auto csc(const std::shared_ptr<N>& n)
        {
            return make_node<Csc<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Exp {
        public:
            using tag = node_tag;

            Exp(const std::shared_ptr<N>& v)
                : n_(v) {}

            [[nodiscard]] auto compute() const
            {
                using std::exp;
                return exp(n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) * make_node<Exp<N, Internal_allocator>, Internal_allocator>(n_);
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "exp(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto exp(const std::shared_ptr<N>& n)
        {
            return make_node<Exp<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Ln {
        public:
            using tag = node_tag;

            Ln(const std::shared_ptr<N>& v)
                : n_(v) {}

            [[nodiscard]] auto compute() const
            {
                using std::log;
                auto d = full_value<decltype(n_->compute())>(n_->compute());
                if (d <= zero_value<decltype(d)>()) {
                    throw std::overflow_error{ "log of non-positive number" };
                }

                return log(d);
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id) / n_;
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "ln(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto ln(const std::shared_ptr<N>& n)
        {
            return make_node<Ln<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, typename T, template <typename> typename Internal_allocator = std::allocator>
        class Pow_fn {
        public:
            using tag = node_tag;

            Pow_fn(const std::shared_ptr<N>& f, const T& n)
                : f_(f), n_(n) {}

            [[nodiscard]] auto compute() const
            {
                using std::pow;
                return pow(f_->compute(), n_);
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return f_->backward(id)
                    * (make_node<Const<T, Internal_allocator>, Internal_allocator>(n_)
                        * make_node<Pow_fn<N, decltype(n_ - unit_value<T>()), Internal_allocator>, Internal_allocator>(
                            f_, n_ - unit_value<T>()));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << f_ << ")^" << '(' << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> f_;
            T n_;
        };
        template <node_type N, typename T, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto pow(const std::shared_ptr<N>& f, const T& n)
        {
            return make_node<Pow_fn<N, T, Internal_allocator>, Internal_allocator>(f, n);
        }
        template <node_type N, typename T, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator^(const std::shared_ptr<N>& f, const T& n)
        {
            return make_node<Pow_fn<N, T, Internal_allocator>, Internal_allocator>(f, n);
        }

        template <typename T, node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Pow_af {
        public:
            using tag = node_tag;

            Pow_af(const T& a, const std::shared_ptr<N>& f)
                : a_(a), f_(f) {}

            [[nodiscard]] auto compute() const
            {
                using std::pow;
                return pow(a_, f_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                using std::log;
                return f_->backward(id)
                    * (make_node<Pow_af<T, N, Internal_allocator>, Internal_allocator>(a_, f_)
                        * make_node<Const<T, Internal_allocator>, Internal_allocator>(log(a_)));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << a_ << ')' << "^(" << f_ << ')';
                return os;
            }

        private:
            T a_;
            std::shared_ptr<N> f_;
        };
        template <typename T, node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto pow(const T& a, const std::shared_ptr<N>& f)
        {
            return make_node<Pow_af<T, N, Internal_allocator>, Internal_allocator>(a, f);
        }
        template <typename T, node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator^(const T& a, const std::shared_ptr<N>& f)
        {
            return make_node<Pow_af<T, N, Internal_allocator>, Internal_allocator>(a, f);
        }

        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        class Pow_fg {
        public:
            using tag = node_tag;

            Pow_fg(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
                : n1_(n1), n2_(n2) {}

            [[nodiscard]] auto compute() const
            {
                using std::pow;
                return pow(n1_->compute(), n2_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return make_node<Pow_fg<N1, N2, Internal_allocator>, Internal_allocator>(n1_, n2_)
                    * (((n2_ / n1_) * n1_->backward(id)) + (ln(n1_) * n2_->backward(id)));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << '(' << n1_ << ")^" << '(' << n2_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N1> n1_;
            std::shared_ptr<N2> n2_;
        };
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto pow(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Pow_fg<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }
        template <node_type N1, node_type N2, template <typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto operator^(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
        {
            return make_node<Pow_fg<N1, N2, Internal_allocator>, Internal_allocator>(n1, n2);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Asin {
        public:
            using tag = node_tag;

            Asin(const std::shared_ptr<N>& n)
                : n_(n) {}

            [[nodiscard]] auto compute() const
            {
                using std::asin;
                return asin(n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id)
                    * pow((make_node<Const<decltype(n_->compute()), Internal_allocator>, Internal_allocator>(
                               unit_value<decltype(n_->compute())>())
                              - pow(n_, full_value<decltype(n_->compute())>(2))),
                        full_value<decltype(n_->compute())>(-0.5));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "asin(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto asin(const std::shared_ptr<N>& n)
        {
            return make_node<Asin<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Acos {
        public:
            using tag = node_tag;

            Acos(const std::shared_ptr<N>& n)
                : n_(n) {}

            [[nodiscard]] auto compute() const
            {
                using std::acos;
                return acos(n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id)
                    * (-(pow((make_node<Const<decltype(n_->compute()), Internal_allocator>, Internal_allocator>(
                                  unit_value<decltype(n_->compute())>())
                                 - pow(n_, full_value<decltype(n_->compute())>(2))),
                        full_value<decltype(n_->compute())>(-0.5))));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "acos(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto acos(const std::shared_ptr<N>& n)
        {
            return make_node<Acos<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Atan {
        public:
            using tag = node_tag;

            Atan(const std::shared_ptr<N>& n)
                : n_(n) {}

            [[nodiscard]] auto compute() const
            {
                using std::atan;
                return atan(n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id)
                    * pow((make_node<Const<decltype(n_->compute()), Internal_allocator>, Internal_allocator>(
                               unit_value<decltype(n_->compute())>())
                              + pow(n_, full_value<decltype(n_->compute())>(2))),
                        full_value<decltype(n_->compute())>(-1));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "atan(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto atan(const std::shared_ptr<N>& n)
        {
            return make_node<Atan<N, Internal_allocator>, Internal_allocator>(n);
        }

        template <node_type N, template <typename> typename Internal_allocator = std::allocator>
        class Acot {
        public:
            using tag = node_tag;

            Acot(const std::shared_ptr<N>& n)
                : n_(n) {}

            [[nodiscard]] auto compute() const
            {
                using std::atan;
                auto d = full_value<decltype(n_->compute())>(n_->compute());
                if (d == zero_value<decltype(d)>()) {
                    throw std::overflow_error{ "division by zero" };
                }

                return atan(d / n_->compute());
            }

            [[nodiscard]] auto backward(std::int64_t id) const
            {
                return n_->backward(id)
                    * (-(pow((make_node<Const<decltype(n_->compute()), Internal_allocator>, Internal_allocator>(
                                  unit_value<decltype(n_->compute())>())
                                 + pow(n_, full_value<decltype(n_->compute())>(-1))),
                        full_value<decltype(n_->compute())>(-1))));
            }

            std::ostream& print(std::ostream& os) const
            {
                os << "acot(" << n_ << ')';
                return os;
            }

        private:
            std::shared_ptr<N> n_;
        };
        template <node_type N, template<typename> typename Internal_allocator = std::allocator>
        [[nodiscard]] auto acot(const std::shared_ptr<N>& n)
        {
            return make_node<Acot<N, Internal_allocator>, Internal_allocator>(n);
        }
    }

    using details::node_tag;
    using details::node_type;

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

#endif // OC_DERIV_H
