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
    struct Node {
        using tag = node_tag;
        using value_type = T;
        template <typename U>
        using allocator = Internal_allocator<U>;
        using base_type = Node<T, Internal_allocator>;
        template <typename U>
        using replaced_base = Node<U, Internal_allocator>;

        virtual void set(const T&) { }
        virtual value_type compute() const = 0;
        virtual std::shared_ptr<Node<value_type, Internal_allocator>> backward(std::int64_t id) const = 0;
        virtual std::ostream& print(std::ostream& os) const = 0;
    };

    template <node_type N, typename T>
    using replaced_base_t = typename N::template replaced_base<T>;

    template <node_type N, typename... Args>
    [[nodiscard]] std::shared_ptr<N> make_node(Args&&... args)
    {
        return std::allocate_shared<N>(
            typename N::template allocator<typename N::value_type>{}, std::forward<Args>(args)...);
    }

    template <node_type N>
    [[nodiscard]] auto compute_of()
    {
        return ((N*)nullptr)->compute();
    }

    template <typename T, template <typename> typename Internal_allocator = std::allocator>
    class Const : public Node<T, Internal_allocator> {
    public:
        using value_type = T;

        explicit Const(const T& value)
            : value_(value)
        { }

        value_type compute() const override
        {
            return value_;
        }

        std::shared_ptr<Node<value_type, Internal_allocator>> backward(std::int64_t) const override
        {
            return make_node<Const<decltype(zero_value<T>()), Internal_allocator>>(zero_value<T>());
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << value_ << ')';
            return os;
        }

    private:
        T value_;
    };
    template <typename T, template <typename> typename Internal_allocator = std::allocator>
    [[nodiscard]] auto constant(const T& value)
    {
        return make_node<Const<T, Internal_allocator>>(value);
    }

    template <typename T, template <typename> typename Internal_allocator = std::allocator>
    class Var : public Node<T, Internal_allocator> {
    public:
        using value_type = T;

        explicit Var(std::int64_t id, const T& value = zero_value<T>())
            : id_(id)
            , value_(value)
        { }

        void set(const T& value) override
        {
            value_ = value;
        }

        value_type compute() const override
        {
            return value_;
        }

        std::shared_ptr<Node<value_type, Internal_allocator>> backward(std::int64_t id) const override
        {
            static_assert(std::is_same_v<decltype(unit_value<T>()), decltype(zero_value<T>())>);

            return id_ == id ? make_node<Const<decltype(unit_value<T>()), Internal_allocator>>(unit_value<T>())
                             : make_node<Const<decltype(zero_value<T>()), Internal_allocator>>(zero_value<T>());
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "x_" << id_;
            return os;
        }

    private:
        std::int64_t id_;
        T value_;
    };
    template <typename T, template <typename> typename Internal_allocator = std::allocator>
    [[nodiscard]] auto variable(std::int64_t id, const T& value = zero_value<T>())
    {
        return make_node<Var<T, Internal_allocator>>(id, value);
    }

    template <node_type N1, node_type N2>
    class Add : public replaced_base_t<N1, decltype(compute_of<N1>() + compute_of<N2>())> {
    public:
        using value_type = decltype(compute_of<N1>() + compute_of<N2>());

        Add(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return n1_->compute() + n2_->compute();
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N1, value_type>> backward(std::int64_t id) const override
        {
            //return nullptr;
            return n1_->backward(id) + n2_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '+' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N1::base_type> n1_;
        std::shared_ptr<typename N2::base_type> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto add(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Add<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator+(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Add<N1, N2>>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator+(const std::shared_ptr<N>& n1, const T& value)
    {
        return make_node<Add<N, Const<T, N::template allocator>>>(
            n1, make_node<Const<T, N::template allocator>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator+(const T& value, const std::shared_ptr<N>& n2)
    {
        return make_node<Add<Const<T, N::template allocator>, N>>(
            make_node<Const<T, N::template allocator>>(value), n2);
    }

    template <node_type N1, node_type N2>
    class Sub : public replaced_base_t<N1, decltype(compute_of<N1>() - compute_of<N2>())> {
    public:
        using value_type = Node<decltype(compute_of<N1>() - compute_of<N2>())>::value_type;

        Sub(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return n1_->compute() - n2_->compute();
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N1, value_type>> backward(std::int64_t id) const override
        {
            return n1_->backward(id) - n2_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '-' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N1::base_type> n1_;
        std::shared_ptr<typename N2::base_type> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto subtract(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Sub<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator-(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Sub<N1, N2>>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator-(const std::shared_ptr<N>& n1, const T& value)
    {
        return make_node<Sub<N, Const<T, N::template allocator>>>(
            n1, make_node<Const<T, N::template allocator>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator-(const T& value, const std::shared_ptr<N>& n2)
    {
        return make_node<Sub<Const<T, N::template allocator>, N>>(
            make_node<Const<T, N::template allocator>>(value), n2);
    }

    template <node_type N>
    class Neg : public replaced_base_t<N, decltype(-compute_of<N>())> {
    public:
        using value_type = decltype(-compute_of<N>());

        Neg(const std::shared_ptr<N>& n)
            : n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return -n_->compute();
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return -n_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << '-' << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto negate(const std::shared_ptr<N>& n)
    {
        return make_node<Neg<N>>(n);
    }
    template <node_type N>
    [[nodiscard]] auto operator-(const std::shared_ptr<N>& n)
    {
        return make_node<Neg<N>>(n);
    }

    template <node_type N1, node_type N2>
    class Mul : public replaced_base_t<N1, decltype(compute_of<N1>() * compute_of<N2>())> {
    public:
        using value_type = decltype(compute_of<N1>() * compute_of<N2>());

        Mul(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return n1_->compute() * n2_->compute();
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N1, value_type>> backward(std::int64_t id) const override
        {
            return n1_->backward(id) * n2_ + n1_ * n2_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '*' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N1::base_type> n1_;
        std::shared_ptr<typename N2::base_type> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto multiply(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Mul<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator*(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Mul<N1, N2>>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator*(const std::shared_ptr<N>& n1, const T& value)
    {
        return make_node<Mul<N, Const<T, N::template allocator>>>(
            n1, make_node<Const<T, N::template allocator>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator*(const T& value, const std::shared_ptr<N>& n2)
    {
        return make_node<Mul<Const<T, N::template allocator>, N>>(
            make_node<Const<T, N::template allocator>>(value), n2);
    }

    template <node_type N1, node_type N2>
    class Div : public replaced_base_t<N1, decltype(compute_of<N1>() / compute_of<N2>())> {
    public:
        using value_type = decltype(compute_of<N1>() / compute_of<N2>());

        Div(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            auto n2_value = n2_->compute();
            if (n2_value == zero_value<decltype(n2_value)>()) {
                throw std::overflow_error{"division by zero"};
            }

            return n1_->compute() / n2_value;
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N1, value_type>> backward(std::int64_t id) const override
        {
            return (n1_->backward(id) * n2_ - n1_ * n2_->backward(id)) / (n2_ * n2_);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '/' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N1::base_type> n1_;
        std::shared_ptr<typename N2::base_type> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto divide(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Div<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator/(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Div<N1, N2>>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator/(const std::shared_ptr<N>& n1, const T& value)
    {
        return make_node<Div<N, Const<T, N::template allocator>>>(
            n1, make_node<Const<T, N::template allocator>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator/(const T& value, const std::shared_ptr<N>& n2)
    {
        return make_node<Div<Const<T, N::template allocator>, N>>(
            make_node<Const<T, N::template allocator>>(value), n2);
    }

    template <node_type N>
    class Cos;

    using std::sin;
    template <node_type N>
    class Sin : public replaced_base_t<N, decltype(sin(compute_of<N>()))> {
    public:
        using value_type = decltype(sin(compute_of<N>()));

        Sin(const std::shared_ptr<N>& n)
            : n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::sin;
            return sin(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * cos(n_);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "sin(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto sin(const std::shared_ptr<N>& n)
    {
        return make_node<Sin<N>>(n);
    }

    using std::cos;
    template <node_type N>
    class Cos : public replaced_base_t<N, decltype(cos(compute_of<N>()))> {
    public:
        using value_type = decltype(cos(compute_of<N>()));

        Cos(const std::shared_ptr<N>& v)
            : n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::cos;
            return cos(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (-sin(n_));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "cos(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto cos(const std::shared_ptr<N>& n)
    {
        return make_node<Cos<N>>(n);
    }

    template <node_type N>
    class Sec;

    using std::tan;
    template <node_type N>
    class Tan : public replaced_base_t<N, decltype(tan(compute_of<N>()))> {
    public:
        using value_type = decltype(tan(compute_of<N>()));

        Tan(const std::shared_ptr<N>& v)
            : n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::tan;
            return tan(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (sec(n_) * sec(n_));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "tan(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto tan(const std::shared_ptr<N>& n)
    {
        return make_node<Tan<N>>(n);
    }

    using std::cos;
    template <node_type N>
    class Sec : public replaced_base_t<N, decltype(cos(compute_of<N>()))> {
    public:
        using value_type = decltype(cos(compute_of<N>()));

        Sec(const std::shared_ptr<N>& v)
            : n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::cos;
            auto d = full_value<decltype(cos(n_->compute()))>(cos(n_->compute()));
            if (d == zero_value<decltype(d)>()) {
                throw std::overflow_error{"division by zero"};
            }

            return unit_value<decltype(d)>() / d;
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (sec(n_) * tan(n_));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "sec(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto sec(const std::shared_ptr<N>& n)
    {
        return make_node<Sec<N>>(n);
    }

    template <node_type N>
    class Csc;

    using std::tan;
    template <node_type N>
    class Cot : public replaced_base_t<N, decltype(tan(compute_of<N>()))> {
    public:
        using value_type = decltype(tan(compute_of<N>()));

        Cot(const std::shared_ptr<N>& v)
            : n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::tan;
            auto d = full_value<decltype(tan(n_->compute()))>(tan(n_->compute()));
            if (d == zero_value<decltype(d)>()) {
                throw std::overflow_error{"division by zero"};
            }

            return unit_value<decltype(d)>() / d;
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (-(csc(n_) * csc(n_)));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "cot(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto cot(const std::shared_ptr<N>& n)
    {
        return make_node<Cot<N>>(n);
    }

    using std::sin;
    template <node_type N>
    class Csc : public replaced_base_t<N, decltype(sin(compute_of<N>()))> {
    public:
        using value_type = decltype(sin(compute_of<N>()));

        Csc(const std::shared_ptr<N>& v)
            : n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::sin;
            auto d = full_value<decltype(sin(n_->compute()))>(sin(n_->compute()));
            if (d == zero_value<decltype(d)>()) {
                throw std::overflow_error{"division by zero"};
            }

            return unit_value<decltype(d)>() / d;
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (-(csc(n_) * cot(n_)));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "csc(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    auto csc(const std::shared_ptr<N>& n)
    {
        return make_node<Csc<N>>(n);
    }

    using std::exp;
    template <node_type N>
    class Exp : public replaced_base_t<N, decltype(exp(compute_of<N>()))> {
    public:
        using value_type = decltype(exp(compute_of<N>()));

        Exp(const std::shared_ptr<N>& v)
            : n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::exp;
            return exp(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * exp(n_);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "exp(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto exp(const std::shared_ptr<N>& n)
    {
        return make_node<Exp<N>>(n);
    }

    using std::log;
    template <node_type N>
    class Ln : public replaced_base_t<N, decltype(log(compute_of<N>()))> {
    public:
        using value_type = decltype(log(compute_of<N>()));

        Ln(const std::shared_ptr<N>& v)
            : n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::log;
            auto d = full_value<decltype(n_->compute())>(n_->compute());
            if (d <= zero_value<decltype(d)>()) {
                throw std::overflow_error{"log of non-positive number"};
            }

            return log(d);
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) / n_;
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "ln(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto ln(const std::shared_ptr<N>& n)
    {
        return make_node<Ln<N>>(n);
    }

    using std::pow;
    template <node_type N, typename T>
    class Pow_fn : public replaced_base_t<N, decltype(pow(compute_of<N>(), unit_value<T>()))> {
    public:
        using value_type = decltype(pow(compute_of<N>(), unit_value<T>()));

        Pow_fn(const std::shared_ptr<N>& f, const T& n)
            : f_(f)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::pow;
            return pow(f_->compute(), n_);
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            return f_->backward(id) * n_ * (f_ ^ (n_ - unit_value<T>()));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << f_ << ")^" << '(' << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> f_;
        T n_;
    };
    template <node_type N, typename T>
    [[nodiscard]] auto pow(const std::shared_ptr<N>& f, const T& n)
    {
        return make_node<Pow_fn<N, T>>(f, n);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator^(const std::shared_ptr<N>& f, const T& n)
    {
        return make_node<Pow_fn<N, T>>(f, n);
    }

    using std::pow;
    template <typename T, node_type N>
    class Pow_af : public replaced_base_t<N, decltype(pow(unit_value<T>(), compute_of<N>()))> {
    public:
        using value_type = decltype(pow(unit_value<T>(), compute_of<N>()));

        Pow_af(const T& a, const std::shared_ptr<N>& f)
            : a_(a)
            , f_(f)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::pow;
            return pow(a_, f_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            using std::log;
            return f_->backward(id) * (a_ ^ f_) * log(a_);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << a_ << ')' << "^(" << f_ << ')';
            return os;
        }

    private:
        T a_;
        std::shared_ptr<typename N::base_type> f_;
    };
    template <typename T, node_type N>
    [[nodiscard]] auto pow(const T& a, const std::shared_ptr<N>& f)
    {
        return make_node<Pow_af<T, N>>(a, f);
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator^(const T& a, const std::shared_ptr<N>& f)
    {
        return make_node<Pow_af<T, N>>(a, f);
    }

    using std::pow;
    template <node_type N1, node_type N2>
    class Pow_fg : public replaced_base_t<N1, decltype(pow(compute_of<N1>(), compute_of<N2>()))> {
    public:
        using value_type = decltype(pow(compute_of<N1>(), compute_of<N2>()));

        Pow_fg(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::pow;
            return pow(n1_->compute(), n2_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N1, value_type>> backward(std::int64_t id) const override
        {
            return (n1_ ^ n2_) * ((n2_ / n1_) * n1_->backward(id) + ln(n1_) * n2_->backward(id));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << ")^" << '(' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N1::base_type> n1_;
        std::shared_ptr<typename N2::base_type> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto pow(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Pow_fg<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator^(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return make_node<Pow_fg<N1, N2>>(n1, n2);
    }

    using std::asin;
    template <node_type N>
    class Asin : public replaced_base_t<N, decltype(asin(compute_of<N>()))> {
    public:
        using value_type = decltype(asin(compute_of<N>()));

        Asin(const std::shared_ptr<N>& n)
            : n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::asin;
            return asin(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * ((constant<decltype(constant_value), N::template allocator>(constant_value)
                       - (n_ ^ full_value<decltype(n_->compute())>(2)))
                    ^ full_value<decltype(n_->compute())>(-0.5));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "asin(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto asin(const std::shared_ptr<N>& n)
    {
        return make_node<Asin<N>>(n);
    }

    using std::acos;
    template <node_type N>
    class Acos : public replaced_base_t<N, decltype(acos(compute_of<N>()))> {
    public:
        using value_type = decltype(acos(compute_of<N>()));

        Acos(const std::shared_ptr<N>& n)
            : n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::acos;
            return acos(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * (-(((constant<decltype(constant_value), N::template allocator>(constant_value)
                          - (n_ ^ full_value<decltype(n_->compute())>(2)))
                    ^ full_value<decltype(n_->compute())>(-0.5))));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "acos(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto acos(const std::shared_ptr<N>& n)
    {
        return make_node<Acos<N>>(n);
    }

    using std::atan;
    template <node_type N>
    class Atan : public replaced_base_t<N, decltype(atan(compute_of<N>()))> {
    public:
        using value_type = decltype(atan(compute_of<N>()));

        Atan(const std::shared_ptr<N>& n)
            : n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::atan;
            return atan(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * ((constant<decltype(constant_value), N::template allocator>(constant_value)
                       + (n_ ^ full_value<decltype(n_->compute())>(2)))
                    ^ full_value<decltype(n_->compute())>(-1));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "atan(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto atan(const std::shared_ptr<N>& n)
    {
        return make_node<Atan<N>>(n);
    }

    using std::atan;
    template <node_type N>
    class Acot : public replaced_base_t<N, decltype(atan(compute_of<N>()))> {
    public:
        using value_type = decltype(atan(compute_of<N>()));

        Acot(const std::shared_ptr<N>& n)
            : n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::atan;
            auto d = full_value<decltype(n_->compute())>(n_->compute());
            if (d == zero_value<decltype(d)>()) {
                throw std::overflow_error{"division by zero"};
            }

            return atan(d / n_->compute());
        }

        [[nodiscard]] std::shared_ptr<replaced_base_t<N, value_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * (-(((constant<decltype(constant_value), N::template allocator>(constant_value)
                          + (n_ ^ full_value<decltype(n_->compute())>(-1)))
                    ^ full_value<decltype(n_->compute())>(-1))));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "acot(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<typename N::base_type> n_;
    };
    template <node_type N>
    [[nodiscard]] auto acot(const std::shared_ptr<N>& n)
    {
        return make_node<Acot<N>>(n);
    }
}

using details::node_tag;
using details::node_type;

using details::zero_value;
using details::unit_value;
using details::full_value;

using details::Node;

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
