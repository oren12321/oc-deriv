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

    template <typename, typename>
    struct replace_targ { };

    template <typename A, template <typename> typename C, typename B>
    struct replace_targ<A, C<B>> {
        using type = C<A>;
    };

    template <typename R, typename T>
    using replace_targ_t = typename replace_targ<R, T>::type;

    enum class NodeType {
        acos,
        acot,
        add,
        asin,
        atan,
        constant,
        cos,
        cot,
        csc,
        divide,
        exp,
        ln,
        multiply,
        negate,
        pow_af,
        pow_fg,
        pow_fn,
        sec,
        sin,
        subtract,
        tan,
        variable,

        unknown,
    };

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Node {
    public:
        using tag = node_tag;

        using value_type = T;
        using allocator_type = Internal_allocator;

        Node(NodeType type)
            : type_(type)
        { }

        virtual void set(std::int64_t, const T&) { }
        virtual value_type compute() const = 0;
        virtual std::shared_ptr<Node<T, Internal_allocator>> backward(std::int64_t id) const = 0;
        virtual std::ostream& print(std::ostream& os) const = 0;

        NodeType type() const
        {
            return type_;
        }

    protected:
        NodeType type_ = NodeType::unknown;
    };

    template <node_type N, typename... Args>
    [[nodiscard]] std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> make_node(Args&&... args)
    {
        return std::allocate_shared<N>(typename N::allocator_type{}, std::forward<Args>(args)...);
    }

    template <node_type N>
    [[nodiscard]] auto compute_of()
    {
        return ((N*)nullptr)->compute();
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Const : public Node<T, Internal_allocator> {
    public:
        using value_type = T;
        using allocator_type = Internal_allocator;

        explicit Const(const T& value)
            : Node<value_type, allocator_type>(NodeType::constant)
            , value_(value)
        { }

        value_type compute() const override
        {
            return value_;
        }

        std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t) const override
        {
            static_assert(std::is_same_v<value_type, decltype(zero_value<value_type>())>);

            return make_node<Const<value_type, allocator_type>>(zero_value<value_type>());
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << value_ << ')';
            return os;
        }

    private:
        value_type value_;
    };
    template <typename T, typename Internal_allocator = std::allocator<T>>
    [[nodiscard]] auto constant(const T& value)
    {
        return make_node<Const<T, Internal_allocator>>(value);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Var : public Node<T, Internal_allocator> {
    public:
        using value_type = T;
        using allocator_type = Internal_allocator;

        explicit Var(std::int64_t id, const value_type& value = zero_value<value_type>())
            : Node<value_type, allocator_type>(NodeType::variable)
            , id_(id)
            , value_(value)
        { }

        void set(std::int64_t id, const value_type& value) override
        {
            if (id == id_) {
                value_ = value;
            }
        }

        value_type compute() const override
        {
            return value_;
        }

        std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            static_assert(std::is_same_v<value_type, decltype(zero_value<T>())>);
            static_assert(std::is_same_v<decltype(zero_value<T>()), decltype(unit_value<T>())>);

            return id_ == id ? make_node<Const<value_type, allocator_type>>(unit_value<T>())
                             : make_node<Const<value_type, allocator_type>>(zero_value<T>());
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "x_" << id_;
            return os;
        }

    private:
        std::int64_t id_;
        value_type value_;
    };
    template <typename T, typename Internal_allocator = std::allocator<T>>
    [[nodiscard]] auto variable(std::int64_t id, const T& value = zero_value<T>())
    {
        return make_node<Var<T, Internal_allocator>>(id, value);
    }

    template <node_type N1, node_type N2>
    class Add : public Node<decltype(compute_of<N1>() + compute_of<N2>()),
                    replace_targ_t<decltype(compute_of<N1>() + compute_of<N2>()), typename N1::allocator_type>> {
    public:
        using value_type = decltype(compute_of<N1>() + compute_of<N2>());
        using allocator_type = replace_targ_t<value_type, typename N1::allocator_type>;

        Add(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : Node<value_type, allocator_type>(NodeType::add)
            , n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return n1_->compute() + n2_->compute();
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n1_->backward(id) + n2_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '+' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N1::value_type, typename N1::allocator_type>> n1_;
        std::shared_ptr<Node<typename N2::value_type, typename N2::allocator_type>> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto add(std::shared_ptr<N1> n1, std::shared_ptr<N2> n2)
    {
        if constexpr (std::is_same_v<typename N1::value_type, typename N2::value_type>) {
            if (n1->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N1::value_type, typename N1::allocator_type>>(n1);
                if (c && c->compute() == zero_value<typename N1::value_type>()) {
                    return n2;
                }
            }

            if (n2->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N2::value_type, typename N2::allocator_type>>(n2);
                if (c && c->compute() == zero_value<typename N2::value_type>()) {
                    return n1;
                }
            }
        }

        return make_node<Add<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator+(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return add<N1, N2>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator+(const std::shared_ptr<N>& n1, const T& value)
    {
        return add<N, Node<T, replace_targ_t<T, typename N::allocator_type>>>(
            n1, make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator+(const T& value, const std::shared_ptr<N>& n2)
    {
        return add<Node<T, replace_targ_t<T, typename N::allocator_type>>, N>(
            make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value), n2);
    }

    template <node_type N1, node_type N2>
    class Sub : public Node<decltype(compute_of<N1>() - compute_of<N2>()),
                    replace_targ_t<decltype(compute_of<N1>() - compute_of<N2>()), typename N1::allocator_type>> {
    public:
        using value_type = decltype(compute_of<N1>() - compute_of<N2>());
        using allocator_type = replace_targ_t<value_type, typename N1::allocator_type>;

        Sub(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : Node<value_type, allocator_type>(NodeType::subtract)
            , n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return n1_->compute() - n2_->compute();
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n1_->backward(id) - n2_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '-' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N1::value_type, typename N1::allocator_type>> n1_;
        std::shared_ptr<Node<typename N2::value_type, typename N2::allocator_type>> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto subtract(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        if constexpr (std::is_same_v<typename N1::value_type, typename N2::value_type>) {
            if (n1->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N1::value_type, typename N1::allocator_type>>(n1);
                if (c && c->compute() == zero_value<typename N1::value_type>()) {
                    return -n2;
                }
            }

            if (n2->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N2::value_type, typename N2::allocator_type>>(n2);
                if (c && c->compute() == zero_value<typename N2::value_type>()) {
                    return n1;
                }
            }
        }

        return make_node<Sub<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator-(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return subtract<N1, N2>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator-(const std::shared_ptr<N>& n1, const T& value)
    {
        return subtract<N, Node<T, replace_targ_t<T, typename N::allocator_type>>>(
            n1, make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator-(const T& value, const std::shared_ptr<N>& n2)
    {
        return subtract<Node<T, replace_targ_t<T, typename N::allocator_type>>, N>(
            make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value), n2);
    }

    template <node_type N>
    class Neg : public Node<decltype(-compute_of<N>()), replace_targ_t<decltype(-compute_of<N>()), typename N::allocator_type>> {
    public:
        using value_type = decltype(-compute_of<N>());
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Neg(const std::shared_ptr<N>& n)
            : Node<value_type, allocator_type>(NodeType::negate)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return -n_->compute();
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return -n_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << '-' << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto negate(const std::shared_ptr<N>& n)
    {
        if (n->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<typename N::value_type, typename N::allocator_type>>(n);
            if (c && c->compute() == zero_value<typename N::value_type>()) {
                return n;
            }
        }

        return make_node<Neg<N>>(n);
    }
    template <node_type N>
    [[nodiscard]] auto operator-(const std::shared_ptr<N>& n)
    {
        return negate<N>(n);
    }

    template <node_type N1, node_type N2>
    class Mul : public Node<decltype(compute_of<N1>() * compute_of<N2>()), replace_targ_t<decltype(compute_of<N1>() * compute_of<N2>()), typename N1::allocator_type>> {
    public:
        using value_type = decltype(compute_of<N1>() * compute_of<N2>());
        using allocator_type = replace_targ_t<value_type, typename N1::allocator_type>;

        Mul(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : Node<value_type, allocator_type>(NodeType::multiply)
            , n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            return n1_->compute() * n2_->compute();
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n1_->backward(id) * n2_ + n1_ * n2_->backward(id);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '*' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N1::value_type, typename N1::allocator_type>> n1_;
        std::shared_ptr<Node<typename N2::value_type, typename N2::allocator_type>> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto multiply(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        if constexpr (std::is_same_v<typename N1::value_type, typename N2::value_type>) {
            if (n1->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N1::value_type, typename N1::allocator_type>>(n1);
                if (c && c->compute() == zero_value<typename N1::value_type>()) {
                    return constant<typename N2::value_type, typename N2::allocator_type>(
                        zero_value<typename N2::value_type>());
                }
                if (c && c->compute() == unit_value<typename N1::value_type>()) {
                    return n2;
                }
            }

            if (n2->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N2::value_type, typename N2::allocator_type>>(n2);
                if (c && c->compute() == zero_value<typename N2::value_type>()) {
                    return constant<typename N1::value_type, typename N1::allocator_type>(
                        zero_value<typename N1::value_type>());
                }
                if (c && c->compute() == unit_value<typename N2::value_type>()) {
                    return n1;
                }
            }
        }

        return make_node<Mul<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator*(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return multiply<N1, N2>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator*(const std::shared_ptr<N>& n1, const T& value)
    {
        return multiply<N, Node<T, replace_targ_t<T, typename N::allocator_type>>>(
            n1, make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator*(const T& value, const std::shared_ptr<N>& n2)
    {
        return multiply<Node<T, replace_targ_t<T, typename N::allocator_type>>, N>(
            make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value), n2);
    }

    template <node_type N1, node_type N2>
    class Div : public Node<decltype(compute_of<N1>() / compute_of<N2>()), replace_targ_t<decltype(compute_of<N1>() / compute_of<N2>()), typename N1::allocator_type>> {
    public:
        using value_type = decltype(compute_of<N1>() / compute_of<N2>());
        using allocator_type = replace_targ_t<value_type, typename N1::allocator_type>;

        Div(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : Node<value_type, allocator_type>(NodeType::divide)
            , n1_(n1)
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

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return (n1_->backward(id) * n2_ - n1_ * n2_->backward(id)) / (n2_ * n2_);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << '/' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N1::value_type, typename N1::allocator_type>> n1_;
        std::shared_ptr<Node<typename N2::value_type, typename N2::allocator_type>> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto divide(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        if constexpr (std::is_same_v<typename N1::value_type, typename N2::value_type>) {
            if (n1->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N1::value_type, typename N1::allocator_type>>(n1);
                if (c && c->compute() == zero_value<typename N1::value_type>() && n2->compute() != zero_value<typename N2::value_type>()) {
                    return constant<typename N1::value_type, typename N1::allocator_type>(
                        zero_value<typename N1::value_type>());
                }
            }
        }

        return make_node<Div<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator/(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return divide<N1, N2>(n1, n2);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator/(const std::shared_ptr<N>& n1, const T& value)
    {
        return divide<N, Node<T, replace_targ_t<T, typename N::allocator_type>>>(
            n1, make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value));
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator/(const T& value, const std::shared_ptr<N>& n2)
    {
        return divide<Node<T, replace_targ_t<T, typename N::allocator_type>>, N>(
            make_node<Const<T, replace_targ_t<T, typename N::allocator_type>>>(value), n2);
    }

    template <node_type N>
    class Cos;

    using std::sin;
    template <node_type N>
    class Sin : public Node<decltype(sin(compute_of<N>())), replace_targ_t<decltype(sin(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(sin(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Sin(const std::shared_ptr<N>& n)
            : Node<value_type, allocator_type>(NodeType::sin)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::sin;
            return sin(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * cos(n_);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "sin(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto sin(const std::shared_ptr<N>& n)
    {
        return make_node<Sin<N>>(n);
    }

    using std::cos;
    template <node_type N>
    class Cos : public Node<decltype(cos(compute_of<N>())), replace_targ_t<decltype(cos(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(cos(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Cos(const std::shared_ptr<N>& v)
            : Node<value_type, allocator_type>(NodeType::cos)
            , n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::cos;
            return cos(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (-sin(n_));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "cos(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
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
    class Tan : public Node<decltype(tan(compute_of<N>())), replace_targ_t<decltype(tan(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(tan(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Tan(const std::shared_ptr<N>& v)
            : Node<value_type, allocator_type>(NodeType::tan)
            , n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::tan;
            return tan(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (sec(n_) * sec(n_));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "tan(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto tan(const std::shared_ptr<N>& n)
    {
        return make_node<Tan<N>>(n);
    }

    using std::cos;
    template <node_type N>
    class Sec : public Node<decltype(cos(compute_of<N>())), replace_targ_t<decltype(cos(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(cos(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Sec(const std::shared_ptr<N>& v)
            : Node<value_type, allocator_type>(NodeType::sec)
            , n_(v)
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

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (sec(n_) * tan(n_));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "sec(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
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
    class Cot : public Node<decltype(tan(compute_of<N>())), replace_targ_t<decltype(tan(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(tan(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Cot(const std::shared_ptr<N>& v)
            : Node<value_type, allocator_type>(NodeType::cot)
            , n_(v)
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

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (-(csc(n_) * csc(n_)));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "cot(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto cot(const std::shared_ptr<N>& n)
    {
        return make_node<Cot<N>>(n);
    }

    using std::sin;
    template <node_type N>
    class Csc : public Node<decltype(sin(compute_of<N>())), replace_targ_t<decltype(sin(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(sin(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Csc(const std::shared_ptr<N>& v)
            : Node<value_type, allocator_type>(NodeType::csc)
            , n_(v)
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

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * (-(csc(n_) * cot(n_)));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "csc(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    auto csc(const std::shared_ptr<N>& n)
    {
        return make_node<Csc<N>>(n);
    }

    using std::exp;
    template <node_type N>
    class Exp : public Node<decltype(exp(compute_of<N>())), replace_targ_t<decltype(exp(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(exp(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Exp(const std::shared_ptr<N>& v)
            : Node<value_type, allocator_type>(NodeType::exp)
            , n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::exp;
            return exp(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) * exp(n_);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "exp(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto exp(const std::shared_ptr<N>& n)
    {
        return make_node<Exp<N>>(n);
    }

    using std::log;
    template <node_type N>
    class Ln : public Node<decltype(log(compute_of<N>())), replace_targ_t<decltype(log(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(log(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Ln(const std::shared_ptr<N>& v)
            : Node<value_type, allocator_type>(NodeType::ln)
            , n_(v)
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

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return n_->backward(id) / n_;
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "ln(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto ln(const std::shared_ptr<N>& n)
    {
        return make_node<Ln<N>>(n);
    }

    using std::pow;
    template <node_type N, typename T>
    class Pow_fn : public Node<decltype(pow(compute_of<N>(), unit_value<T>())), replace_targ_t<decltype(pow(compute_of<N>(), unit_value<T>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(pow(compute_of<N>(), unit_value<T>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Pow_fn(const std::shared_ptr<N>& f, const T& n)
            : Node<value_type, allocator_type>(NodeType::pow_fn)
            , f_(f)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::pow;
            return pow(f_->compute(), n_);
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return f_->backward(id) * n_ * (f_ ^ (n_ - unit_value<T>()));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << f_ << ")^" << '(' << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> f_;
        T n_;
    };
    template <node_type N, typename T>
    [[nodiscard]] auto pow(const std::shared_ptr<N>& f, const T& n)
    {
        if constexpr (std::is_same_v<typename N::value_type, T>) {
            if (n == unit_value<T>()) {
                return f;
            }

            if (n == zero_value<T>()) {
                return constant<typename N::value_type, typename N::allocator_type>(
                    unit_value<typename N::value_type>());
            }
        }

        return make_node<Pow_fn<N, T>>(f, n);
    }
    template <node_type N, typename T>
    [[nodiscard]] auto operator^(const std::shared_ptr<N>& f, const T& n)
    {
        return pow<N, T>(f, n);
    }

    using std::pow;
    template <typename T, node_type N>
    class Pow_af : public Node<decltype(pow(unit_value<T>(), compute_of<N>())), replace_targ_t<decltype(pow(unit_value<T>(), compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(pow(unit_value<T>(), compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Pow_af(const T& a, const std::shared_ptr<N>& f)
            : Node<value_type, allocator_type>(NodeType::pow_af)
            , a_(a)
            , f_(f)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::pow;
            return pow(a_, f_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
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
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> f_;
    };
    template <typename T, node_type N>
    [[nodiscard]] auto pow(const T& a, const std::shared_ptr<N>& f)
    {
        if constexpr (std::is_same_v<typename N::value_type, T>) {
            if (a == unit_value<T>()) {
                return constant<typename N::value_type, typename N::allocator_type>(
                    unit_value<typename N::value_type>());
            }

            if (a == zero_value<T>()) {
                return constant<typename N::value_type, typename N::allocator_type>(
                    zero_value<typename N::value_type>());
            }
        }

        return make_node<Pow_af<T, N>>(a, f);
    }
    template <typename T, node_type N>
    [[nodiscard]] auto operator^(const T& a, const std::shared_ptr<N>& f)
    {
        return pow<T, N>(a, f);
    }

    using std::pow;
    template <node_type N1, node_type N2>
    class Pow_fg : public Node<decltype(pow(compute_of<N1>(), compute_of<N2>())), replace_targ_t<decltype(pow(compute_of<N1>(), compute_of<N2>())), typename N1::allocator_type>> {
    public:
        using value_type = decltype(pow(compute_of<N1>(), compute_of<N2>()));
        using allocator_type = replace_targ_t<value_type, typename N1::allocator_type>;

        Pow_fg(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
            : Node<value_type, allocator_type>(NodeType::pow_fg)
            , n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::pow;
            return pow(n1_->compute(), n2_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            return (n1_ ^ n2_) * ((n2_ / n1_) * n1_->backward(id) + ln(n1_) * n2_->backward(id));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << '(' << n1_ << ")^" << '(' << n2_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N1::value_type, typename N1::allocator_type>> n1_;
        std::shared_ptr<Node<typename N2::value_type, typename N2::allocator_type>> n2_;
    };
    template <node_type N1, node_type N2>
    [[nodiscard]] auto pow(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        if constexpr (std::is_same_v<typename N1::value_type, typename N2::value_type>) {
            if (n1->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N1::value_type, typename N1::allocator_type>>(n1);
                if (c && c->compute() == zero_value<typename N1::value_type>()) {
                    return constant<typename N1::value_type, typename N1::allocator_type>(
                        zero_value<typename N1::value_type>());
                }
                if (c && c->compute() == unit_value<typename N1::value_type>()) {
                    return constant<typename N1::value_type, typename N1::allocator_type>(
                        unit_value<typename N1::value_type>());
                }
            }

            if (n2->type() == NodeType::constant) {
                auto c = std::dynamic_pointer_cast<Const<typename N2::value_type, typename N2::allocator_type>>(n2);
                if (c && c->compute() == zero_value<typename N2::value_type>()) {
                    return constant<typename N1::value_type, typename N1::allocator_type>(
                        unit_value<typename N1::value_type>());
                }
                if (c && c->compute() == unit_value<typename N2::value_type>()) {
                    return n1;
                }
            }
        }

        return make_node<Pow_fg<N1, N2>>(n1, n2);
    }
    template <node_type N1, node_type N2>
    [[nodiscard]] auto operator^(const std::shared_ptr<N1>& n1, const std::shared_ptr<N2>& n2)
    {
        return pow<N1, N2>(n1, n2);
    }

    using std::asin;
    template <node_type N>
    class Asin : public Node<decltype(asin(compute_of<N>())), replace_targ_t<decltype(asin(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(asin(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Asin(const std::shared_ptr<N>& n)
            : Node<value_type, allocator_type>(NodeType::asin)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::asin;
            return asin(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * ((constant<decltype(constant_value),
                        replace_targ_t<decltype(constant_value), typename N::allocator_type>>(constant_value)
                       - (n_ ^ full_value<decltype(n_->compute())>(2)))
                    ^ full_value<decltype(n_->compute())>(-0.5));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "asin(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto asin(const std::shared_ptr<N>& n)
    {
        return make_node<Asin<N>>(n);
    }

    using std::acos;
    template <node_type N>
    class Acos : public Node<decltype(acos(compute_of<N>())), replace_targ_t<decltype(acos(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(acos(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Acos(const std::shared_ptr<N>& n)
            : Node<value_type, allocator_type>(NodeType::acos)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::acos;
            return acos(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * (-(((constant<decltype(constant_value),
                           replace_targ_t<decltype(constant_value), typename N::allocator_type>>(constant_value)
                          - (n_ ^ full_value<decltype(n_->compute())>(2)))
                    ^ full_value<decltype(n_->compute())>(-0.5))));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "acos(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto acos(const std::shared_ptr<N>& n)
    {
        return make_node<Acos<N>>(n);
    }

    using std::atan;
    template <node_type N>
    class Atan : public Node<decltype(atan(compute_of<N>())), replace_targ_t<decltype(atan(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(atan(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Atan(const std::shared_ptr<N>& n)
            : Node<value_type, allocator_type>(NodeType::atan)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::atan;
            return atan(n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * ((constant<decltype(constant_value),
                        replace_targ_t<decltype(constant_value), typename N::allocator_type>>(constant_value)
                       + (n_ ^ full_value<decltype(n_->compute())>(2)))
                    ^ full_value<decltype(n_->compute())>(-1));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "atan(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
    };
    template <node_type N>
    [[nodiscard]] auto atan(const std::shared_ptr<N>& n)
    {
        return make_node<Atan<N>>(n);
    }

    using std::atan;
    template <node_type N>
    class Acot : public Node<decltype(atan(compute_of<N>())), replace_targ_t<decltype(atan(compute_of<N>())), typename N::allocator_type>> {
    public:
        using value_type = decltype(atan(compute_of<N>()));
        using allocator_type = replace_targ_t<value_type, typename N::allocator_type>;

        Acot(const std::shared_ptr<N>& n)
            : Node<value_type, allocator_type>(NodeType::acot)
            , n_(n)
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

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<decltype(n_->compute())>();

            return n_->backward(id)
                * (-(((constant<decltype(constant_value),
                           replace_targ_t<decltype(constant_value), typename N::allocator_type>>(constant_value)
                          + (n_ ^ full_value<decltype(n_->compute())>(-1)))
                    ^ full_value<decltype(n_->compute())>(-1))));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "acot(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<typename N::value_type, typename N::allocator_type>> n_;
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
