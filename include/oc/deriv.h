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
    [[nodiscard]] T zero_value()
    {
        return T{0};
    }

    template <typename T>
    [[nodiscard]] T unit_value()
    {
        return T{1};
    }

    template <typename T>
    [[nodiscard]] T full_value(const T& value)
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

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Const : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

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
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

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

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Add : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Add(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
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
        std::shared_ptr<Node<T, Internal_allocator>> n1_;
        std::shared_ptr<Node<T, Internal_allocator>> n2_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto add(
        std::shared_ptr<Node<T, Internal_allocator>> n1, std::shared_ptr<Node<T, Internal_allocator>> n2)
    {
        if (n1->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n1);
            if (c && c->compute() == zero_value<T>()) {
                return n2;
            }
        }

        if (n2->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n2);
            if (c && c->compute() == zero_value<T>()) {
                return n1;
            }
        }

        return make_node<Add<T, Internal_allocator>>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator+(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return add<T, Internal_allocator>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator+(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const T& value)
    {
        return add<T, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>>(value));
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator+(const T& value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return add<T, Internal_allocator>(make_node<Const<T, Internal_allocator>>(value), n2);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Sub : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Sub(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
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
        std::shared_ptr<Node<T, Internal_allocator>> n1_;
        std::shared_ptr<Node<T, Internal_allocator>> n2_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto subtract(
        const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        if (n1->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n1);
            if (c && c->compute() == zero_value<T>()) {
                return -n2;
            }
        }

        if (n2->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n2);
            if (c && c->compute() == zero_value<T>()) {
                return n1;
            }
        }

        return make_node<Sub<T, Internal_allocator>>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator-(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return subtract<T, Internal_allocator>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator-(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const T& value)
    {
        return subtract<T, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>>(value));
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator-(const T& value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return subtract<T, Internal_allocator>(make_node<Const<T, Internal_allocator>>(value), n2);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Neg : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Neg(const std::shared_ptr<Node<T, Internal_allocator>>& n)
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto negate(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        if (n->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n);
            if (c && c->compute() == zero_value<T>()) {
                return n;
            }
        }

        return make_node<Neg<T, Internal_allocator>>(n);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator-(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return negate<T, Internal_allocator>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Mul : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Mul(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
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
        std::shared_ptr<Node<T, Internal_allocator>> n1_;
        std::shared_ptr<Node<T, Internal_allocator>> n2_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto multiply(
        const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        if (n1->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n1);
            if (c && c->compute() == zero_value<T>()) {
                return constant<T, Internal_allocator>(
                    zero_value<T>());
            }
            if (c && c->compute() == unit_value<T>()) {
                return n2;
            }
        }

        if (n2->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n2);
            if (c && c->compute() == zero_value<T>()) {
                return constant<T, Internal_allocator>(
                    zero_value<T>());
            }
            if (c && c->compute() == unit_value<T>()) {
                return n1;
            }
        }

        return make_node<Mul<T, Internal_allocator>>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator*(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return multiply<T, Internal_allocator>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator*(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const T& value)
    {
        return multiply<T, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>>(value));
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator*(const T& value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return multiply<T, Internal_allocator>(make_node<Const<T, Internal_allocator>>(value), n2);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Div : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Div(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
            : Node<value_type, allocator_type>(NodeType::divide)
            , n1_(n1)
            , n2_(n2)
        { }

        [[nodiscard]] value_type compute() const override
        {
            auto n2_value = n2_->compute();
            if (n2_value == zero_value<value_type>()) {
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
        std::shared_ptr<Node<T, Internal_allocator>> n1_;
        std::shared_ptr<Node<T, Internal_allocator>> n2_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto divide(
        const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        if (n1->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n1);
            if (c && c->compute() == zero_value<T>() && n2->compute() != zero_value<T>()) {
                return constant<T, Internal_allocator>(zero_value<T>());
            }
        }

        return make_node<Div<T, Internal_allocator>>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator/(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return divide<T, Internal_allocator>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator/(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const T& value)
    {
        return divide<T, Internal_allocator>(n1, make_node<Const<T, Internal_allocator>>(value));
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator/(const T& value, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return divide<T, Internal_allocator>(make_node<Const<T, Internal_allocator>>(value), n2);
    }

    template <typename T, typename Internal_allocator>
    class Cos;

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Sin : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Sin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto sin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Sin<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Cos : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Cos(const std::shared_ptr<Node<T, Internal_allocator>>& v)
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto cos(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Cos<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator>
    class Sec;

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Tan : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Tan(const std::shared_ptr<Node<T, Internal_allocator>>& v)
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto tan(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Tan<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Sec : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Sec(const std::shared_ptr<Node<T, Internal_allocator>>& v)
            : Node<value_type, allocator_type>(NodeType::sec)
            , n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::cos;
            auto d = full_value<value_type>(cos(n_->compute()));
            if (d == zero_value<value_type>()) {
                throw std::overflow_error{"division by zero"};
            }

            return unit_value<value_type>() / d;
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto sec(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Sec<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator>
    class Csc;

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Cot : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Cot(const std::shared_ptr<Node<T, Internal_allocator>>& v)
            : Node<value_type, allocator_type>(NodeType::cot)
            , n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::tan;
            auto d = full_value<value_type>(tan(n_->compute()));
            if (d == zero_value<value_type>()) {
                throw std::overflow_error{"division by zero"};
            }

            return unit_value<value_type>() / d;
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto cot(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Cot<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Csc : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Csc(const std::shared_ptr<Node<T, Internal_allocator>>& v)
            : Node<value_type, allocator_type>(NodeType::csc)
            , n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::sin;
            auto d = full_value<value_type>(sin(n_->compute()));
            if (d == zero_value<value_type>()) {
                throw std::overflow_error{"division by zero"};
            }

            return unit_value<value_type>() / d;
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    auto csc(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Csc<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Exp : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Exp(const std::shared_ptr<Node<T, Internal_allocator>>& v)
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto exp(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Exp<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Ln : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Ln(const std::shared_ptr<Node<T, Internal_allocator>>& v)
            : Node<value_type, allocator_type>(NodeType::ln)
            , n_(v)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::log;
            auto d = full_value<value_type>(n_->compute());
            if (d <= zero_value<value_type>()) {
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
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto ln(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Ln<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Pow_fn : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Pow_fn(const std::shared_ptr<Node<T, Internal_allocator>>& f, const T& n)
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
        std::shared_ptr<Node<T, Internal_allocator>> f_;
        T n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto pow(const std::shared_ptr<Node<T, Internal_allocator>>& f, const T& n)
    {
        if (n == unit_value<T>()) {
            return f;
        }

        if (n == zero_value<T>()) {
            return constant<T, Internal_allocator>(unit_value<T>());
        }

        return make_node<Pow_fn<T, Internal_allocator>>(f, n);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator^(const std::shared_ptr<Node<T, Internal_allocator>>& f, const T& n)
    {
        return pow<T, Internal_allocator>(f, n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Pow_af : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Pow_af(const T& a, const std::shared_ptr<Node<T, Internal_allocator>>& f)
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
        std::shared_ptr<Node<T, Internal_allocator>> f_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto pow(const T& a, const std::shared_ptr<Node<T, Internal_allocator>>& f)
    {
        if (a == unit_value<T>()) {
            return constant<T, Internal_allocator>(unit_value<T>());
        }

        if (a == zero_value<T>()) {
            return constant<T, Internal_allocator>(zero_value<T>());
        }

        return make_node<Pow_af<T, Internal_allocator>>(a, f);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator^(const T& a, const std::shared_ptr<Node<T, Internal_allocator>>& f)
    {
        return pow<T, Internal_allocator>(a, f);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Pow_fg : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Pow_fg(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
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
        std::shared_ptr<Node<T, Internal_allocator>> n1_;
        std::shared_ptr<Node<T, Internal_allocator>> n2_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto pow(
        const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        if (n1->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n1);
            if (c && c->compute() == zero_value<T>()) {
                return constant<T, Internal_allocator>(
                    zero_value<T>());
            }
            if (c && c->compute() == unit_value<T>()) {
                return constant<T, Internal_allocator>(
                    unit_value<T>());
            }
        }

        if (n2->type() == NodeType::constant) {
            auto c = std::dynamic_pointer_cast<Const<T, Internal_allocator>>(n2);
            if (c && c->compute() == zero_value<T>()) {
                return constant<T, Internal_allocator>(
                    unit_value<T>());
            }
            if (c && c->compute() == unit_value<T>()) {
                return n1;
            }
        }

        return make_node<Pow_fg<T, Internal_allocator>>(n1, n2);
    }
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto operator^(const std::shared_ptr<Node<T, Internal_allocator>>& n1, const std::shared_ptr<Node<T, Internal_allocator>>& n2)
    {
        return pow<T, Internal_allocator>(n1, n2);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Asin : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Asin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
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
            auto constant_value = unit_value<value_type>();

            return n_->backward(id)
                * ((constant<value_type, allocator_type>(constant_value) - (n_ ^ full_value<value_type>(2)))
                    ^ full_value<value_type>(-0.5));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "asin(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto asin(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Asin<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Acos : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Acos(const std::shared_ptr<Node<T, Internal_allocator>>& n)
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
            auto constant_value = unit_value<value_type>();

            return n_->backward(id)
                * (-(((constant<value_type, allocator_type>(constant_value) - (n_ ^ full_value<value_type>(2)))
                    ^ full_value<value_type>(-0.5))));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "acos(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto acos(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Acos<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Atan : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Atan(const std::shared_ptr<Node<T, Internal_allocator>>& n)
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
            auto constant_value = unit_value<value_type>();

            return n_->backward(id)
                * ((constant<value_type, allocator_type>(constant_value) + (n_ ^ full_value<value_type>(2)))
                    ^ full_value<value_type>(-1));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "atan(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto atan(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Atan<T, Internal_allocator>>(n);
    }

    template <typename T, typename Internal_allocator = std::allocator<T>>
    class Acot : public Node<T, Internal_allocator> {
    public:
        using value_type = Node<T, Internal_allocator>::value_type;
        using allocator_type = Node<T, Internal_allocator>::allocator_type;

        Acot(const std::shared_ptr<Node<T, Internal_allocator>>& n)
            : Node<value_type, allocator_type>(NodeType::acot)
            , n_(n)
        { }

        [[nodiscard]] value_type compute() const override
        {
            using std::atan;
            auto d = full_value<value_type>(n_->compute());
            if (d == zero_value<value_type>()) {
                throw std::overflow_error{"division by zero"};
            }

            return atan(d / n_->compute());
        }

        [[nodiscard]] std::shared_ptr<Node<value_type, allocator_type>> backward(std::int64_t id) const override
        {
            auto constant_value = unit_value<value_type>();

            return n_->backward(id)
                * (-(((constant<value_type, allocator_type>(constant_value) + (n_ ^ full_value<value_type>(-1)))
                    ^ full_value<value_type>(-1))));
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << "acot(" << n_ << ')';
            return os;
        }

    private:
        std::shared_ptr<Node<T, Internal_allocator>> n_;
    };
    template <typename T, typename Internal_allocator>
    [[nodiscard]] auto acot(const std::shared_ptr<Node<T, Internal_allocator>>& n)
    {
        return make_node<Acot<T, Internal_allocator>>(n);
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
