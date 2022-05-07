#ifndef MATH_ALGORITHMS_H
#define MATH_ALGORITHMS_H

#include <type_traits>
#include <cmath>
#include <limits>
#include <memory>

#include <math/core/allocators.h>

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
        virtual F compute() = 0;
        virtual std::shared_ptr<Node<F, Internal_allocator>> backward(std::size_t id) = 0;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Const : public Node<F, Internal_allocator> {
    public:
        Const(F value)
            : value_(value) {}

        F compute() override
        {
            return value_;
        }

        std::shared_ptr<Node<F, Internal_allocator>> backward(std::size_t id) override
        {
            return math::core::allocators::aux::make_shared<Internal_allocator, Const<F, Internal_allocator>>(F{ 0 });
        }

    private:
        F value_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Var : public Node<F, Internal_allocator> {
    public:
        Var(std::size_t id, F value)
            : id_(id), value_(value) {}

        F compute() override
        {
            return value_;
        }

        std::shared_ptr<Node<F, Internal_allocator>> backward(std::size_t id) override
        {
            return id_ == id ? 
                math::core::allocators::aux::make_shared<Internal_allocator, Const<F, Internal_allocator>>(F{ 1 }) :
                math::core::allocators::aux::make_shared<Internal_allocator, Const<F, Internal_allocator>>(F{ 0 });
        }

    private:
        std::size_t id_;
        double value_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Add : public Node<F, Internal_allocator> {
    public:
        Add(const std::shared_ptr<Node<F, Internal_allocator>>& n1, const std::shared_ptr<Node<F, Internal_allocator>>& n2)
            : n1_(n1), n2_(n2) {}

        F compute() override
        {
            return n1_->compute() + n2_->compute();
        }

        std::shared_ptr<Node<F, Internal_allocator>> backward(std::size_t id)
        {
            return math::core::allocators::aux::make_shared<Internal_allocator, Add<F, Internal_allocator>>(n1_->backward(id), n2_->backward(id));
        }

    private:
        std::shared_ptr<Node<F, Internal_allocator>> n1_;
        std::shared_ptr<Node<F, Internal_allocator>> n2_;
    };

    template <Decimal F, math::core::allocators::Allocator Internal_allocator>
    class Mul : public Node<F, Internal_allocator> {
    public:
        Mul(const std::shared_ptr<Node<F, Internal_allocator>>& n1, const std::shared_ptr<Node<F, Internal_allocator>>& n2)
            : n1_(n1), n2_(n2) {}

        F compute() override
        {
            return n1_->compute() * n2_->compute();
        }

        std::shared_ptr<Node<F, Internal_allocator>> backward(std::size_t id) override
        {
            return math::core::allocators::aux::make_shared<Internal_allocator, Add<F, Internal_allocator>>(
                math::core::allocators::aux::make_shared<Internal_allocator, Mul<F, Internal_allocator>>(n1_->backward(id), n2_),
                math::core::allocators::aux::make_shared<Internal_allocator, Mul<F, Internal_allocator>>(n1_, n2_->backward(id)));
        }
    private:
        std::shared_ptr<Node<F, Internal_allocator>> n1_;
        std::shared_ptr<Node<F, Internal_allocator>> n2_;
    };

}

#endif // MATH_ALGORITHMS_H