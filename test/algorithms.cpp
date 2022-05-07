#include <gtest/gtest.h>

#include <memory>

#include <math/core/algorithms.h>
#include <math/core/allocators.h>

TEST(Algorithms_test, two_numbers_can_be_compared_with_specified_percision)
{
    using namespace math::algorithms;

    int a = 0;
    int b = 1;
    EXPECT_FALSE(is_equal(a, b));
    EXPECT_TRUE(is_equal(a, b, 1));

    double c = 0.0;
    double d = 1.0e-10;
    EXPECT_TRUE(is_equal(c, d));
    EXPECT_FALSE(is_equal(c, d, 1.0e-15));
}

TEST(Algorithms_test, can_perform_backward_derivation)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;

    using Allocator = Shared_allocator<Malloc_allocator, 0>;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_add = Add<float, Allocator>;
    using D_mul = Mul<float, Allocator>;
    using D_const = Const<float, Allocator>;
    
    // Z = X^2 + 3xy + 1
    std::shared_ptr<D_node> x = aux::make_shared<Allocator, D_var>(0, 3.0f);
    std::shared_ptr<D_node> y = aux::make_shared<Allocator, D_var>(1, 2.0f);
    std::shared_ptr<D_node> z = aux::make_shared<Allocator, D_add>(
        aux::make_shared<Allocator, D_add>(
            aux::make_shared<Allocator, D_mul>(x, x),
            aux::make_shared<Allocator, D_mul>(
                aux::make_shared<Allocator, D_const>(3.0),
                aux::make_shared<Allocator, D_mul>(x, y))),
        aux::make_shared<Allocator, D_const>(1.0));

    EXPECT_EQ(28.f, z->compute());
    EXPECT_EQ(12.f, z->backward(0)->compute());
    EXPECT_EQ(2.f, z->backward(0)->backward(0)->compute());
    EXPECT_EQ(9.f, z->backward(1)->compute());
}

