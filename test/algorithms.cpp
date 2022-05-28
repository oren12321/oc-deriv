#include <gtest/gtest.h>

#include <memory>
#include <cmath>

#include <math/core/algorithms.h>
#include <math/core/allocators.h>
#include <math/core/pointers.h>

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

TEST(Algorithm_test, constant_backward_derivative)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Malloc_allocator;
    using D_node = Node<float, Allocator>;
    using D_const = Const<float, Allocator>;

    Shared_ptr<D_const, Allocator> c = Shared_ptr<D_const, Allocator>::make_shared(1.f);

    EXPECT_EQ(1.f, c->compute());
    EXPECT_EQ(0.f, c->backward(0)->compute());
}

TEST(Algorithm_test, variable_backward_derivative)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Malloc_allocator;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);

    EXPECT_EQ(1.f, v->compute());
    EXPECT_EQ(1.f, v->backward(0)->compute());
}

TEST(Algorithm_test, addition_backward_derivative)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Malloc_allocator;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_add = Add<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);

    Shared_ptr<D_add, Allocator> a = Shared_ptr<D_add, Allocator>::make_shared(v1, v2);

    EXPECT_EQ(v1->compute() + v2->compute(), a->compute());
    EXPECT_EQ(v1->backward(0)->compute() + v2->backward(0)->compute(), a->backward(0)->compute());
}

TEST(Algorithm_test, subtraction_backward_derivative)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Malloc_allocator;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_sub = Sub<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);

    Shared_ptr<D_sub, Allocator> s = Shared_ptr<D_sub, Allocator>::make_shared(v1, v2);

    EXPECT_EQ(v1->compute() - v2->compute(), s->compute());
    EXPECT_EQ(v1->backward(0)->compute() - v2->backward(0)->compute(), s->backward(0)->compute());
}

TEST(Algorithm_test, negation_backward_derivative)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Malloc_allocator;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_neg = Neg<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);

    Shared_ptr<D_neg, Allocator> n = Shared_ptr<D_neg, Allocator>::make_shared(v);

    EXPECT_EQ(-1.f, n->compute());
    EXPECT_EQ(-1.f, n->backward(0)->compute());
}

TEST(Algorithm_test, multiplication_backward_derivative)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Malloc_allocator;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_mul = Mul<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);

    Shared_ptr<D_mul, Allocator> m = Shared_ptr<D_mul, Allocator>::make_shared(v1, v2);

    EXPECT_EQ(v1->compute() * v2->compute(), m->compute());
    EXPECT_EQ(v1->backward(0)->compute() * v2->compute() + v1->compute() * v2->backward(0)->compute(), m->backward(0)->compute());
}

TEST(Algorithm_test, division_backward_derivative)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Malloc_allocator;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_div = Div<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = Shared_ptr<D_var, Allocator>::make_shared(0, 1.f);

    Shared_ptr<D_div, Allocator> d = Shared_ptr<D_div, Allocator>::make_shared(v1, v2);

    EXPECT_EQ(v1->compute() / v2->compute(), d->compute());
    EXPECT_EQ((v1->backward(0)->compute() * v2->compute() - v1->compute() * v2->backward(0)->compute()) / (v2->compute() * v2->compute()), d->backward(0)->compute());
}

TEST(Algorithms_test, can_perform_backward_derivation)
{
    using namespace math::algorithms::derivatives::backward;
    using namespace math::core::allocators;
    using namespace math::core::pointers;

    using Allocator = Shared_allocator<Malloc_allocator, 0>;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_add = Add<float, Allocator>;
    using D_mul = Mul<float, Allocator>;
    using D_const = Const<float, Allocator>;
    
    // Z = X^2 + 3xy + 1
    Shared_ptr<D_node, Allocator> x = Shared_ptr<D_var, Allocator>::make_shared(0, 3.0f);
    Shared_ptr<D_node, Allocator> y = Shared_ptr<D_var, Allocator>::make_shared(1, 2.0f);
    Shared_ptr<D_node, Allocator> z = Shared_ptr<D_add, Allocator>::make_shared(
        Shared_ptr<D_add, Allocator>::make_shared(
            Shared_ptr<D_mul, Allocator>::make_shared(x, x),
            Shared_ptr<D_mul, Allocator>::make_shared(
                Shared_ptr<D_const, Allocator>::make_shared(3.0),
                Shared_ptr<D_mul, Allocator>::make_shared(x, y))),
        Shared_ptr<D_const, Allocator>::make_shared(1.0));

    EXPECT_EQ(28.f, z->compute());
    EXPECT_EQ(12.f, z->backward(0)->compute());
    EXPECT_EQ(2.f, z->backward(0)->backward(0)->compute());
    EXPECT_EQ(9.f, z->backward(1)->compute());
}

