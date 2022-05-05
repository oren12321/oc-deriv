#include <gtest/gtest.h>

#include <math/core/algorithms.h>

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

TEST(test, test)
{
    using namespace math::algorithms::derivatives;

    Var x(3.0, 0);
    Var y(2.0, 1);
    auto z = Add(
        Add(
            Mul(x, x),
            Mul(Const(3.0), Mul(x, y))),
        Const(1.0));

    double v = z.compute();
    double d = z.backward(1);
}

