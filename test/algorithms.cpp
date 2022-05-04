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