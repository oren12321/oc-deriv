#include <gtest/gtest.h>

#include <memory>
#include <cmath>

#include <computoc/algorithms.h>
#include <memoc/allocators.h>
#include <memoc/pointers.h>
//#include <computoc/complex.h>

TEST(Algorithms_test, two_numbers_can_be_compared_with_specified_percision)
{
    using namespace computoc::algorithms;

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
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_const = Const<float, Allocator>;

    Shared_ptr<D_const, Allocator> c = make_shared<D_const, Allocator>(1.f);

    EXPECT_EQ(1.f, c->compute());
    EXPECT_EQ(0.f, c->backward(0)->compute());
}

TEST(Algorithm_test, variable_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    EXPECT_EQ(1.f, v->compute());
    EXPECT_EQ(1.f, v->backward(0)->compute());

    v->set(2.f);

    EXPECT_EQ(2.f, v->compute());
    EXPECT_EQ(1.f, v->backward(0)->compute());
}

TEST(Algorithm_test, addition_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_add = Add<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = make_shared<D_var, Allocator>(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_add, Allocator> a = make_shared<D_add, Allocator>(v1, v2);

    EXPECT_EQ(v1->compute() + v2->compute(), a->compute());
    EXPECT_EQ(v1->backward(0)->compute() + v2->backward(0)->compute(), a->backward(0)->compute());
}

TEST(Algorithm_test, subtraction_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_sub = Sub<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = make_shared<D_var, Allocator>(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_sub, Allocator> s = make_shared<D_sub, Allocator>(v1, v2);

    EXPECT_EQ(v1->compute() - v2->compute(), s->compute());
    EXPECT_EQ(v1->backward(0)->compute() - v2->backward(0)->compute(), s->backward(0)->compute());
}

TEST(Algorithm_test, negation_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_neg = Neg<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_neg, Allocator> n = make_shared<D_neg, Allocator>(v);

    EXPECT_EQ(-1.f, n->compute());
    EXPECT_EQ(-1.f, n->backward(0)->compute());
}

TEST(Algorithm_test, multiplication_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_mul = Mul<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = make_shared<D_var, Allocator>(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_mul, Allocator> m = make_shared<D_mul, Allocator>(v1, v2);

    EXPECT_EQ(v1->compute() * v2->compute(), m->compute());
    EXPECT_EQ(v1->backward(0)->compute() * v2->compute() + v1->compute() * v2->backward(0)->compute(), m->backward(0)->compute());
}

TEST(Algorithm_test, division_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_div = Div<float, Allocator>;

    Shared_ptr<D_var, Allocator> v1 = make_shared<D_var, Allocator>(0, 1.f);
    Shared_ptr<D_var, Allocator> v2 = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_div, Allocator> d = make_shared<D_div, Allocator>(v1, v2);

    EXPECT_EQ(v1->compute() / v2->compute(), d->compute());
    EXPECT_EQ((v1->backward(0)->compute() * v2->compute() - v1->compute() * v2->backward(0)->compute()) / (v2->compute() * v2->compute()), d->backward(0)->compute());
}

TEST(Algorithm_test, sin_and_cos_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_sin = Sin<float, Allocator>;
    using D_cos = Cos<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_sin, Allocator> s = make_shared<D_sin, Allocator>(v);
    EXPECT_EQ(std::sin(1.f), s->compute());
    EXPECT_EQ(1.f * std::cos(1.f), s->backward(0)->compute());

    Shared_ptr<D_cos, Allocator> c = make_shared<D_cos, Allocator>(v);
    EXPECT_EQ(std::cos(1.f), c->compute());
    EXPECT_EQ(1.f * (-std::sin(1.f)), c->backward(0)->compute());

    //using namespace computoc::types;

    //using C_var = Var<Complex<float>, Allocator>;
    //using C_sin = Sin<Complex<float>, Allocator>;

    //Shared_ptr<C_var, Allocator> cv = make_shared<C_var, Allocator>(0, Complex{ 1.f, 2.f });

    //Shared_ptr<C_sin, Allocator> cs = make_shared<C_sin, Allocator>(cv);
    //EXPECT_EQ(sin(Complex{ 1.f, 2.f }), cs->compute());
    //EXPECT_EQ(1.f *cos(Complex{ 1.f, 2.f }), cs->backward(0)->compute());
}

TEST(Algorithm_test, tan_and_sec_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_tan = Tan<float, Allocator>;
    using D_sec = Sec<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_tan, Allocator> t = make_shared<D_tan, Allocator>(v);
    EXPECT_EQ(std::tan(1.f), t->compute());
    EXPECT_EQ(1.f * (1.f / (std::cos(1.f) * std::cos(1.f))), t->backward(0)->compute());

    Shared_ptr<D_sec, Allocator> s = make_shared<D_sec, Allocator>(v);
    EXPECT_EQ(1.f / std::cos(1.f), s->compute());
    EXPECT_EQ(1.f * (std::tan(1.f) / std::cos(1.f)), s->backward(0)->compute());
}

TEST(Algorithm_test, cot_and_csc_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_cot = Cot<float, Allocator>;
    using D_csc = Csc<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_cot, Allocator> ct = make_shared<D_cot, Allocator>(v);
    EXPECT_EQ(1.f / std::tan(1.f), ct->compute());
    EXPECT_EQ(1.f * (-1.f / (std::sin(1.f) * std::sin(1.f))), ct->backward(0)->compute());

    Shared_ptr<D_csc, Allocator> cs = make_shared<D_csc, Allocator>(v);
    EXPECT_EQ(1.f / std::sin(1.f), cs->compute());
    EXPECT_EQ(1.f * (-1.f / (std::tan(1.f) * std::sin(1.f))), cs->backward(0)->compute());
}

TEST(Algorithm_test, exp_and_ln_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_exp = Exp<float, Allocator>;
    using D_ln = Ln<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_exp, Allocator> e = make_shared<D_exp, Allocator>(v);
    EXPECT_EQ(std::exp(1.f), e->compute());
    EXPECT_EQ(1.f * std::exp(1.f), e->backward(0)->compute());

    Shared_ptr<D_ln, Allocator> l = make_shared<D_ln, Allocator>(v);
    EXPECT_EQ(std::log(1.f), l->compute());
    EXPECT_EQ(1.f / 1.f, l->backward(0)->compute());
}

TEST(Algorithm_test, pow_f_by_n_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_pow = Pow_fn<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_pow, Allocator> p = make_shared<D_pow, Allocator>(v, 2.f);
    EXPECT_EQ(std::pow(1.f, 2.f), p->compute());
    EXPECT_EQ(2.f * 1.f, p->backward(0)->compute());
}

TEST(Algorithm_test, pow_a_by_f_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_pow = Pow_af<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_pow, Allocator> p = make_shared<D_pow, Allocator>(2.f, v);
    EXPECT_EQ(std::pow(2.f, 1.f), p->compute());
    EXPECT_EQ(1.f * std::pow(2.f, 1.f) * std::log(2.f), p->backward(0)->compute());
}

TEST(Algorithm_test, pow_f_by_g_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_const = Const<float, Allocator>;
    using D_pow = Pow_fg<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);
    Shared_ptr<D_const, Allocator> c = make_shared<D_const, Allocator>(2.f);

    Shared_ptr<D_pow, Allocator> p = make_shared<D_pow, Allocator>(v, c);
    EXPECT_EQ(std::pow(1.f, 2.f), p->compute());
    EXPECT_EQ(2.f * 1.f, p->backward(0)->compute());
}

TEST(Algorithm_test, asin_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_asin = Asin<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_asin, Allocator> a = make_shared<D_asin, Allocator>(v);
    EXPECT_EQ(std::asin(1.f), a->compute());
    EXPECT_EQ(1.f * std::pow(1.f - std::pow(1.f, 2.f), -.5f), a->backward(0)->compute());
}

TEST(Algorithm_test, acos_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_acos = Acos<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_acos, Allocator> a = make_shared<D_acos, Allocator>(v);
    EXPECT_EQ(std::acos(1.f), a->compute());
    EXPECT_EQ(1.f * -std::pow(1.f - std::pow(1.f, 2.f), -.5f), a->backward(0)->compute());
}

TEST(Algorithm_test, atan_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_atan = Atan<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_atan, Allocator> a = make_shared<D_atan, Allocator>(v);
    EXPECT_EQ(std::atan(1.f), a->compute());
    EXPECT_EQ(1.f * std::pow(1.f + std::pow(1.f, 2.f), -1.f), a->backward(0)->compute());
}

TEST(Algorithm_test, acot_backward_derivative)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Malloc_allocator;
    using D_var = Var<float, Allocator>;
    using D_acot = Acot<float, Allocator>;

    Shared_ptr<D_var, Allocator> v = make_shared<D_var, Allocator>(0, 1.f);

    Shared_ptr<D_acot, Allocator> a = make_shared<D_acot, Allocator>(v);
    EXPECT_EQ(std::atan(1.f / 1.f), a->compute());
    EXPECT_EQ(1.f * -std::pow(1.f + std::pow(1.f, 2.f), -1.f), a->backward(0)->compute());
}

TEST(Algorithms_test, can_perform_backward_derivation)
{
    using namespace computoc::algorithms::derivatives;
    using namespace memoc;

    using Allocator = Shared_allocator<Malloc_allocator, 0>;
    using D_node = Node<float, Allocator>;
    using D_var = Var<float, Allocator>;
    using D_add = Add<float, Allocator>;
    using D_mul = Mul<float, Allocator>;
    using D_const = Const<float, Allocator>;
    using D_sin = Sin<float, Allocator>;

    // z = sin(x^2 + 3xy + 1)
    Shared_ptr<D_node, Allocator> x = make_shared<D_var, Allocator>(0, 3.0f);
    Shared_ptr<D_node, Allocator> y = make_shared<D_var, Allocator>(1, 2.0f);
    Shared_ptr<D_node, Allocator> z = make_shared<D_sin, Allocator>(
        make_shared<D_add, Allocator>(
            make_shared<D_add, Allocator>(
                make_shared<D_mul, Allocator>(x, x),
                make_shared<D_mul, Allocator>(
                    make_shared<D_const, Allocator>(3.0),
                    make_shared<D_mul, Allocator>(x, y))),
            make_shared<D_const, Allocator>(1.0)));

    EXPECT_EQ(std::sin(28.f), z->compute());
    EXPECT_EQ(12.f * std::cos(28.f), z->backward(0)->compute());
    EXPECT_EQ(2.f * std::cos(28.f) - 12.f * 12.f * std::sin(28.f), z->backward(0)->backward(0)->compute());
    EXPECT_EQ(9.f * std::cos(28.f), z->backward(1)->compute());
}

