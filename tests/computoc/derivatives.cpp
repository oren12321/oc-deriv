#include <gtest/gtest.h>

#include <memory>
#include <cmath>

#include <computoc/derivatives.h>

TEST(Derivation, constant_backward_derivative)
{
    using namespace computoc;


    {

        using D_const = Const<float>;

        std::shared_ptr<D_const> c = std::make_shared<D_const>(1.f);

        EXPECT_EQ(1.f, c->compute());
        EXPECT_EQ(0.f, c->backward(0)->compute());
    }

    {
        auto c = constant(1.f);

        EXPECT_EQ(1.f, c->compute());
        EXPECT_EQ(0.f, c->backward(0)->compute());
    }
}

TEST(Derivation, variable_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        EXPECT_EQ(1.f, v->compute());
        EXPECT_EQ(1.f, v->backward(0)->compute());

        v->set(2.f);

        EXPECT_EQ(2.f, v->compute());
        EXPECT_EQ(1.f, v->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        EXPECT_EQ(1.f, v->compute());
        EXPECT_EQ(1.f, v->backward(0)->compute());

        v->set(2.f);

        EXPECT_EQ(2.f, v->compute());
        EXPECT_EQ(1.f, v->backward(0)->compute());
    }
}

TEST(Derivation, addition_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_add = Add<float>;

        std::shared_ptr<D_var> v1 = std::make_shared<D_var>(0, 1.f);
        std::shared_ptr<D_var> v2 = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_add> a = std::make_shared<D_add>(v1, v2);

        EXPECT_EQ(v1->compute() + v2->compute(), a->compute());
        EXPECT_EQ(v1->backward(0)->compute() + v2->backward(0)->compute(), a->backward(0)->compute());
    }

    {
        auto v1 = variable(0, 1.f);
        auto v2 = variable(0, 1.f);

        auto a1 = add(v1, v2);

        EXPECT_EQ(v1->compute() + v2->compute(), a1->compute());
        EXPECT_EQ(v1->backward(0)->compute() + v2->backward(0)->compute(), a1->backward(0)->compute());

        auto a2 = v1 + v2;

        EXPECT_EQ(v1->compute() + v2->compute(), a2->compute());
        EXPECT_EQ(v1->backward(0)->compute() + v2->backward(0)->compute(), a2->backward(0)->compute());

        auto a3 = v1 + 1.f;

        EXPECT_EQ(v1->compute() + 1.f, a3->compute());
        EXPECT_EQ(v1->backward(0)->compute() + 0.f, a3->backward(0)->compute());

        auto a4 = 1.f + v2;

        EXPECT_EQ(1.f + v2->compute(), a4->compute());
        EXPECT_EQ(0.f + v2->backward(0)->compute(), a4->backward(0)->compute());
    }
}

TEST(Derivation, subtraction_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_sub = Sub<float>;

        std::shared_ptr<D_var> v1 = std::make_shared<D_var>(0, 1.f);
        std::shared_ptr<D_var> v2 = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_sub> s = std::make_shared<D_sub>(v1, v2);

        EXPECT_EQ(v1->compute() - v2->compute(), s->compute());
        EXPECT_EQ(v1->backward(0)->compute() - v2->backward(0)->compute(), s->backward(0)->compute());
    }

    {
        auto v1 = variable(0, 1.f);
        auto v2 = variable(0, 1.f);

        auto a1 = subtract(v1, v2);

        EXPECT_EQ(v1->compute() - v2->compute(), a1->compute());
        EXPECT_EQ(v1->backward(0)->compute() - v2->backward(0)->compute(), a1->backward(0)->compute());

        auto a2 = v1 - v2;

        EXPECT_EQ(v1->compute() - v2->compute(), a2->compute());
        EXPECT_EQ(v1->backward(0)->compute() - v2->backward(0)->compute(), a2->backward(0)->compute());

        auto a3 = v1 - 1.f;

        EXPECT_EQ(v1->compute() - 1.f, a3->compute());
        EXPECT_EQ(v1->backward(0)->compute() - 0.f, a3->backward(0)->compute());

        auto a4 = 1.f - v2;

        EXPECT_EQ(1.f - v2->compute(), a4->compute());
        EXPECT_EQ(0.f - v2->backward(0)->compute(), a4->backward(0)->compute());
    }
}

TEST(Derivation, negation_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_neg = Neg<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_neg> n = std::make_shared<D_neg>(v);

        EXPECT_EQ(-1.f, n->compute());
        EXPECT_EQ(-1.f, n->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto n1 = negate(v);

        EXPECT_EQ(-1.f, n1->compute());
        EXPECT_EQ(-1.f, n1->backward(0)->compute());

        auto n2 = -v;

        EXPECT_EQ(-1.f, n2->compute());
        EXPECT_EQ(-1.f, n2->backward(0)->compute());
    }
}

TEST(Derivation, multiplication_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_mul = Mul<float>;

        std::shared_ptr<D_var> v1 = std::make_shared<D_var>(0, 1.f);
        std::shared_ptr<D_var> v2 = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_mul> m = std::make_shared<D_mul>(v1, v2);

        EXPECT_EQ(v1->compute() * v2->compute(), m->compute());
        EXPECT_EQ(v1->backward(0)->compute() * v2->compute() + v1->compute() * v2->backward(0)->compute(), m->backward(0)->compute());
    }

    {
        auto v1 = variable(0, 1.f);
        auto v2 = variable(0, 1.f);

        auto m1 = multiply(v1, v2);

        EXPECT_EQ(v1->compute() * v2->compute(), m1->compute());
        EXPECT_EQ(v1->backward(0)->compute() * v2->compute() + v1->compute() * v2->backward(0)->compute(), m1->backward(0)->compute());

        auto m2 = v1 * v2;

        EXPECT_EQ(v1->compute() * v2->compute(), m2->compute());
        EXPECT_EQ(v1->backward(0)->compute() * v2->compute() + v1->compute() * v2->backward(0)->compute(), m2->backward(0)->compute());

        auto m3 = v1 * 1.f;

        EXPECT_EQ(v1->compute() * 1.f, m3->compute());
        EXPECT_EQ(v1->backward(0)->compute() * 1.f + v1->compute() * 0.f, m3->backward(0)->compute());

        auto m4 = 1.f * v2;

        EXPECT_EQ(1.f * v2->compute(), m4->compute());
        EXPECT_EQ(0.f * v2->compute() + 1.f * v2->backward(0)->compute(), m4->backward(0)->compute());
    }
}

TEST(Derivation, division_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_div = Div<float>;

        std::shared_ptr<D_var> v1 = std::make_shared<D_var>(0, 1.f);
        std::shared_ptr<D_var> v2 = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_div> d = std::make_shared<D_div>(v1, v2);

        EXPECT_EQ(v1->compute() / v2->compute(), d->compute());
        EXPECT_EQ((v1->backward(0)->compute() * v2->compute() - v1->compute() * v2->backward(0)->compute()) / (v2->compute() * v2->compute()), d->backward(0)->compute());
    }

    {
        auto v1 = variable(0, 1.f);
        auto v2 = variable(0, 1.f);

        auto d1 = divide(v1, v2);

        EXPECT_EQ(v1->compute() / v2->compute(), d1->compute());
        EXPECT_EQ((v1->backward(0)->compute() * v2->compute() - v1->compute() * v2->backward(0)->compute()) / (v2->compute() * v2->compute()), d1->backward(0)->compute());

        auto d2 = v1 / v2;

        EXPECT_EQ(v1->compute() / v2->compute(), d2->compute());
        EXPECT_EQ((v1->backward(0)->compute() * v2->compute() - v1->compute() * v2->backward(0)->compute()) / (v2->compute() * v2->compute()), d2->backward(0)->compute());

        auto d3 = v1 / 1.f;

        EXPECT_EQ(v1->compute() / 1.f, d3->compute());
        EXPECT_EQ((v1->backward(0)->compute() * 1.f - v1->compute() * 0.f) / (1.f * 1.f), d3->backward(0)->compute());

        auto d4 = 1.f / v2;

        EXPECT_EQ(1.f / v2->compute(), d4->compute());
        EXPECT_EQ((0.f * v2->compute() - 1.f * v2->backward(0)->compute()) / (v2->compute() * v2->compute()), d4->backward(0)->compute());
    }
}

TEST(Derivation, sin_and_cos_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_sin = Sin<float>;
        using D_cos = Cos<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_sin> s = std::make_shared<D_sin>(v);
        EXPECT_EQ(std::sin(1.f), s->compute());
        EXPECT_EQ(1.f * std::cos(1.f), s->backward(0)->compute());

        std::shared_ptr<D_cos> c = std::make_shared<D_cos>(v);
        EXPECT_EQ(std::cos(1.f), c->compute());
        EXPECT_EQ(1.f * (-std::sin(1.f)), c->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto s = sin(v);
        EXPECT_EQ(std::sin(1.f), s->compute());
        EXPECT_EQ(1.f * std::cos(1.f), s->backward(0)->compute());

        auto c = cos(v);
        EXPECT_EQ(std::cos(1.f), c->compute());
        EXPECT_EQ(1.f * (-std::sin(1.f)), c->backward(0)->compute());
    }
}

TEST(Derivation, tan_and_sec_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_tan = Tan<float>;
        using D_sec = Sec<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_tan> t = std::make_shared<D_tan>(v);
        EXPECT_EQ(std::tan(1.f), t->compute());
        EXPECT_EQ(1.f * (1.f / (std::cos(1.f) * std::cos(1.f))), t->backward(0)->compute());

        std::shared_ptr<D_sec> s = std::make_shared<D_sec>(v);
        EXPECT_EQ(1.f / std::cos(1.f), s->compute());
        EXPECT_EQ(1.f * (std::tan(1.f) / std::cos(1.f)), s->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto t = tan(v);
        EXPECT_EQ(std::tan(1.f), t->compute());
        EXPECT_EQ(1.f * (1.f / (std::cos(1.f) * std::cos(1.f))), t->backward(0)->compute());

        auto s = sec(v);
        EXPECT_EQ(1.f / std::cos(1.f), s->compute());
        EXPECT_EQ(1.f * (std::tan(1.f) / std::cos(1.f)), s->backward(0)->compute());
    }
}

TEST(Derivation, cot_and_csc_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_cot = Cot<float>;
        using D_csc = Csc<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_cot> ct = std::make_shared<D_cot>(v);
        EXPECT_EQ(1.f / std::tan(1.f), ct->compute());
        EXPECT_EQ(1.f * (-1.f / (std::sin(1.f) * std::sin(1.f))), ct->backward(0)->compute());

        std::shared_ptr<D_csc> cs = std::make_shared<D_csc>(v);
        EXPECT_EQ(1.f / std::sin(1.f), cs->compute());
        EXPECT_EQ(1.f * (-1.f / (std::tan(1.f) * std::sin(1.f))), cs->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto ct = cot(v);
        EXPECT_EQ(1.f / std::tan(1.f), ct->compute());
        EXPECT_EQ(1.f * (-1.f / (std::sin(1.f) * std::sin(1.f))), ct->backward(0)->compute());

        auto cs = csc(v);
        EXPECT_EQ(1.f / std::sin(1.f), cs->compute());
        EXPECT_EQ(1.f * (-1.f / (std::tan(1.f) * std::sin(1.f))), cs->backward(0)->compute());
    }
}

TEST(Derivation, exp_and_ln_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_exp = Exp<float>;
        using D_ln = Ln<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_exp> e = std::make_shared<D_exp>(v);
        EXPECT_EQ(std::exp(1.f), e->compute());
        EXPECT_EQ(1.f * std::exp(1.f), e->backward(0)->compute());

        std::shared_ptr<D_ln> l = std::make_shared<D_ln>(v);
        EXPECT_EQ(std::log(1.f), l->compute());
        EXPECT_EQ(1.f / 1.f, l->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto e = exp(v);
        EXPECT_EQ(std::exp(1.f), e->compute());
        EXPECT_EQ(1.f * std::exp(1.f), e->backward(0)->compute());

        auto l = ln(v);
        EXPECT_EQ(std::log(1.f), l->compute());
        EXPECT_EQ(1.f / 1.f, l->backward(0)->compute());
    }
}

TEST(Derivation, pow_f_by_n_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_pow = Pow_fn<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_pow> p = std::make_shared<D_pow>(v, 2.f);
        EXPECT_EQ(std::pow(1.f, 2.f), p->compute());
        EXPECT_EQ(2.f * 1.f, p->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto p1 = pow(v, 2.f);
        EXPECT_EQ(std::pow(1.f, 2.f), p1->compute());
        EXPECT_EQ(2.f * 1.f, p1->backward(0)->compute());

        auto p2 = v ^ 2.f;
        EXPECT_EQ(std::pow(1.f, 2.f), p2->compute());
        EXPECT_EQ(2.f * 1.f, p2->backward(0)->compute());
    }
}

TEST(Derivation, pow_a_by_f_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_pow = Pow_af<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_pow> p = std::make_shared<D_pow>(2.f, v);
        EXPECT_EQ(std::pow(2.f, 1.f), p->compute());
        EXPECT_EQ(1.f * std::pow(2.f, 1.f) * std::log(2.f), p->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto p1 = pow(2.f, v);
        EXPECT_EQ(std::pow(2.f, 1.f), p1->compute());
        EXPECT_EQ(1.f * std::pow(2.f, 1.f) * std::log(2.f), p1->backward(0)->compute());

        auto p2 = 2.f ^ v;
        EXPECT_EQ(std::pow(2.f, 1.f), p2->compute());
        EXPECT_EQ(1.f * std::pow(2.f, 1.f) * std::log(2.f), p2->backward(0)->compute());
    }
}

TEST(Derivation, pow_f_by_g_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_const = Const<float>;
        using D_pow = Pow_fg<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);
        std::shared_ptr<D_const> c = std::make_shared<D_const>(2.f);

        std::shared_ptr<D_pow> p = std::make_shared<D_pow>(v, c);
        EXPECT_EQ(std::pow(1.f, 2.f), p->compute());
        EXPECT_EQ(2.f * 1.f, p->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);
        auto c = constant(2.f);

        auto p1 = pow(v, c);
        EXPECT_EQ(std::pow(1.f, 2.f), p1->compute());
        EXPECT_EQ(2.f * 1.f, p1->backward(0)->compute());

        auto p2 = v ^ c;
        EXPECT_EQ(std::pow(1.f, 2.f), p2->compute());
        EXPECT_EQ(2.f * 1.f, p2->backward(0)->compute());
    }
}

TEST(Derivation, asin_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_asin = Asin<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_asin> a = std::make_shared<D_asin>(v);
        EXPECT_EQ(std::asin(1.f), a->compute());
        EXPECT_EQ(1.f * std::pow(1.f - std::pow(1.f, 2.f), -.5f), a->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto a = asin(v);
        EXPECT_EQ(std::asin(1.f), a->compute());
        EXPECT_EQ(1.f * std::pow(1.f - std::pow(1.f, 2.f), -.5f), a->backward(0)->compute());
    }
}

TEST(Derivation, acos_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_acos = Acos<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_acos> a = std::make_shared<D_acos>(v);
        EXPECT_EQ(std::acos(1.f), a->compute());
        EXPECT_EQ(1.f * -std::pow(1.f - std::pow(1.f, 2.f), -.5f), a->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto a = acos(v);
        EXPECT_EQ(std::acos(1.f), a->compute());
        EXPECT_EQ(1.f * -std::pow(1.f - std::pow(1.f, 2.f), -.5f), a->backward(0)->compute());
    }
}

TEST(Derivation, atan_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_atan = Atan<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_atan> a = std::make_shared<D_atan>(v);
        EXPECT_EQ(std::atan(1.f), a->compute());
        EXPECT_EQ(1.f * std::pow(1.f + std::pow(1.f, 2.f), -1.f), a->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto a = atan(v);
        EXPECT_EQ(std::atan(1.f), a->compute());
        EXPECT_EQ(1.f * std::pow(1.f + std::pow(1.f, 2.f), -1.f), a->backward(0)->compute());
    }
}

TEST(Derivation, acot_backward_derivative)
{
    using namespace computoc;


    {

        using D_var = Var<float>;
        using D_acot = Acot<float>;

        std::shared_ptr<D_var> v = std::make_shared<D_var>(0, 1.f);

        std::shared_ptr<D_acot> a = std::make_shared<D_acot>(v);
        EXPECT_EQ(std::atan(1.f / 1.f), a->compute());
        EXPECT_EQ(1.f * -std::pow(1.f + std::pow(1.f, 2.f), -1.f), a->backward(0)->compute());
    }

    {
        auto v = variable(0, 1.f);

        auto a = acot(v);
        EXPECT_EQ(std::atan(1.f / 1.f), a->compute());
        EXPECT_EQ(1.f * -std::pow(1.f + std::pow(1.f, 2.f), -1.f), a->backward(0)->compute());
    }
}

TEST(Algorithms_test, can_perform_backward_derivation)
{
    using namespace computoc;


    {
        using D_node = Node<float>;
        using D_var = Var<float>;
        using D_add = Add<float>;
        using D_mul = Mul<float>;
        using D_const = Const<float>;
        using D_sin = Sin<float>;

        // z = sin(x^2 + 3xy + 1)
        std::shared_ptr<D_node> x = std::make_shared<D_var>(0, 3.0f);
        std::shared_ptr<D_node> y = std::make_shared<D_var>(1, 2.0f);
        std::shared_ptr<D_node> z = std::make_shared<D_sin>(
            std::make_shared<D_add>(
                std::make_shared<D_add>(
                    std::make_shared<D_mul>(x, x),
                    std::make_shared<D_mul>(
                        std::make_shared<D_const>(3.0),
                        std::make_shared<D_mul>(x, y))),
                std::make_shared<D_const>(1.0)));

        EXPECT_EQ(std::sin(28.f), z->compute());
        EXPECT_EQ(12.f * std::cos(28.f), z->backward(0)->compute());
        EXPECT_EQ(2.f * std::cos(28.f) - 12.f * 12.f * std::sin(28.f), z->backward(0)->backward(0)->compute());
        EXPECT_EQ(9.f * std::cos(28.f), z->backward(1)->compute());
    }

    {


        // z = sin(x^2 + 3xy + 1)
        auto x = variable(0, 3.f);
        auto y = variable(1, 2.f);

        auto z1 = sin(
            add(
                add(
                    pow(x, 2.f),
                    multiply(
                        constant(3.f),
                        multiply(x, y))),
                constant(1.f)));

        EXPECT_EQ(std::sin(28.f), z1->compute());
        EXPECT_EQ(12.f * std::cos(28.f), z1->backward(0)->compute());
        EXPECT_EQ(2.f * std::cos(28.f) - 12.f * 12.f * std::sin(28.f), z1->backward(0)->backward(0)->compute());
        EXPECT_EQ(9.f * std::cos(28.f), z1->backward(1)->compute());

        auto z2 = sin((x ^ 2.f) + (3.f * x * y) + 1.f);

        EXPECT_EQ(std::sin(28.f), z2->compute());
        EXPECT_EQ(12.f * std::cos(28.f), z2->backward(0)->compute());
        EXPECT_EQ(2.f * std::cos(28.f) - 12.f * 12.f * std::sin(28.f), z2->backward(0)->backward(0)->compute());
        EXPECT_EQ(9.f * std::cos(28.f), z2->backward(1)->compute());
    }
}
