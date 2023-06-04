# oc-deriv

A C++ implementation of backward symbolic high order derivation

Usage example:

```cpp
// Representaion and derivation of e.g. the
// function z = f(x,y) = sin(x^2 + 3xy + 1)

using namespace oc::deriv;

// Using Node types:
auto x = std::make_shared<Var<float>>(0, 3.f);
auto y = std::make_shared<Var<float>>(1, 2.f);
auto z = std::make_shared<Sin<float>>(
    std::make_shared<Add<float>>(
        std::make_shared<Add<float>>(
            std::make_shared<Add<float>>(x, x),
            std::make_shared<Mul<float>>(
                std::make_shared<Const<float>>(3.f),
                std::make_shared<Mul<float>>(x, y))),
        std::make_shared<Const<float>>(1.f)));

// Using functions:
auto x = variable(0, 3.f);
auto y = variable(1, 2.f);
auto z = sin(
    add(
        add(
            pow(x, 2.f),
            multiply(
                constant(3.f),
                multiply(x, y))),
        constant(1.f)));

// Using operator overloads:
auto z = sin((x ^ 2.f) + (3.f * x * y) + 1.f);
auto x = variable(0, 3.f);
auto y = variable(1, 2.f);

// Function value computation:
auto c = z->compute();

// First order derivative by x:
auto zx = z->backward(0);
// Second order derivative by x:
auto zx2 = z->backward(0)->backawrd(0);

// First order derivative by y:
auto zy = z->backward(1);
```
