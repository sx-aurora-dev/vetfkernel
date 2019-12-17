#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include "test.h"
using namespace test;

bool test_LeakyRelu(TestParam const& param)
{
  Tensor<float> activations({2, 5});
  Tensor<float> features({2, 5});
  Tensor<float> expected({2, 5});

  float a[2][5] = {{-0.09, 0.7, -0.05, 0.3, -0.01},
                   {0.1, -0.03, 0.5, -0.07, 0.9}};
  float f[2][5] = {{-0.9, 0.7, -0.5, 0.3, -0.1}, 
                   {0.1, -0.3, 0.5, -0.7, 0.9}};
  float alpha = 0.1;

  memcpy(features.data(), f, sizeof(float) * 2 * 5);
  memcpy(expected.data(), a, sizeof(float) * 2 * 5);

  vml::leaky_relu(activations, features, alpha);

  bool flag = checkTensor(activations, expected);

  if (param.verbose > 1 || (!flag && param.verbose > 0)) {
    fprintf(stderr, "alpha = %f\n", alpha);
    fprintf(stderr, "features = \n");
    printTensor(features);
    fprintf(stderr, "activations = \n");
    printTensor(activations);
    fprintf(stderr, "expected = \n");
    printTensor(expected);
  }

  return flag;
}

// gradients = dy / dx = (y1 - y0) / dx;
// where
//   dx = (x + delta) - (x - delta) = 2 * delta
//   y0 = func(x - delta)
//   y1 = func(x + delta)

template <typename T, typename F>
void compute_gradients(Tensor<T> const& gradients,
                       Tensor<T> const& x,
                       F func)
{
  T delta = 1e-2;
  Tensor<T> x0(x.shape());
  Tensor<T> x1(x.shape());
  Tensor<T> y0(x.shape());
  Tensor<T> y1(x.shape());
  Tensor<T> dx({1});
  Tensor<T> dy(x.shape());

  for (int i = 0; i < x.nelems(); ++i) {
    x0.data()[i] = x.data()[i] - delta;
    x1.data()[i] = x.data()[i] + delta;
  }
  dx.data()[0] = T(2 * delta);

  func(y0, x0);
  func(y1, x1);
  vml::sub(dy, y1, y0);
  vml::div(gradients, dy, dx);
}

template<typename T, typename F>
void compute_gradients_by_backprop(Tensor<T> const& gradients,
                                   Tensor<T> const& x,
                                   F backprop)
{
  Tensor<float> one(x.shape());
  for (int i = 0; i < x.nelems(); ++i)
    one.data()[i] = T(1.0);

  backprop(gradients, one, x);
}

bool test_LeakyReluGrad(TestParam const& param)
{
  float alpha = 0.1;

  Tensor<float> features({2, 5});
  float f[10] = {-0.9, -0.7, -0.5, -0.3, -0.1, 
                 0.1, 0.3, 0.5, 0.7, 0.9};
  for (int i = 0; i < 10; ++i)
    features.data()[i] = f[i];

  Tensor<float> backprops({2, 5});
  compute_gradients_by_backprop(backprops, features,
                                [alpha](vml::Tensor const& g,
                                        vml::Tensor const& y,
                                        vml::Tensor const& x) {
                                  vml::leaky_relu_grad(g, y, x, alpha);
                                });

  Tensor<float> expected({2, 5});
  compute_gradients(expected, features,
                    [alpha](vml::Tensor const& y, vml::Tensor const& x) {
                      vml::leaky_relu(y, x, alpha);
                    });

  bool flag = checkTensor(backprops, expected);

  if (param.verbose > 1 || (!flag && param.verbose > 0)) {
    fprintf(stderr, "backprops:\n");
    printTensor(backprops);
    fprintf(stderr, "expected:\n");
    printTensor(expected);
  }

  return flag;
}

REGISTER_TEST("LeakyRelu", test_LeakyRelu);
REGISTER_TEST("LeakyReluGrad", test_LeakyReluGrad);


