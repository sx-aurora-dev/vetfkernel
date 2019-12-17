#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <vml.h>
#include <vml/types.h>

#include "test.h"
using namespace test;

namespace test {
std::vector<Test> test_vec_;

void register_test(std::string const& name, 
                   bool (*func)(TestParam const& param)) {
      test_vec_.push_back({name, func});
}
} // namespace test

//
// UnaryOp
//

template<typename T>
bool test_UnaryOp(TestParam const& param,
                  Tensor<T>& out,
                  Tensor<T> const& in,
                  Tensor<T> const& exp,
                  int (*op)(vml::Tensor const&out, vml::Tensor const& in))
{
    int ret = op(out.tensor(), in.tensor());

    bool flag = false;
    if (ret == 0)
        flag = checkTensor(out, exp);

    if (param.verbose > 1 || (!flag && param.verbose > 0)) {
        fprintf(stderr, "in = \n");
        printTensor(in);
        fprintf(stderr, "out = \n");
        printTensor(out);
        fprintf(stderr, "expected = \n");
        printTensor(exp);
    }

    return flag;
}

template<typename T>
bool test_UnaryOp_01(TestParam const& param, 
                     int (*op)(vml::Tensor const&out, vml::Tensor const& in),
                     T (*func)(T))
{
    Tensor<T> out({10});
    Tensor<T> in({10});
    Tensor<T> exp({10});

    for (size_t i = 0; i < 10; ++i) {
      for (size_t j = 0; j < 10; ++j) {
          T v = T(i);
          in.data()[i] = v;
          out.data()[i] = 0;
          exp.data()[i] = func(v);
        }
    }

    return test_UnaryOp(param, out, in, exp, op);
}

#define DEFINE_TEST_UNARY_OP_01(name, op, func) \
bool test_##name##_01(TestParam const& param) { \
  return test_UnaryOp_01<float>(param, op, func); \
} \
REGISTER_TEST(#name"_01", test_##name##_01)

DEFINE_TEST_UNARY_OP_01(Abs, vml::abs, std::abs);
DEFINE_TEST_UNARY_OP_01(Sign, vml::sign, [](float x) { return (float)((x > 0.0f) - (x < 0.0f)); });
DEFINE_TEST_UNARY_OP_01(Exp, vml::exp, expf);
DEFINE_TEST_UNARY_OP_01(Expm1, vml::expm1, [](float x) { return expf(x) - 1.0f; });
DEFINE_TEST_UNARY_OP_01(Floor, vml::floor, std::floor);
DEFINE_TEST_UNARY_OP_01(Neg, vml::neg, [](float x) { return -x; });
DEFINE_TEST_UNARY_OP_01(Log, vml::log, std::log);
DEFINE_TEST_UNARY_OP_01(Log1p, vml::log1p, std::log1p);
DEFINE_TEST_UNARY_OP_01(Reciprocal, vml::reciprocal, [](float x) { return 1/x; });
DEFINE_TEST_UNARY_OP_01(Rsqrt, vml::rsqrt, [](float x) { return 1/std::sqrt(x); });
DEFINE_TEST_UNARY_OP_01(Sigmoid, vml::sigmoid,
                        [](float x) { return 1/(1+std::exp(-x)); });
DEFINE_TEST_UNARY_OP_01(Sqrt, vml::sqrt, std::sqrt);
DEFINE_TEST_UNARY_OP_01(Square, vml::square, [](float x) { return x * x; });
DEFINE_TEST_UNARY_OP_01(Sin, vml::sin, std::sin);
DEFINE_TEST_UNARY_OP_01(Cos, vml::cos, std::cos);
DEFINE_TEST_UNARY_OP_01(Tan, vml::tan, std::tan);
DEFINE_TEST_UNARY_OP_01(Sinh, vml::sinh, std::sinh);
DEFINE_TEST_UNARY_OP_01(Cosh, vml::cosh, std::cosh);
DEFINE_TEST_UNARY_OP_01(Tanh, vml::tanh, std::tanh);
DEFINE_TEST_UNARY_OP_01(Asinh, vml::asinh, std::asinh);
DEFINE_TEST_UNARY_OP_01(Acosh, vml::acosh, std::acosh);
DEFINE_TEST_UNARY_OP_01(Atanh, vml::atanh, std::atanh);

template <typename T>
bool test_AvgPool(TestParam const& param,
                  vml::Tensor& out,
                  vml::Tensor const& in,
                  vml::PoolingParam const& p,
                  T const* expected,
                  size_t nelems)
{
  if (vml::avgpool(out, in, p) != 0)
    return false;
  float* pout = reinterpret_cast<float*>(out.addr);

  bool flag = true;
  for (int i = 0; i < nelems; ++i) {
    bool tmp = pout[i] == expected[i];
    flag &= tmp;
    if (!tmp && param.verbose > 0) {
      fprintf(stderr, "pout[%d]=%f (expected is %f)\n", i, pout[i], expected[i]);
    }
  }

  return flag;
}

bool test_AvgPool_01(TestParam const& param)
{
  vml::Tensor* out = makeTensor<float>(4, {1, 1, 2, 2});
  vml::Tensor* in = makeTensor<float>(4, {1, 1, 2, 2});
#if 0
  vml::PoolingParam p = {
    .ksize = {1, 1, 1, 2},
    .stride = {1, 1, 1, 1},
    .data_format = FORMAT_NCHW,
    .padding = SAME,
    .pad_rows = 0,
    .pad_cols = 0,
  };
#else
  vml::PoolingParam p;
  p.ksize[0] = 1;
  p.ksize[1] = 1;
  p.ksize[2] = 1;
  p.ksize[3] = 2;
  p.stride[0] = 1;
  p.stride[1] = 1;
  p.stride[2] = 1;
  p.stride[3] = 1;
  p.data_format = FORMAT_NCHW;
  p.padding = SAME;
  p.pad_rows = 0;
  p.pad_cols = 0;
#endif

  float expected[4] = {1.5, 2.0, 3.5, 4.0};

  float* pin = reinterpret_cast<float*>(in->addr);
  for (int i = 0; i < 4; ++i)
    pin[i] = i + 1.0;

  return test_AvgPool(param, *out, *in, p, expected, 9);
}

bool test_AvgPool_02(TestParam const& param)
{
  vml::Tensor* out = makeTensor<float>(4, {1, 1, 3, 3});
  vml::Tensor* in = makeTensor<float>(4, {1, 1, 3, 3});
#if 0
  vml::PoolingParam p = {
    .ksize = {1, 1, 3, 3},
    .stride = {1, 1, 1, 1},
    .data_format = FORMAT_NCHW,
    .padding = SAME,
    .pad_rows = 0,
    .pad_cols = 0,
  };
#else
  vml::PoolingParam p;
  p.ksize[0] = 1;
  p.ksize[1] = 1;
  p.ksize[2] = 3;
  p.ksize[3] = 3;
  p.stride[0] = 1;
  p.stride[1] = 1;
  p.stride[2] = 1;
  p.stride[3] = 1;
  p.data_format = FORMAT_NCHW;
  p.padding = SAME;
  p.pad_rows = 1;
  p.pad_cols = 1;
#endif

  float expected[9] = {3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0};

  float* pin = reinterpret_cast<float*>(in->addr);
  for (int i = 0; i < 9; ++i)
    pin[i] = i + 1.0;

  return test_AvgPool(param, *out, *in, p, expected, 9);
}

REGISTER_TEST("AvgPool_01", test_AvgPool_01);
REGISTER_TEST("AvgPool_02", test_AvgPool_02);


int main(int argc, char* argv[])
{
    TestParam param;
    param.verbose = 0;

    char const* name = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) {
            ++param.verbose;
        } else if (strcmp(argv[i], "-t") == 0) {
          name = argv[++i];
        } else if (argv[i][0] == '-') {
          fprintf(stderr, "unknown option: %s\n", argv[i]);
          return 1;
        }
    }

    int ok = 0;
    int run = 0;
    for (Test& test : test_vec_) {
      if (name == nullptr || test.name == name) {
        ++run;
        bool flag = test.func(param);
        if (!flag || param.verbose > 0)
          fprintf(stderr, "%-30s %s\n", test.name.c_str(), flag ? "OK" : "NG");
        if (flag)
            ++ok;
      }
    }
    fprintf(stderr, "%d / %d tests failed. %s\n", run - ok, run, run == ok ? "OK" : "NG");
    return 0;
}
