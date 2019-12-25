#include "test.h"

using namespace test;

namespace {

template <typename T>
std::string join(std::vector<T> const& v, std::string const& delm)
{
  std::string s = "";
  for (size_t i = 0; i < v.size(); ++i) {
    s += std::to_string(v[i]);
    if (i +1 < v.size())
      s += delm;
  }
  return s;
}

template<typename T>
bool test_mean(TestParam const& param,
               Tensor<T>& out, Tensor<T> const& in, std::vector<int> const& axis,
               Tensor<T> const& exp)
{
  if (vml::mean(out, in, axis) != 0)
    return false;

  int flag = checkTensor(out, exp);
  if (param.verbose > 1 || (!flag && param.verbose > 0)) {
    fprintf(stderr, "in = \n");
    printTensor(in);
    fprintf(stderr, "axis=[%s]\n", join(axis, ", ").c_str());
    fprintf(stderr, "out = \n");
    printTensor(out);
    fprintf(stderr, "expected = \n");
    printTensor(exp);
  }

  return flag;

}

bool test_mean_d2a0(TestParam const& param)
{
  Tensor<float> in({2, 5});
  Tensor<float> out({5});
  Tensor<float> exp({5});
  auto axis = {0};

  in.copy({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  exp.copy({2.5, 3.5, 4.5, 5.5, 6.5});

  return test_mean(param, out, in, axis, exp);
}

bool test_mean_d2a1(TestParam const& param)
{
  Tensor<float> in({2, 5});
  Tensor<float> out({2});
  Tensor<float> exp({2});
  auto axis = {1};

  in.copy({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  exp.copy({2, 7});

  return test_mean(param, out, in, axis, exp);
}

bool test_mean_d3a02(TestParam const& param)
{
  Tensor<float> in({2, 2, 2});
  Tensor<float> out({2});
  Tensor<float> exp({2});
  auto axis = {0, 2};

  in.copy({0, 1, 2, 3, 4, 5, 6, 7});
  exp.copy({2.5, 4.5});

  return test_mean(param, out, in, axis, exp);
}

bool test_mean_d3a1(TestParam const& param)
{
  Tensor<float> in({2, 2, 2});
  Tensor<float> out({2, 2});
  Tensor<float> exp({2, 2});
  auto axis = {1};

  in.copy({0, 1, 2, 3, 4, 5, 6, 7});
  exp.copy({1, 2, 5, 6});

  return test_mean(param, out, in, axis, exp);
}

} // namespace

REGISTER_TEST("mean_d2a0", test_mean_d2a0);
REGISTER_TEST("mean_d2a1", test_mean_d2a1);
REGISTER_TEST("mean_d3a02", test_mean_d3a02);
REGISTER_TEST("mean_d3a1", test_mean_d3a1);
