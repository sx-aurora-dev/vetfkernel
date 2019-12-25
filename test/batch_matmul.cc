#include "test.h"

using namespace test;

namespace {

template<typename T>
bool test_batch_matmul(
    TestParam const& param,
    Tensor<T> const& in_x,
    Tensor<T> const& in_y,
    Tensor<T>& out,
    const bool adj_x,
    const bool adj_y,
    Tensor<T> const& exp)
{
  if (vml::batch_matmul(in_x, in_y, out, adj_x, adj_y) != 0)
    return false;

  int flag = checkTensor(out, exp);
  if (param.verbose > 1 || (!flag && param.verbose > 0)) {
    fprintf(stderr, "in_x = \n");
    printTensor(in_x);
    fprintf(stderr, "in_y = \n");
    printTensor(in_y);
    fprintf(stderr, "out = \n");
    printTensor(out);
    fprintf(stderr, "expected = \n");
    printTensor(exp);
  }

  return flag;

}

bool test_batch_matmul_ff(TestParam const& param)
{
  Tensor<float> in_x({2, 3});
  Tensor<float> in_y({2, 3, 5});
  Tensor<float> out({2, 2, 5});
  Tensor<float> exp({2, 2, 5});
  auto axis = {1};

  in_x.copy({0, 1, 2,
             3, 4, 5});
  in_y.copy({ 0, 1, 2, 3, 4,
              5, 6, 7, 8, 9,
             10,11,12,13,14,
	     15,16,17,18,19,
             20,21,22,23,24,
	     25,26,27,28,29});

  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118,
             70, 73, 76, 79, 82,
            250,262,274,286,298}) ;

  return test_batch_matmul(param, in_x, in_y, out, false, false, exp);
}

bool test_batch_matmul_ft(TestParam const& param)
{
  Tensor<float> in_x({2, 3});
  Tensor<float> in_y({2, 5, 3});
  Tensor<float> out({2, 2, 5});
  Tensor<float> exp({2, 2, 5});
  auto axis = {1};

  in_x.copy({0, 1, 2, 3, 4, 5});
  in_y.copy({ 0, 5,10,
              1, 6,11,
              2, 7,12,
	      3, 8,13,
              4, 9,14,
	     15,20,25,
	     16,21,26,
	     17,22,27,
	     18,23,28,
             19,24,29});

  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118,
             70, 73, 76, 79, 82,
            250,262,274,286,298}) ;

  return test_batch_matmul(param, in_x, in_y, out, false, true, exp);
}

bool test_batch_matmul_tf(TestParam const& param)
{
  Tensor<float> in_x({3, 2});
  Tensor<float> in_y({2, 3, 5});
  Tensor<float> out({2, 2, 5});
  Tensor<float> exp({2, 2, 5});
  auto axis = {1};

  in_x.copy({0, 3, 1, 4, 2, 5});
  in_y.copy({ 0, 1, 2, 3, 4,
              5, 6, 7, 8, 9,
             10,11,12,13,14,
	     15,16,17,18,19,
             20,21,22,23,24,
	     25,26,27,28,29});

  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118,
             70, 73, 76, 79, 82,
            250,262,274,286,298}) ;

  return test_batch_matmul(param, in_x, in_y, out, true, false, exp);
}

bool test_batch_matmul_tt(TestParam const& param)
{
  Tensor<float> in_x({3, 2});
  Tensor<float> in_y({2, 5, 3});
  Tensor<float> out({2, 2, 5});
  Tensor<float> exp({2, 2, 5});
  auto axis = {1};

  in_x.copy({0, 3, 1, 4, 2, 5});
  in_y.copy({ 0, 5,10,
              1, 6,11,
              2, 7,12,
	      3, 8,13,
              4, 9,14,
	     15,20,25,
	     16,21,26,
	     17,22,27,
	     18,23,28,
             19,24,29});

  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118,
             70, 73, 76, 79, 82,
            250,262,274,286,298}) ;

  return test_batch_matmul(param, in_x, in_y, out, true, true, exp);
}

} // namespace

REGISTER_TEST("batch_matmul_ff", test_batch_matmul_ff);
REGISTER_TEST("batch_matmul_ft", test_batch_matmul_ft);
REGISTER_TEST("batch_matmul_tf", test_batch_matmul_tf);
REGISTER_TEST("batch_matmul_tt", test_batch_matmul_tt);
