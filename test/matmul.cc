#include "test.h"

using namespace test;

extern "C" {
  int op_MatMul(const void* args, size_t len);
}

namespace {

template<typename T>
bool test_matmul(
    TestParam const& param,
    Tensor<T> const& in_x,
    Tensor<T> const& in_y,
    Tensor<T>& out,
    int64_t dim_x[2],
    int64_t dim_y[2],
    const bool adj_x,
    const bool adj_y,
    Tensor<T> const& exp)
{
  struct Args2 {
    int dtype;
    uint64_t a;
    uint64_t b;
    uint64_t out;
    int64_t dim_size_a[2];
    int64_t dim_size_b[2];
    int32_t transpose_a;
    int32_t transpose_b;
  } args;

  size_t args_size;
  bool ret = false;

  args_size = sizeof(args);
  memset((char *)(&args), 0, args_size);

  args.dtype = DT_FLOAT;
  args.a     = reinterpret_cast<uint64_t>(in_x.data());
  args.b     = reinterpret_cast<uint64_t>(in_y.data());
  args.out   = reinterpret_cast<uint64_t>(out.data());

  args.dim_size_a[0] = dim_x[0];
  args.dim_size_a[1] = dim_x[1];
  args.dim_size_b[0] = dim_y[0];
  args.dim_size_b[1] = dim_y[1];

  args.transpose_a   = (int32_t)adj_x;
  args.transpose_b   = (int32_t)adj_y;

  ret = op_MatMul((const void *)&args, args_size);

  if (ret != 0)
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

bool test_matmul_ff(TestParam const& param)
{
  Tensor<float> in_x({2, 3});
  Tensor<float> in_y({3, 5});
  Tensor<float> out({2, 5});
  Tensor<float> exp({2, 5});
  int64_t dim_x[2] = {2, 3};
  int64_t dim_y[2] = {3, 5};

  in_x.copy({0, 1, 2,
             3, 4, 5});
  in_y.copy({ 0, 1, 2, 3, 4,
              5, 6, 7, 8, 9,
             10,11,12,13,14});
  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118});

  return test_matmul(param, in_x, in_y, out, dim_x, dim_y, false, false, exp);
}

bool test_matmul_ft(TestParam const& param)
{
  Tensor<float> in_x({2, 3});
  Tensor<float> in_y({5, 3});
  Tensor<float> out({2, 5});
  Tensor<float> exp({2, 5});
  int64_t dim_x[2] = {2, 3};
  int64_t dim_y[2] = {5, 3};

  in_x.copy({0, 1, 2, 3, 4, 5});
  in_y.copy({ 0, 5,10,
              1, 6,11,
              2, 7,12,
	      3, 8,13,
              4, 9,14});
  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118});

  return test_matmul(param, in_x, in_y, out, dim_x, dim_y, false, true, exp);
}

bool test_matmul_tf(TestParam const& param)
{
  Tensor<float> in_x({3, 2});
  Tensor<float> in_y({3, 5});
  Tensor<float> out({2, 5});
  Tensor<float> exp({2, 5});
  int64_t dim_x[2] = {3, 2};
  int64_t dim_y[2] = {3, 5};

  in_x.copy({0, 3, 1, 4, 2, 5});
  in_y.copy({ 0, 1, 2, 3, 4,
              5, 6, 7, 8, 9,
             10,11,12,13,14});
  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118});

  return test_matmul(param, in_x, in_y, out, dim_x, dim_y, true, false, exp);
}

bool test_matmul_tt(TestParam const& param)
{
  Tensor<float> in_x({3, 2});
  Tensor<float> in_y({5, 3});
  Tensor<float> out({2, 5});
  Tensor<float> exp({2, 5});
  int64_t dim_x[2] = {3, 2};
  int64_t dim_y[2] = {5, 3};

  in_x.copy({0, 3, 1, 4, 2, 5});
  in_y.copy({ 0, 5,10,
              1, 6,11,
              2, 7,12,
	      3, 8,13,
              4, 9,14});
  exp.copy({ 25, 28, 31, 34, 37,
             70, 82, 94,106,118});

  return test_matmul(param, in_x, in_y, out, dim_x, dim_y, true, true, exp);
}

} // namespace

REGISTER_TEST("matmul_ff", test_matmul_ff);
REGISTER_TEST("matmul_ft", test_matmul_ft);
REGISTER_TEST("matmul_tf", test_matmul_tf);
REGISTER_TEST("matmul_tt", test_matmul_tt);
