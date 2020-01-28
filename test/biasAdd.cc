#include "test.h"

using namespace test;

bool test_biasAdd_NHWC_01(TestParam const & param)
{
  Tensor<float> in({2, 2, 2, 2});
  Tensor<float> out({2, 2, 2, 2});
  Tensor<float> bias({2});
  Tensor<float> exp({2, 2, 2, 2});

  bias.copy({1, 2});

  in.copy( {0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15});
  exp.copy({1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 17});

  if (vml::biasAdd(out, in, bias, FORMAT_NHWC) != 0) {
    return 1;
  }

  return checkTensor(out, exp);
}

bool test_biasAdd_NCHW_01(TestParam const & param)
{
  Tensor<float> in({2, 2, 2, 2});
  Tensor<float> out({2, 2, 2, 2});
  Tensor<float> bias({2});
  Tensor<float> exp({2, 2, 2, 2});

  bias.copy({1, 2});

  in.copy( {0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15});
  exp.copy({1, 2, 3, 4, 6, 7, 8, 9, 9, 10, 11, 12, 14, 15, 16, 17});

  if (vml::biasAdd(out, in, bias, FORMAT_NCHW) != 0) {
    return 1;
  }

  return checkTensor(out, exp);
}

REGISTER_TEST("biasAdd_NHWC_01", test_biasAdd_NHWC_01);
REGISTER_TEST("biasAdd_NCHW_01", test_biasAdd_NCHW_01);
