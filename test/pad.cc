#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <functional>
#include "test.h"
using namespace test;

#define VERBOSE(_DIMS)						\
  if (param.verbose > 1 || (!flag && param.verbose > 0)) {	\
    fprintf(stderr, "pad_value = %f\n", pad_value);		\
    fprintf(stderr, "input = \n");				\
    printTensor(input);						\
    fprintf(stderr, "[");					\
    for (int i=0; i<(_DIMS)*2; i++){				\
      fprintf(stderr, "[%d, %d] ", paddings[i], paddings[i+1]);	\
    }								\
    fprintf(stderr, "]\n");					\
    fprintf(stderr, "output = \n");				\
    printTensor(output);					\
    fprintf(stderr, "expected = \n");				\
    printTensor(expected);					\
  }

bool test_Pad_2D_f32(TestParam const& param)
{
#define DIMS 2
  Tensor<float> input({2, 3});
  int32_t       dims = DIMS;
  uint32_t      mult = sizeof(float);
  int32_t       paddings[DIMS*2] = {1,5,7,9};
  float         pad_value = 88.;
  Tensor<float> output({8, 19});
  Tensor<float> expected({8, 19});

  float in[2][3]  = {{ 1., 2., 3.}, {4., 5., 6.}};
  float exp[8][19] = {
    /* 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17 18*/
    {88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
    {88.,88.,88.,88.,88.,88.,88., 1., 2., 3.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
    {88.,88.,88.,88.,88.,88.,88., 4., 5., 6.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
    {88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
    {88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
    {88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
    {88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
    {88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.,88.},
  };

  memcpy(input.data(),    in,  sizeof(float) * 2 *  3);
  memcpy(expected.data(), exp, sizeof(float) * 8 * 19);

  vml::pad(output, input, pad_value, paddings);

  bool flag = checkTensor(output, expected);

#if 1
  VERBOSE(DIMS)
#else
  if (param.verbose > 1 || (!flag && param.verbose > 0)) {
    fprintf(stderr, "pad_value = %f\n", pad_value);
    fprintf(stderr, "input = \n");
    printTensor(input);
    fprintf(stderr, "[");
    for (int i=0; i<DIMS*2; i++){
      fprintf(stderr, "[%d, %d] ", paddings[i], paddings[i+1]);
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "output = \n");
    printTensor(output);
    fprintf(stderr, "expected = \n");
    printTensor(expected);
  }
#endif

  return flag;
#undef DIMS
}


REGISTER_TEST("Pad_2D_f32", test_Pad_2D_f32); // 2D float32
//REGISTER_TEST("Pad_2D_f64", test_Pad_2D_f64); // 2D float64


