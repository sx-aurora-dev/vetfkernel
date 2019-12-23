#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <functional>
#include "test.h"
using namespace test;

#define CHECK(DIMS_)						\
  vml::pad(output, input, pad_value, paddings);			\
  bool flag = checkTensor(output, expected);			\
  if (param.verbose > 1 || (!flag && param.verbose > 0)) {	\
    fprintf(stderr, "pad_value = %f\n", pad_value);		\
    fprintf(stderr, "input = \n");				\
    printTensor(input);						\
    fprintf(stderr, "[");					\
    for (int i=0; i<(DIMS_)*2; i++){				\
      fprintf(stderr, "[%d, %d] ", paddings[i], paddings[i+1]);	\
    }								\
    fprintf(stderr, "]\n");					\
    fprintf(stderr, "output = \n");				\
    printTensor(output);					\
    fprintf(stderr, "expected = \n");				\
    printTensor(expected);					\
  }								\
  return flag;


// float 2D test
bool test_Pad_2D_f32(TestParam const& param)
{
#define DIMS 2
#define TTYPE float
  Tensor<TTYPE> input({2, 3});
  int32_t       dims = (DIMS);
  uint32_t      mult = sizeof(TTYPE);
  int32_t       paddings[(DIMS)*2] = {1,5,7,9};
  TTYPE         pad_value = 88.;
  Tensor<TTYPE> output({8, 19});
  Tensor<TTYPE> expected({8, 19});

  TTYPE in[2][3]  = {{ 1., 2., 3.}, {4., 5., 6.}};
  TTYPE exp[8][19] = {
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

  memcpy(input.data(),    in,  sizeof(TTYPE) * 2 *  3);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 8 * 19);

  CHECK((DIMS))

#undef TTYPE
#undef DIMS
}

// double 2D test
bool test_Pad_2D_f64(TestParam const& param)
{
#define DIMS 2
#define TTYPE double
  Tensor<TTYPE> input({2, 3});
  int32_t       dims = (DIMS);
  uint32_t      mult = sizeof(TTYPE);
  int32_t       paddings[(DIMS)*2] = {1,5,7,9};
  TTYPE         pad_value = 99.;
  Tensor<TTYPE> output({8, 19});
  Tensor<TTYPE> expected({8, 19});

  TTYPE in[2][3]  = {{ 1., 2., 3.}, {4., 5., 6.}};
  TTYPE exp[8][19] = {
    /* 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17 18*/
    {99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
    {99.,99.,99.,99.,99.,99.,99., 1., 2., 3.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
    {99.,99.,99.,99.,99.,99.,99., 4., 5., 6.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
    {99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
    {99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
    {99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
    {99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
    {99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.,99.},
  };

  memcpy(input.data(),    in,  sizeof(TTYPE) * 2 *  3);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 8 * 19);

  CHECK((DIMS))

#undef TTYPE
#undef DIMS
}

// float 3D test
bool test_Pad_3D_f32(TestParam const& param)
{
#define DIMS 3
#define TTYPE float
  Tensor<TTYPE> input({2, 3, 4});
  int32_t       dims = (DIMS);
  uint32_t      mult = sizeof(TTYPE);
  int32_t       paddings[(DIMS)*2] = {1,1,1,1,1,1};
  TTYPE         pad_value = 66.;
  Tensor<TTYPE> output({4,5,6});
  Tensor<TTYPE> expected({4,5,6});

  TTYPE in[2][3][4]  = {
    {{  1.0,   2.0,   3.0,   4.0}, {  5.0,   6.0,   7.0,   8.0}, {  9.0,  10.0,  11.0,  12.0}},
    {{100.0, 110.0, 120.0, 130.0}, {140.0, 150.0, 160.0, 170.0}, {180.0, 190.0, 200.0, 210.0}}
  };
  TTYPE exp[4][5][6] = {
    {{ 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.}
    },
    
    {{ 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,    1.,    2.,    3.,    4.,   66.},
     { 66.,    5.,    6.,    7.,    8.,   66.},
     { 66.,    9.,   10.,   11.,   12.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.}
    },
    
    {{ 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,  100.,  110.,  120.,  130.,   66.},
     { 66.,  140.,  150.,  160.,  170.,   66.},
     { 66.,  180.,  190.,  200.,  210.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.}
    },
    
    {{ 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.},
     { 66.,   66.,   66.,   66.,   66.,   66.}
    }
  };

  memcpy(input.data(),    in,  sizeof(TTYPE) * 2 * 3 * 4);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 4 * 5 * 6);

  CHECK((DIMS))

#undef TTYPE
#undef DIMS
}

bool test_Pad_3D_f64(TestParam const& param)
{
#define DIMS 3
#define TTYPE float
  Tensor<TTYPE> input({2, 3, 4});
  int32_t       dims = (DIMS);
  uint32_t      mult = sizeof(TTYPE);
  int32_t       paddings[(DIMS)*2] = {1,2,1,2,1,2};
  TTYPE         pad_value = 55.;
  Tensor<TTYPE> output({5,6,7});
  Tensor<TTYPE> expected({5,6,7});

  TTYPE in[2][3][4]  = {
    {{  1.0,   2.0,   3.0,   4.0}, {  5.0,   6.0,   7.0,   8.0}, {  9.0,  10.0,  11.0,  12.0}},
    {{100.0, 110.0, 120.0, 130.0}, {140.0, 150.0, 160.0, 170.0}, {180.0, 190.0, 200.0, 210.0}}
  };
  TTYPE exp[5][6][7] = {
    {{ 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.}
    },
    
    {{ 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,    1.,    2.,    3.,    4.,   55.,   55.},
     { 55.,    5.,    6.,    7.,    8.,   55.,   55.},
     { 55.,    9.,   10.,   11.,   12.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.}
    },
    
    {{ 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,  100.,  110.,  120.,  130.,   55.,   55.},
     { 55.,  140.,  150.,  160.,  170.,   55.,   55.},
     { 55.,  180.,  190.,  200.,  210.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.}
    },
    
    {{ 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.}
    },
    
    {{ 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.},
     { 55.,   55.,   55.,   55.,   55.,   55.,   55.}
    }
  };

  memcpy(input.data(),    in,  sizeof(TTYPE) * 2 * 3 * 4);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 5 * 6 * 7);

  CHECK((DIMS))

#undef TTYPE
#undef DIMS
}

// float 3D test
bool test_Pad_3D_f32_0(TestParam const& param)
{
#define DIMS 3
#define TTYPE float
  Tensor<TTYPE> input({2, 3, 4});
  int32_t       dims = (DIMS);
  uint32_t      mult = sizeof(TTYPE);
  int32_t       paddings[(DIMS)*2] = {0,0,2,3,0,0};
  TTYPE         pad_value = 44.;
  Tensor<TTYPE> output({2,8,4});
  Tensor<TTYPE> expected({2,8,4});

  TTYPE in[2][3][4]  = {
    {
      {  1.0,   2.0,   3.0,   4.0},
      {  5.0,   6.0,   7.0,   8.0},
      {  9.0,  10.0,  11.0,  12.0}
    },
    {
      {100.0, 110.0, 120.0, 130.0},
      {140.0, 150.0, 160.0, 170.0},
      {180.0, 190.0, 200.0, 210.0}
    }
  };
  TTYPE exp[2][8][4] = {
    {
      { 44.,  44.,  44.,  44.},
      { 44.,  44.,  44.,  44.},
      {  1.,   2.,   3.,   4.},
      {  5.,   6.,   7.,   8.},
      {  9.,  10.,  11.,  12.},
      { 44.,  44.,  44.,  44.},
      { 44.,  44.,  44.,  44.},
      { 44.,  44.,  44.,  44.}
    },

    {
      { 44.,  44.,  44.,  44.},
      { 44.,  44.,  44.,  44.},
      {100., 110., 120., 130.},
      {140., 150., 160., 170.},
      {180., 190., 200., 210.},
      { 44.,  44.,  44.,  44.},
      { 44.,  44.,  44.,  44.},
      { 44.,  44.,  44.,  44.}
    }
  };

  memcpy(input.data(),    in,  sizeof(TTYPE) * 2 * 3 * 4);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 2 * 8 * 4);

  CHECK((DIMS))

#undef TTYPE
#undef DIMS
}

#undef CHECK

REGISTER_TEST("Pad_01", test_Pad_2D_f32);   // 2D float32
REGISTER_TEST("Pad_02", test_Pad_2D_f64);   // 2D float64
REGISTER_TEST("Pad_03", test_Pad_3D_f32);   // 3D float32
REGISTER_TEST("Pad_04", test_Pad_3D_f64);   // 3D float64
REGISTER_TEST("Pad_05", test_Pad_3D_f32_0); // 3D float32


