#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <functional>
#include "test.h"
using namespace test;

#define CHECK							\
  vml::conv2d(input, filter, output, convparams);		\
  bool flag = checkTensor(output, expected);			\
  if (param.verbose > 1 || (!flag && param.verbose > 0)) {	\
    fprintf(stderr, "input = \n");				\
    printTensor(input);						\
    fprintf(stderr, "filter = \n");				\
    printTensor(filter);						\
    fprintf(stderr, "output = \n");				\
    printTensor(output);					\
    fprintf(stderr, "expected = \n");				\
    printTensor(expected);					\
  }								\
  return flag;


// Ch:F,Pad:S,Fil:1d,Type:f32
bool test_Conv2D_01(TestParam const& param)
{
#define TTYPE float
  Tensor<TTYPE> input({1, 1, 3, 4});
  Tensor<TTYPE> filter({1, 1, 1, 2});

  Tensor<TTYPE> output({1, 1, 3, 4});
  Tensor<TTYPE> expected({1, 1, 3, 4});

  std::vector<int> convparams = { 1, 1, 1, 1, 0, 0, 1};
  /*                             stride lila  padd  NCHW */

  TTYPE in[1][1][3][4] = {
    {
      {
	{ 1,  2,  3,  4 },
	{ 5,  6,  7,  8 },
	{ 9, 10, 11, 12 }
      }
    }
  }; 
  TTYPE fil[1][1][1][2] = {
    {
      {
	{ 1, 2 }
      }
    }
  };
  TTYPE exp[1][1][3][4] = {
    {
      {
	{ 5,  8, 11,  4 },
	{17, 20, 23,  8 },
	{29, 32, 35, 12 }
      }
    }
  }; 

  memcpy(input.data(),    in,  sizeof(TTYPE) * 1 * 1 * 3 * 4);
  memcpy(filter.data(),   fil, sizeof(TTYPE) * 1 * 1 * 1 * 2);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 1 * 1 * 3 * 4);

  CHECK

#undef TTYPE
}

// Ch:F,Pad:S,Fil:2d,Type:f32
bool test_Conv2D_02(TestParam const& param)
{
#define TTYPE float
  Tensor<TTYPE> input({1, 1, 3, 4});
  Tensor<TTYPE> filter({1, 1, 2, 2});

  Tensor<TTYPE> output({1, 1, 3, 4});
  Tensor<TTYPE> expected({1, 1, 3, 4});

  std::vector<int> convparams = { 1, 1, 1, 1, 0, 0, 1};
  /*                             stride lila  padd  NCHW */

  TTYPE in[1][1][3][4] = {
    {
      {
	{ 1,  2,  3,  4 },
	{ 5,  6,  7,  8 },
	{ 9, 10, 11, 12 }
      }
    }
  }; 
  TTYPE fil[1][1][2][2] = {
    {
      {
	{ 1, 2 },
	{ 3, 4 }
      }
    }
  };
  TTYPE exp[1][1][3][4] = {
    {
      {
	{44, 54, 64, 28 },
	{84, 94,104, 44 },
	{29, 32, 35, 12 }
      }
    }
  }; 

  memcpy(input.data(),    in,  sizeof(TTYPE) * 1 * 1 * 3 * 4);
  memcpy(filter.data(),   fil, sizeof(TTYPE) * 1 * 1 * 2 * 2);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 1 * 1 * 3 * 4);

  CHECK

#undef TTYPE
}

// Ch:F,Pad:V,Fil:1d,Type:f32
bool test_Conv2D_03(TestParam const& param)
{
#define TTYPE float
  Tensor<TTYPE> input({1, 1, 3, 4});
  Tensor<TTYPE> filter({1, 1, 1, 2});

  Tensor<TTYPE> output({1, 1, 3, 3});
  Tensor<TTYPE> expected({1, 1, 3, 3});

  std::vector<int> convparams = { 1, 1, 1, 1, 0, 0, 1};
  /*                             stride lila  padd  NCHW */

  TTYPE in[1][1][3][4] = {
    {
      {
	{ 1,  2,  3,  4 },
	{ 5,  6,  7,  8 },
	{ 9, 10, 11, 12 }
      }
    }
  }; 
  TTYPE fil[1][1][1][2] = {
    {
      {
	{ 1, 2 }
      }
    }
  };
  TTYPE exp[1][1][3][3] = {
    {
      {
	{ 5,  8, 11 },
	{17, 20, 23 },
	{29, 32, 35 }
      }
    }
  }; 

  memcpy(input.data(),    in,  sizeof(TTYPE) * 1 * 1 * 3 * 4);
  memcpy(filter.data(),   fil, sizeof(TTYPE) * 1 * 1 * 1 * 2);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 1 * 1 * 3 * 3);

  CHECK

#undef TTYPE
}

// Ch:F,Pad:V,Fil:2d,Type:f32
bool test_Conv2D_04(TestParam const& param)
{
#define TTYPE float
  Tensor<TTYPE> input({1, 1, 3, 4});
  Tensor<TTYPE> filter({1, 1, 2, 2});

  Tensor<TTYPE> output({1, 1, 2, 3});
  Tensor<TTYPE> expected({1, 1, 2, 3});

  std::vector<int> convparams = { 1, 1, 1, 1, 0, 0, 1};
  /*                             stride lila  padd  NCHW */

  TTYPE in[1][1][3][4] = {
    {
      {
	{ 1,  2,  3,  4 },
	{ 5,  6,  7,  8 },
	{ 9, 10, 11, 12 }
      }
    }
  }; 
  TTYPE fil[1][1][2][2] = {
    {
      {
	{ 1, 2 },
	{ 3, 4 }
      }
    }
  };
  TTYPE exp[1][1][2][3] = {
    {
      {
	{44, 54, 64 },
	{84, 94,104 }
      }
    }
  }; 

  memcpy(input.data(),    in,  sizeof(TTYPE) * 1 * 1 * 3 * 4);
  memcpy(filter.data(),   fil, sizeof(TTYPE) * 1 * 1 * 2 * 2);
  memcpy(expected.data(), exp, sizeof(TTYPE) * 1 * 1 * 2 * 3);

  CHECK

#undef TTYPE
}

#undef CHECK

// means:
//   Ch : F(irst), L(ast)
//   Pad: S(ame), V(alid)
//   Fil: filter dimensionw
//   f32: Tensor type
REGISTER_TEST("Conv2D_01-Ch:F,Pad:S,Fil:1d,Type:f32", test_Conv2D_01);
REGISTER_TEST("Conv2D_02-Ch:F,Pad:S,Fil:2d,Type:f32", test_Conv2D_02);
REGISTER_TEST("Conv2D_03-Ch:F,Pad:V,Fil:1d,Type:f32", test_Conv2D_03);
REGISTER_TEST("Conv2D_04-Ch:F,Pad:V,Fil:2d,Type:f32", test_Conv2D_04);
// VML only support Channel First
//REGISTER_TEST("Conv2D_05-Ch:L,Pad:S,Fil:1d,Type:f32", test_Conv2D_05);
//REGISTER_TEST("Conv2D_06-Ch:L,Pad:S,Fil:2d,Type:f32", test_Conv2D_06);
//REGISTER_TEST("Conv2D_07-Ch:L,Pad:V,Fil:1d,Type:f32", test_Conv2D_07);
//REGISTER_TEST("Conv2D_08-Ch:L,Pad:V,Fil:2d,Type:f32", test_Conv2D_08);
