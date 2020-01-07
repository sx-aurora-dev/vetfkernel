#include <cmath>
#include "test.h"

using namespace test;

//
// BinaryOp
//

template<typename TOUT, typename T>
bool test_BinaryOp(TestParam const& param,
                   Tensor<TOUT>& out,
                   Tensor<T> const& in0,
                   Tensor<T> const& in1,
                   Tensor<TOUT> const& exp,
                   int (*op)(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1))
{
    int ret = op(out, in0, in1);

    bool flag = false;
    if (ret == 0)
        flag = checkTensor(out, exp);

    if (param.verbose > 1 || (!flag && param.verbose > 0)) {
        fprintf(stderr, "in0 = \n");
        printTensor(in0);
        fprintf(stderr, "in1 = \n");
        printTensor(in1);
        fprintf(stderr, "out = \n");
        printTensor(out);
        fprintf(stderr, "expected = \n");
        printTensor(exp);
    }

    return flag;
}


template <typename TOUT, typename T, typename F>
int ref_Binop(Tensor<TOUT>& X, Tensor<T> const& Y, Tensor<T> const& Z, F op,
        TOUT* pX, T const* pY, T const* pZ, int dim)
{
  //fprintf(stderr, "%s: dim=%d X.stride[%d]=%d\n", __FUNCTION__, dim, dim, X.stride(dim));
  if (dim + 1 == X.dims()) {
    for (size_t i = 0; i < X.dim_size(dim); ++i) {
      T y = pY[i % Y.dim_size(dim)];
      T z = pZ[i % Z.dim_size(dim)];
      pX[i] = op(y, z);
      //fprintf(stderr, "%s: %8.3f = %8.3f op %8.3f\n", __FUNCTION__, pX[i], y, z);
    }
  } else {
    for (size_t i = 0; i < X.dim_size(dim); ++i) {
#if 0
      fprintf(stderr, "%s: dim=%d X.dim_size[%d]=%d i=%d %d %d\n",
              __FUNCTION__, dim, dim, X.dim_size(dim), i, Y.dim_size(dim), Y.stride(dim));
#endif
      TOUT* pX0 = pX + i * X.stride(dim);
      T const* pY0 = pY + (i % Y.dim_size(dim)) * Y.stride(dim);
      T const* pZ0 = pZ + (i % Z.dim_size(dim)) * Z.stride(dim);
      ref_Binop(X, Y, Z, op, pX0, pY0, pZ0, dim + 1);
    }
  }
  return 0;
}

template <typename TO, typename TI, typename F>
int ref_Binop(Tensor<TO>& X, Tensor<TI> const& Y, Tensor<TI> const& Z, F op)
{
  return ref_Binop(X, Y, Z, op, X.data(), Y.data(), Z.data(), 0);
}

#define DEFINE_REF_BINOP(name, func) \
  template <typename TO, typename TI> \
  int ref_##name(Tensor<TO>& X, Tensor<TI> const& Y, Tensor<TI> const& Z) { \
    return ref_Binop(X, Y, Z, [](TI y, TI z) -> TO { return func; }); \
  }

DEFINE_REF_BINOP(Add, y + z);
DEFINE_REF_BINOP(Sub, y - z);
DEFINE_REF_BINOP(Mul, y * z);
DEFINE_REF_BINOP(Div, y / z);
DEFINE_REF_BINOP(DivNoNaN, (z == TI(0.)) ? TO(0.0) : (y / z)) ;
DEFINE_REF_BINOP(Pow, std::pow(y, z));
DEFINE_REF_BINOP(SquaredDifference, (y - z) * (y - z));
DEFINE_REF_BINOP(RsqrtGrad, TI(-0.5) * z * y * y * y);
DEFINE_REF_BINOP(Minimum, y < z ? y : z);
DEFINE_REF_BINOP(Maximum, y > z ? y : z);
DEFINE_REF_BINOP(Equal, y == z);
DEFINE_REF_BINOP(NotEqual, y != z);
DEFINE_REF_BINOP(Less, y < z);
DEFINE_REF_BINOP(LessEqual, y <= z);
DEFINE_REF_BINOP(Greater, y > z);
DEFINE_REF_BINOP(GreaterEqual, y >= z);


bool test_Add_01(TestParam const& param)
{
    Tensor<float> out({1, 5, 10});
    Tensor<float> in0({1, 5, 10});
    Tensor<float> in1({1, 1, 10});
    Tensor<float> exp({1, 5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            ++c;
        }
    }

    for (size_t i = 0; i < 1; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            in1.data()[j] = j * 100;
        }
    }

    ref_Add(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, vml::add);
}

bool test_Add_02(TestParam const& param)
{
    Tensor<float> out({1, 5, 10});
    Tensor<float> in0({1, 5, 10});
    Tensor<float> in1({1, 5,  1});
    Tensor<float> exp({1, 5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            //exp.data()[i * 10 + j] = c + i * 100;
            ++c;
        }
    }

    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 1; ++j) {
            in1.data()[i] = i * 100;
        }
    }

    ref_Add(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, vml::add);
}

bool test_Add_03(TestParam const& param)
{
    Tensor<float> out({2, 3, 10});
    Tensor<float> in0({2, 1, 10});
    Tensor<float> in1({1, 3,  1});
    Tensor<float> exp({2, 3, 10});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    int c = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            in0.data()[i * 10 + j] = i * 10 + j;
        }
    }

    for (size_t i = 0; i < 3; ++i) {
        in1.data()[i] = i * 100;
    }

    ref_Add(exp, in0, in1);
    return test_BinaryOp(param, out, in0, in1, exp, vml::add);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_04(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16, 16, 32, 32});
    Tensor<T> in0({8, 16, 16, 32, 32});
    Tensor<T> in1({1, 16, 16, 1, 1});
    Tensor<T> exp({8, 16, 16, 32, 32});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_05(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16, 64, 8, 8});
    Tensor<T> in0({8, 16, 64, 8, 8});
    Tensor<T> in1({1, 16, 64, 1, 1});
    Tensor<T> exp({8, 16, 64, 8, 8});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_06(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16, 64, 8, 8});
    Tensor<T> in0({8, 16, 64, 8, 8});
    Tensor<T> in1({1,  1, 64, 1, 1});
    Tensor<T> exp({8, 16, 64, 8, 8});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_07(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16, 32, 16, 16});
    Tensor<T> in0({8, 16, 32, 16, 16});
    Tensor<T> in1({1, 16, 32,  1,  1});
    Tensor<T> exp({8, 16, 32, 16, 16});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_08(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16, 32, 16, 16});
    Tensor<T> in0({8, 16, 32, 16, 16});
    Tensor<T> in1({1,  1, 32,  1,  1});
    Tensor<T> exp({8, 16, 32, 16, 16});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_09(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({1, 16, 64,  1,  1});
    Tensor<T> in0({1, 16, 64,  1,  1});
    Tensor<T> in1({1,  1, 64,  1,  1});
    Tensor<T> exp({1, 16, 64,  1,  1});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_10(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({1, 16, 16,  1,  1});
    Tensor<T> in0({1, 16, 16,  1,  1});
    Tensor<T> in1({1,  1, 16,  1,  1});
    Tensor<T> exp({1, 16, 16,  1,  1});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_11(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({1, 16, 32,  1,  1});
    Tensor<T> in0({1, 16, 32,  1,  1});
    Tensor<T> in1({1,  1, 32,  1,  1});
    Tensor<T> exp({1, 16, 32,  1,  1});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_12(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16, 16, 32, 32});
    Tensor<T> in0({8, 16, 16, 32, 32});
    Tensor<T> in1({1,  1, 16,  1,  1});
    Tensor<T> exp({8, 16, 16, 32, 32});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_13(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({32, 12, 128, 128});
    Tensor<T> in0({32, 12, 128, 128});
    Tensor<T> in1({32,  1, 128, 128});
    Tensor<T> exp({32, 12, 128, 128});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_14(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 8, 8, 8});
    Tensor<T> in0({8, 8, 8, 8});
    Tensor<T> in1({8, 8, 8, 1});
    Tensor<T> exp({8, 8, 8, 8});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_15(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 8});
    Tensor<T> in0({1, 8});
    Tensor<T> in1({8, 8});
    Tensor<T> exp({8, 8});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_16(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16});
    Tensor<T> in0({1, 16});
    Tensor<T> in1({8,  1});
    Tensor<T> exp({8, 16});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

template<typename T, typename F0, typename F1>
bool test_BinaryOp_17(TestParam const& param, F0 f0, F1 f1)
{
    Tensor<T> out({8, 16});
    Tensor<T> in0({8,  1});
    Tensor<T> in1({1, 16});
    Tensor<T> exp({8, 16});

    for (size_t i = 0; i < out.nelems(); ++i)
        out.data()[i] = 0;

    for (size_t i = 0; i < in0.nelems(); ++i)
      in0.data()[i] = (T)drand48();

    for (size_t i = 0; i < in1.nelems(); ++i)
      in1.data()[i] = (T)drand48();

    f0(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, f1);
}

bool test_Add_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Add<float,float>, vml::add);
}

bool test_Add_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_Add<float,float>, vml::add);
}

bool test_Add_06(TestParam const& param)
{
  return test_BinaryOp_06<float>(param, ref_Add<float,float>, vml::add);
}

bool test_Sub_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Sub<float,float>, vml::sub);
}

bool test_Sub_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_Sub<float,float>, vml::sub);
}

bool test_Sub_14(TestParam const& param)
{
  return test_BinaryOp_14<float>(param, ref_Sub<float,float>, vml::sub);
}

bool test_Mul_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_06(TestParam const& param)
{
  return test_BinaryOp_06<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_07(TestParam const& param)
{
  return test_BinaryOp_07<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_08(TestParam const& param)
{
  return test_BinaryOp_08<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_09(TestParam const& param)
{
  return test_BinaryOp_09<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_10(TestParam const& param)
{
  return test_BinaryOp_10<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_11(TestParam const& param)
{
  return test_BinaryOp_11<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_12(TestParam const& param)
{
  return test_BinaryOp_12<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_13(TestParam const& param)
{
  return test_BinaryOp_13<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_15(TestParam const& param)
{
  return test_BinaryOp_15<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_16(TestParam const& param)
{
  return test_BinaryOp_16<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_Mul_17(TestParam const& param)
{
  return test_BinaryOp_17<float>(param, ref_Mul<float,float>, vml::mul);
}

bool test_SquaredDifference_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_SquaredDifference<float,float>, 
                                 vml::sqdiff);
}

bool test_Add_nn_1n_01(TestParam const& param)
{
    Tensor<float> out({5, 10});
    Tensor<float> in0({5, 10});
    Tensor<float> in1({10});
    Tensor<float> in1_exp({1, 10});
    Tensor<float> exp({5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            ++c;
        }
    }

    for (size_t j = 0; j < 10; ++j) {
      in1.data()[j] = c;
      in1_exp.data()[j] = c;
      c++;
    }

    ref_Add(exp, in0, in1_exp);

    return test_BinaryOp(param, out, in0, in1, exp, vml::add);
}



bool test_Add_nn_1n_02(TestParam const& param)
{
    Tensor<float> out({5, 10});
    Tensor<float> in0({5, 10});
    Tensor<float> in1({1, 10});
    Tensor<float> exp({5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            ++c;
        }
    }

    for (size_t i = 0; i < 1; ++i) {
      for (size_t j = 0; j < 10; ++j) {
	in1.data()[j] = c;
	c++;
      }
    }

    ref_Add(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, vml::add);
}



bool test_Mul_nn_1n_01(TestParam const& param)
{
    Tensor<float> out({5, 10});
    Tensor<float> in0({5, 10});
    Tensor<float> in1({10});
    Tensor<float> in1_exp({1, 10});
    Tensor<float> exp({5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            ++c;
        }
    }

    for (size_t j = 0; j < 10; ++j) {
      in1.data()[j] = c;
      in1_exp.data()[j] = c;
      c++;
    }

    ref_Mul(exp, in0, in1_exp);

    return test_BinaryOp(param, out, in0, in1, exp, vml::mul);
}



bool test_Mul_nn_1n_02(TestParam const& param)
{
    Tensor<float> out({5, 10});
    Tensor<float> in0({5, 10});
    Tensor<float> in1({1, 10});
    Tensor<float> exp({5, 10});

    int c = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            out.data()[i * 10 + j] = 0;
            in0.data()[i * 10 + j] = c;
            ++c;
        }
    }

    for (size_t i = 0; i < 1; ++i) {
      for (size_t j = 0; j < 10; ++j) {
	in1.data()[j] = c;
	c++;
      }
    }

    ref_Mul(exp, in0, in1);

    return test_BinaryOp(param, out, in0, in1, exp, vml::mul);
}



template<typename TOUT, typename TIN>
bool test_generic(TestParam const& param,
		  int (*ref_op)(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1),
		  int (*op)(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1))
{
  Tensor<TOUT> exp({4, 8, 4});
  Tensor<TOUT> out({4, 8, 4});
  Tensor<TIN>  in0({1, 8, 2});
  Tensor<TIN>  in1({4, 4, 4});

  for (size_t i = 0; i < 1; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      for (size_t k = 0; k < 2; ++k) {
	size_t index = (i * 8 + j) * 2 + k;
	in0.data()[index] = (TIN)drand48();
      }
    }
  }

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      for (size_t k = 0; k < 4; ++k) {
	size_t index = (i * 4 + j) * 4 + k;
	in1.data()[index] = (TIN)drand48();
      }
    }
  }

  ref_op(exp, in0, in1);

  return test_BinaryOp(param, out, in0, in1, exp, op);
}

bool test_Add_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_Add, vml::add);
}

bool test_Sub_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_Sub, vml::sub);
}

bool test_Mul_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_Mul, vml::mul);
}

bool test_Div_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_Div, vml::div);
}

bool test_DivNoNaN_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_DivNoNaN, vml::divnonan);
}

bool test_Pow_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_Pow, vml::pow);
}

bool test_SquaredDifference(TestParam const& param)
{
  return test_generic<float, float>(param, ref_SquaredDifference, vml::rsqrt_grad);
}

bool test_RsqrtGrad(TestParam const& param)
{
  return test_generic<float, float>(param, ref_RsqrtGrad, vml::rsqrt_grad);
}

bool test_Maximum_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_Maximum, vml::maximum);
}

bool test_Minimum_generic(TestParam const& param)
{
  return test_generic<float, float>(param, ref_Minimum, vml::minimum);
}

bool test_Equal_generic(TestParam const& param)
{
  return test_generic<bool, float>(param, ref_Equal, vml::equal);
}

bool test_NotEqual_generic(TestParam const& param)
{
  return test_generic<bool, float>(param, ref_NotEqual, vml::notEqual);
}

bool test_Less_generic(TestParam const& param)
{
  return test_generic<bool, float>(param, ref_Less, vml::less);
}

bool test_LessEqual_generic(TestParam const& param)
{
  return test_generic<bool, float>(param, ref_LessEqual, vml::lessEqual);
}

bool test_Greater_generic(TestParam const& param)
{
  return test_generic<bool, float>(param, ref_Greater, vml::greater);
}

bool test_GreaterEqual_generic(TestParam const& param)
{
  return test_generic<bool, float>(param, ref_GreaterEqual, vml::greaterEqual);
}

REGISTER_TEST( "Add_01", test_Add_01 );
REGISTER_TEST( "Add_02", test_Add_02 );
REGISTER_TEST( "Add_03", test_Add_03 );
REGISTER_TEST( "Add_04", test_Add_04 );
REGISTER_TEST( "Add_05", test_Add_05 );
REGISTER_TEST( "Add_06", test_Add_06 );
REGISTER_TEST( "Sub_04", test_Sub_04 );
REGISTER_TEST( "Sub_05", test_Sub_05 );
REGISTER_TEST( "Sub_14", test_Sub_14 );
REGISTER_TEST( "Mul_04", test_Mul_04 );
REGISTER_TEST( "Mul_05", test_Mul_05 );
REGISTER_TEST( "Mul_06", test_Mul_06 );
REGISTER_TEST( "Mul_07", test_Mul_07 );
REGISTER_TEST( "Mul_08", test_Mul_08 );
REGISTER_TEST( "Mul_09", test_Mul_09 );
REGISTER_TEST( "Mul_10", test_Mul_10 );
REGISTER_TEST( "Mul_11", test_Mul_11 );
REGISTER_TEST( "Mul_12", test_Mul_12 );
REGISTER_TEST( "Mul_13", test_Mul_13 );
REGISTER_TEST( "Mul_15", test_Mul_15 );
REGISTER_TEST( "Mul_16", test_Mul_16 );
REGISTER_TEST( "Mul_17", test_Mul_17 );
REGISTER_TEST( "SquaredDifference_05", test_SquaredDifference_05 );

// 配列の次元数を最大の次元数に変更(次元数2以下のケースの仕様変更)
REGISTER_TEST( "Add[nn_1n]_01 (obsolute)",  test_Add_nn_1n_01         ); // 次元数を合わせないケース
REGISTER_TEST( "Add[nn_1n]_02",             test_Add_nn_1n_02         );
REGISTER_TEST( "Mul[nn_1n]_01 (obsolute)",  test_Mul_nn_1n_01         ); // 次元数を合わせないケース
REGISTER_TEST( "Mul[nn_1n]_02",             test_Mul_nn_1n_02         );

// 汎用kernelに落ちるケース
REGISTER_TEST( "Add[generic]",              test_Add_generic          );
REGISTER_TEST( "Sub[generic]",              test_Sub_generic          );
REGISTER_TEST( "Mul[generic]",              test_Mul_generic          );
REGISTER_TEST( "Div[generic]",              test_Div_generic          );
REGISTER_TEST( "Div No NaN[generic]",       test_DivNoNaN_generic     );
REGISTER_TEST( "Pow[generic]",              test_Pow_generic          );
REGISTER_TEST( "SquareDifference[generic]", test_SquaredDifference    );
REGISTER_TEST( "RsqrtGrad[generic]",        test_RsqrtGrad            );
REGISTER_TEST( "Minimum[generic]",          test_Minimum_generic      );
REGISTER_TEST( "Maximum[generic]",          test_Maximum_generic      );
REGISTER_TEST( "Equal[generic]",            test_Equal_generic        );
REGISTER_TEST( "NotEqual[generic]",         test_NotEqual_generic     );
REGISTER_TEST( "Less[generic]",             test_Less_generic         );
REGISTER_TEST( "LessEqual[generic]",        test_LessEqual_generic    );
REGISTER_TEST( "Greater[generic]",          test_Greater_generic      );
REGISTER_TEST( "GreaterEqual[generic]",     test_GreaterEqual_generic );
