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
}

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
    int ret = op(out.tensor(), in0.tensor(), in1.tensor());

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

template <typename T, typename F>
int ref_Binop(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z, F op)
{
  return ref_Binop(X, Y, Z, op, X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Add(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN y, TIN z) -> TOUT { return y + z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Sub(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN y, TIN z) -> TOUT { return y - z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Mul(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN y, TIN z) -> TOUT { return y * z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Div(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN y, TIN z) -> TOUT { return y / z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_DivNoNaN(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN y, TIN z) -> TOUT { return (z == TIN(0.)) ? TOUT(0.0) : (y / z); },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Pow(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN y, TIN z) -> TOUT { return std::pow(y, z); },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_SquaredDifference(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN y, TIN z) -> TOUT { return (y - z) * (y - z); },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_RsqrtGrad(Tensor<TOUT>& X, Tensor<TIN> const& Y, Tensor<TIN> const& Z)
{
  return ref_Binop(X, Y, Z, [](TIN out, TIN gradout) -> TOUT { return TIN(-0.5) * gradout * out * out * out; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Minimum(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 < i1 ? i0 : i1; },
		   out.data(), in0.data(), in1.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Maximum(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 > i1 ? i0 : i1; },
		   out.data(), in0.data(), in1.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Equal(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 == i1; },
		   out.data(), in0.data(), in1.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_NotEqual(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 != i1; },
		   out.data(), in0.data(), in1.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Less(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 < i1; },
		   out.data(), in0.data(), in1.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_LessEqual(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 <= i1; },
		   out.data(), in0.data(), in1.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_Greater(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 > i1; },
		   out.data(), in0.data(), in1.data(), 0);
}

template <typename TOUT, typename TIN>
int ref_GreaterEqual(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1)
{
  return ref_Binop(out, in0, in1,
		   [](TIN i0, TIN i1) -> TOUT { return i0 >= i1; },
		   out.data(), in0.data(), in1.data(), 0);
}


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

bool test_SquaredDifference_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_SquaredDifference<float,float>, 
                                 vml::sqdiff);
}

bool test_AvgPool_01(TestParam const& param)
{
  vml::Tensor out = makeTensor<float>(4, {1, 1, 2, 2});
  vml::Tensor in = makeTensor<float>(4, {1, 1, 2, 2});
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

  float* pin = reinterpret_cast<float*>(in.addr);
  for (int i = 0; i < 4; ++i)
    pin[i] = i + 1.0;


  if (vml::avgpool(out, in, p) != 0)
    return false;
  float* pout = reinterpret_cast<float*>(out.addr);

  bool flag = true;
  for (int i = 0; i < 4; ++i) {
    bool tmp = pout[i] == expected[i];
    flag &= tmp;
    if (!tmp && param.verbose > 0) {
      fprintf(stderr, "pout[%d]=%f (expected is %f)\n", i, pout[i], expected[i]);
    }
  }

  return flag;
}

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

bool test_AvgPool_02(TestParam const& param)
{
  vml::Tensor out = makeTensor<float>(4, {1, 1, 3, 3});
  vml::Tensor in = makeTensor<float>(4, {1, 1, 3, 3});
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

  float* pin = reinterpret_cast<float*>(in.addr);
  for (int i = 0; i < 9; ++i)
    pin[i] = i + 1.0;

  return test_AvgPool(param, out, in, p, expected, 9);
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

int main(int argc, char* argv[])
{
    Test tests[] = {
        { "Add_01", test_Add_01 },
        { "Add_02", test_Add_02 },
        { "Add_03", test_Add_03 },
        { "Add_04", test_Add_04 },
        { "Add_05", test_Add_05 },
        { "Add_06", test_Add_06 },

        { "Sub_04", test_Sub_04 },
        { "Sub_05", test_Sub_05 },

        { "Mul_04", test_Mul_04 },
        { "Mul_05", test_Mul_05 },
        { "Mul_06", test_Mul_06 },
        { "Mul_07", test_Mul_07 },
        { "Mul_08", test_Mul_08 },
        { "Mul_09", test_Mul_09 },
        { "Mul_10", test_Mul_10 },
        { "Mul_11", test_Mul_11 },
        { "Mul_12", test_Mul_12 },

        { "SquaredDifference_05", test_SquaredDifference_05 },

        { "AvgPool_01", test_AvgPool_01 },
        { "AvgPool_02", test_AvgPool_02 },
        { "Mul_12", test_Mul_12 },

	// 配列の次元数を最大の次元数に変更(次元数2以下のケースの仕様変更)
	{ "Add[nn_1n]_01 (obsolute)",  test_Add_nn_1n_01         },	// 次元数を合わせないケース
	{ "Add[nn_1n]_02",             test_Add_nn_1n_02         },
	{ "Mul[nn_1n]_01 (obsolute)",  test_Mul_nn_1n_01         },	// 次元数を合わせないケース
	{ "Mul[nn_1n]_02",             test_Mul_nn_1n_02         },

	// 汎用kernelに落ちるケース
	{ "Add[generic]",              test_Add_generic          },
	{ "Sub[generic]",              test_Sub_generic          },
	{ "Mul[generic]",              test_Mul_generic          },
	{ "Div[generic]",              test_Div_generic          },
	{ "Div No NaN[generic]",       test_DivNoNaN_generic     },
	{ "Pow[generic]",              test_Pow_generic          },
	{ "SquareDifference[generic]", test_SquaredDifference    },
	{ "RsqrtGrad[generic]",        test_RsqrtGrad            },
        { "Minimum[generic]",          test_Minimum_generic      },
        { "Maximum[generic]",          test_Maximum_generic      },
        { "Equal[generic]",            test_Equal_generic        },
        { "NotEqual[generic]",         test_NotEqual_generic     },
        { "Less[generic]",             test_Less_generic         },
        { "LessEqual[generic]",        test_LessEqual_generic    },
        { "Greater[generic]",          test_Greater_generic      },
        { "GreaterEqual[generic]",     test_GreaterEqual_generic },

#define DEFINE_TEST_01(T) {#T "_01", test_##T##_01}
        DEFINE_TEST_01(Abs),
        DEFINE_TEST_01(Sign),
        DEFINE_TEST_01(Exp),
        DEFINE_TEST_01(Expm1),
        DEFINE_TEST_01(Floor),
        DEFINE_TEST_01(Neg),
        DEFINE_TEST_01(Log),
        DEFINE_TEST_01(Log1p),
        DEFINE_TEST_01(Reciprocal),
        DEFINE_TEST_01(Rsqrt),
        DEFINE_TEST_01(Sigmoid),
        DEFINE_TEST_01(Square),
        DEFINE_TEST_01(Sin),
        DEFINE_TEST_01(Cos),
        DEFINE_TEST_01(Tan),
        DEFINE_TEST_01(Sinh),
        DEFINE_TEST_01(Cosh),
        DEFINE_TEST_01(Tanh),
        DEFINE_TEST_01(Asinh),
        DEFINE_TEST_01(Acosh),
        DEFINE_TEST_01(Atanh),
    };

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

    int ntests = sizeof(tests) / sizeof(Test);
    for (size_t i = 0; i < ntests; ++i) {
      test_vec_.push_back({tests[i].name, tests[i].func});
    }

    int ok = 0;
    int run = 0;
    for (Test& test : test_vec_) {
      if (name == nullptr || test.name == name) {
        ++run;
        bool flag = test.func(param);
        fprintf(stderr, "%-30s %s\n", test.name.c_str(), flag ? "OK" : "NG");
        if (flag)
            ++ok;
      }
    }
    fprintf(stderr, "%d tests failed\n", run - ok);
    return 0;
}
