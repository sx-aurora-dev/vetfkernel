#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <vml.h>
#include <types.h>

struct BinaryOpArgs {
    vml::Tensor in0;
    vml::Tensor in1;
    vml::Tensor out;
} __attribute__((__packed__));

template<typename T> struct dypte_s {};
template<> struct dypte_s<float> { static const int type = 1; };
template<> struct dypte_s<bool>  { static const int type = 10; };

template <typename T>
vml::Tensor makeTensor(size_t dims, std::vector<size_t> const& dim_size)
{
    vml::Tensor t;

    t.dtype = dypte_s<T>::type;
    t.dims = dims;
    t.nelems = 1;
    for (int i = 0; i < dims; ++i) {
        t.dim_size[i] = dim_size[i];
        t.nelems *= dim_size[i];
    }

    t.addr = reinterpret_cast<uint64_t>(new T[t.nelems]);

    return t;
}

template<typename T>
class Tensor {
    public:
        Tensor(std::vector<size_t> const& shape) {
          shape_ = shape;
          t = makeTensor<T>(shape.size(), shape);
          stride_.resize(shape.size());
          size_t dim = t.dims;
          stride_[dim - 1] = 1;
          for (int i = dim - 2; i >= 0; --i) {
            stride_[i] = stride_[i + 1] * t.dim_size[i + 1];
          }
        }
        ~Tensor() { delete[] reinterpret_cast<T*>(t.addr); }
        std::vector<size_t> const& shape() const { return shape_; }
        T* data() { return reinterpret_cast<T*>(t.addr); }
        T const* data() const { return reinterpret_cast<T const*>(t.addr); }
        size_t nelems() const { return t.nelems; }
        size_t dims() const { return t.dims; }
        size_t dim_size(size_t i) const { return t.dim_size[i]; }
        size_t stride(size_t i) const { return stride_[i]; }

        vml::Tensor tensor() const { return t; }

    private:
        vml::Tensor t;
        std::vector<size_t> stride_;
        std::vector<size_t> shape_;
};

template<typename T>
bool checkTensor(Tensor<T> const& a, Tensor<T> const& b)
{
    if (a.nelems() != b.nelems())
        return false;

    for (size_t i = 0; i < a.nelems(); ++i) {
#if 0
        if (a.data()[i] != b.data()[i])
            return false;
#else
        T ai = a.data()[i];
        T bi = b.data()[i];
        double err = ai - bi;
        if (err * err / (ai * bi) > 1e-8)
          return false;
#endif
    }
    return true;
}

template<typename T>
void printTensor(Tensor<T> const& t)
{
    std::string fmt = " %8.3f";
    if (typeid(T) == typeid(bool)) {
      fmt = std::string(" %d");
    }

    std::vector<size_t> s(t.dims() + 1);
    s[t.dims()] = 1;
    for (int i = t.dims() - 1; i >= 0; --i)
        s[i] = s[i + 1] * t.dim_size(i);

#if 0
    fprintf(stderr, "%d %d %d\n", t.dim_size(0), t.dim_size(1), t.dim_size(2));
    fprintf(stderr, "%d %d %d\n", s[0], s[1], s[2]);
#endif

    T const* p = t.data();
    size_t n = t.dim_size(t.dims() - 1); // innermost

    for (size_t i = 0; i < t.nelems(); ++i) {
        if (i % n == 0) {
            for (int j = 0; j < t.dims(); ++j) {
                fprintf(stderr, "%c", i % s[j] == 0 ? '[' : ' ');
            }
        }
        fprintf(stderr, fmt.c_str(), p[i]);
        if ((i + 1) % n == 0) {
            fprintf(stderr, " ");
            for (int j = 0; j < t.dims(); ++j) {
                if ((i + 1) % s[j] == 0) 
                    fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
        }
    }
}

struct TestParam
{
    int verbose;
};

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
DEFINE_TEST_UNARY_OP_01(Exp, vml::exp, expf);
DEFINE_TEST_UNARY_OP_01(Floor, vml::floor, std::floor);
DEFINE_TEST_UNARY_OP_01(Neg, vml::neg, [](float x) { return -x; });
DEFINE_TEST_UNARY_OP_01(Log, vml::log, std::log);
DEFINE_TEST_UNARY_OP_01(Reciprocal, vml::reciprocal, [](float x) { return 1/x; });
DEFINE_TEST_UNARY_OP_01(Rsqrt, vml::rsqrt, [](float x) { return 1/std::sqrt(x); });
DEFINE_TEST_UNARY_OP_01(Sigmoid, vml::sigmoid,
                        [](float x) { return 1/(1+std::exp(-x)); });
DEFINE_TEST_UNARY_OP_01(Sqrt, vml::sqrt, std::sqrt);
DEFINE_TEST_UNARY_OP_01(Square, vml::square, [](float x) { return x * x; });
DEFINE_TEST_UNARY_OP_01(Tanh, vml::tanh, std::tanh);

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
    BinaryOpArgs args;
    args.out = out.tensor();
    args.in0 = in0.tensor();
    args.in1 = in1.tensor();
    int ret = op(args.out, args.in0, args.in1);

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



// obsolute api
extern "C" {
  int op_minimum_(const BinaryOpArgs& args);
  int op_maximum_(const BinaryOpArgs& args);
  int op_equal_(const BinaryOpArgs& args);
  int op_notEqual_(const BinaryOpArgs& args);
  int op_less_(const BinaryOpArgs& args);
  int op_lessEqual_(const BinaryOpArgs& args);
  int op_greater_(const BinaryOpArgs& args);
  int op_greaterEqual_(const BinaryOpArgs& args);
}

template<typename TOUT, typename T>
bool test_BinaryOp_(TestParam const& param,
                   Tensor<TOUT>& out,
                   Tensor<T> const& in0,
                   Tensor<T> const& in1,
                   Tensor<TOUT> const& exp,
                   int (*op)(const BinaryOpArgs& args))
{
    BinaryOpArgs args;
    args.out = out.tensor();
    args.in0 = in0.tensor();
    args.in1 = in1.tensor();
    int ret = op(args);

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

template <typename T>
int ref_Add(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z)
{
  return ref_Binop(X, Y, Z, [](T y, T z) -> T { return y + z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename T>
int ref_Sub(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z)
{
  return ref_Binop(X, Y, Z, [](T y, T z) -> T { return y - z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename T>
int ref_Mul(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z)
{
  return ref_Binop(X, Y, Z, [](T y, T z) -> T { return y * z; },
          X.data(), Y.data(), Z.data(), 0);
}

template <typename T>
int ref_SquaredDifference(Tensor<T>& X, Tensor<T> const& Y, Tensor<T> const& Z)
{
  return ref_Binop(X, Y, Z, [](T y, T z) -> T { return (y - z) * (y - z); },
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
  return test_BinaryOp_04<float>(param, ref_Add<float>, vml::add);
}

bool test_Add_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_Add<float>, vml::add);
}

bool test_Add_06(TestParam const& param)
{
  return test_BinaryOp_06<float>(param, ref_Add<float>, vml::add);
}

bool test_Add_generic(TestParam const& param)
{
  Tensor<float> exp({4, 8, 4});
  Tensor<float> out({4, 8, 4});
  Tensor<float> in0({1, 8, 2});
  Tensor<float> in1({4, 4, 4});

  for (size_t i = 0; i < 1; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      for (size_t k = 0; k < 2; ++k) {
	size_t index = (i * 8 + j) * 2 + k;
	in0.data()[index] = (float)drand48();
      }
    }
  }

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      for (size_t k = 0; k < 4; ++k) {
	size_t index = (i * 4 + j) * 4 + k;
	in1.data()[index] = (float)drand48();
      }
    }
  }

  ref_Add(exp, in0, in1);

  return test_BinaryOp(param, out, in0, in1, exp, vml::add);
}




bool test_Sub_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Sub<float>, vml::sub);
}

bool test_Sub_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_Sub<float>, vml::sub);
}

bool test_Mul_04(TestParam const& param)
{
  return test_BinaryOp_04<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_06(TestParam const& param)
{
  return test_BinaryOp_06<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_07(TestParam const& param)
{
  return test_BinaryOp_07<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_08(TestParam const& param)
{
  return test_BinaryOp_08<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_09(TestParam const& param)
{
  return test_BinaryOp_09<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_10(TestParam const& param)
{
  return test_BinaryOp_10<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_11(TestParam const& param)
{
  return test_BinaryOp_11<float>(param, ref_Mul<float>, vml::mul);
}

bool test_Mul_12(TestParam const& param)
{
  return test_BinaryOp_12<float>(param, ref_Mul<float>, vml::mul);
}

bool test_SquaredDifference_05(TestParam const& param)
{
  return test_BinaryOp_05<float>(param, ref_SquaredDifference<float>, 
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


template<typename TOUT, typename TIN>
bool test_generic_(TestParam const& param,
		   int (*ref_op)(Tensor<TOUT>& out, Tensor<TIN> const& in0, Tensor<TIN> const& in1),
		   int (*op)(const BinaryOpArgs& args))
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

  return test_BinaryOp_(param, out, in0, in1, exp, op);
}

bool test_Maximum_generic(TestParam const& param)
{
  return test_generic_<float, float>(param, ref_Maximum, op_maximum_);
}

bool test_Minimum_generic(TestParam const& param)
{
  return test_generic_<float, float>(param, ref_Minimum, op_minimum_);
}

bool test_Equal_generic(TestParam const& param)
{
  return test_generic_<bool, float>(param, ref_Equal, op_equal_);
}

bool test_NotEqual_generic(TestParam const& param)
{
  return test_generic_<bool, float>(param, ref_NotEqual, op_notEqual_);
}

bool test_Less_generic(TestParam const& param)
{
  return test_generic_<bool, float>(param, ref_Less, op_less_);
}

bool test_LessEqual_generic(TestParam const& param)
{
  return test_generic_<bool, float>(param, ref_LessEqual, op_lessEqual_);
}

bool test_Greater_generic(TestParam const& param)
{
  return test_generic_<bool, float>(param, ref_Greater, op_greater_);
}

bool test_GreaterEqual_generic(TestParam const& param)
{
  return test_generic_<bool, float>(param, ref_GreaterEqual, op_greaterEqual_);
}



struct Test
{
    std::string name;
    bool (*func)(TestParam const&);
};

int main(int argc, char* argv[])
{
    Test tests[] = {
        { "Add_01", test_Add_01 },
        { "Add_02", test_Add_02 },
        { "Add_03", test_Add_03 },
        { "Add_04", test_Add_04 },
        { "Add_05", test_Add_05 },
        { "Add_06", test_Add_06 },
	// { "Add[generic]", test_Add_generic },

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

        { "Minimum[generic]",      test_Minimum_generic      },
        { "Maximum[generic]",      test_Maximum_generic      },
        { "Equal[generic]",        test_Equal_generic        },
        { "NotEqual[generic]",     test_NotEqual_generic     },
        { "Less[generic]",         test_Less_generic         },
        { "LessEqual[generic]",    test_LessEqual_generic    },
        { "Greater[generic]",      test_Greater_generic      },
        { "GreaterEqual[generic]", test_GreaterEqual_generic },

#define DEFINE_TEST_01(T) {#T "_01", test_##T##_01}
        DEFINE_TEST_01(Abs),
        DEFINE_TEST_01(Exp),
        DEFINE_TEST_01(Floor),
        DEFINE_TEST_01(Neg),
        DEFINE_TEST_01(Log),
        DEFINE_TEST_01(Reciprocal),
        DEFINE_TEST_01(Rsqrt),
        DEFINE_TEST_01(Sigmoid),
        DEFINE_TEST_01(Square),
        DEFINE_TEST_01(Tanh),
    };

    TestParam param;
    param.verbose = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) {
            ++param.verbose;
        }
    }

    int ntests = sizeof(tests) / sizeof(Test);
    int ok = 0;
    for (size_t i = 0; i < ntests; ++i) {
        bool flag = tests[i].func(param);
        fprintf(stderr, "%-30s %s\n", tests[i].name.c_str(), flag ? "OK" : "NG");
        if (flag)
            ++ok;
    }
    fprintf(stderr, "%d tests failed\n", ntests - ok);
    return 0;
}
