#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <omp.h>
#include "vml/log.h"
#include "kernel.h"
#include "vml.h"

// ----------------------------------------------------------------------

#define REGISTER_UNARY_OP(NAME, FUNC) \
  REGISTER_KERNEL(#NAME, "__op_" # NAME); \
  extern "C" { \
    int __op_##NAME(const void* args, size_t len) { \
      return unary_op_helper(args, len, FUNC, "op_" # NAME); \
    } \
  }


namespace {

// valid: in.dtype, in.nelems, in.addr, out.addr
struct UnaryOpArgs
{
  vml::TensorDesc<8> in;
  vml::TensorDesc<8> out;
} __attribute__((__packed__));

int unary_op_helper(const void* args, size_t len,
                    int (*func)(vml::Tensor const& out, vml::Tensor const& in),
                    char const* name)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": begin";
  int ret = 1;

  if (sizeof(UnaryOpArgs) == len) {
    const UnaryOpArgs* p = reinterpret_cast<const UnaryOpArgs*>(args);

    LOG(LOG_PARAM) << __FUNCTION__ << "::" << name << ":"
        << " dtype=" << p->in.dtype << " nelems=" << p->in.nelems;

    if (func) {
      ret = func(*reinterpret_cast<vml::Tensor const*>(&p->out),
                 *reinterpret_cast<vml::Tensor const*>(&p->in));
    }
  } else {
    LOG(LOG_ERROR) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(UnaryOpArgs) << " bytes expected";
  }

  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": end. ret=" << ret;
  return ret;
}

} // namespace

REGISTER_UNARY_OP(Abs, vml::abs);
REGISTER_UNARY_OP(Sign, vml::sign);
REGISTER_UNARY_OP(Exp, vml::exp);
REGISTER_UNARY_OP(Expm1, vml::expm1);
REGISTER_UNARY_OP(Floor, vml::floor);
REGISTER_UNARY_OP(Neg, vml::neg);
REGISTER_UNARY_OP(Log, vml::log);
REGISTER_UNARY_OP(Log1p, vml::log1p);
REGISTER_UNARY_OP(Reciprocal, vml::reciprocal);
REGISTER_UNARY_OP(Rsqrt, vml::rsqrt);
REGISTER_UNARY_OP(Sigmoid, vml::sigmoid);
REGISTER_UNARY_OP(Sqrt, vml::sqrt);
REGISTER_UNARY_OP(Square, vml::square);
REGISTER_UNARY_OP(Sin, vml::sin);
REGISTER_UNARY_OP(Cos, vml::cos);
REGISTER_UNARY_OP(Tan, vml::tan);
REGISTER_UNARY_OP(Sinh, vml::sinh);
REGISTER_UNARY_OP(Cosh, vml::cosh);
REGISTER_UNARY_OP(Tanh, vml::tanh);
REGISTER_UNARY_OP(Asinh, vml::asinh);
REGISTER_UNARY_OP(Acosh, vml::acosh);
REGISTER_UNARY_OP(Atanh, vml::atanh);
REGISTER_UNARY_OP(Erf, vml::erf);

