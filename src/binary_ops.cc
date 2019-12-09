#include <algorithm>
#include <cstdint>
#include <cassert>
#include "kernel.h"
#include "log.h"
#include <sstream>
#include <vector>

#include "vml.h"

//#define TIMER
#ifdef TIMER
#include "timer.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

//#define DEBUG

namespace {

struct BinaryOpArgs {
  vml::Tensor in0;
  vml::Tensor in1;
  vml::Tensor out;
} __attribute__((__packed__));


// X = Y op Z
int op_Binary(const void* args, size_t len, 
              int (*func)(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z),
              const char* name)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": begin";
  int ret = 1;

  if (sizeof(BinaryOpArgs) == len) {
    const BinaryOpArgs* p = reinterpret_cast<const BinaryOpArgs*>(args);

    LOG(LOG_PARAM) << __FUNCTION__ << "::" << name << ":"
      << " in0=" << p->in0
      << " in1=" << p->in1
      << " out=" << p->out;

    if (func) {
#ifdef TIMER
      double t0 = second();
#endif
      ret = func(p->out, p->in0, p->in1);
#ifdef TIMER
      double ms = (second() - t0) * 1e3;
      LOG(LOG_TIMER) << __FUNCTION__ << "::" << name << ": " << ms << " msec";
#endif
    }
  } else {
    LOG(LOG_ERROR) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(BinaryOpArgs) << " bytes expected";
  }

  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": end. ret=" << ret;
  return ret;
}

} // namespace


#define DEFINE_BINARY_OP(name, FUNC) \
  extern "C" { \
  int _op_##name(const void* args, size_t len) \
{ \
  return op_Binary(args, len, FUNC, #name); \
} \
} \
REGISTER_KERNEL(#name, "_op_"#name);

DEFINE_BINARY_OP(Add, vml::add);
DEFINE_BINARY_OP(Div, vml::div);
DEFINE_BINARY_OP(Mul, vml::mul);
DEFINE_BINARY_OP(Sub, vml::sub);
DEFINE_BINARY_OP(DivNoNan, vml::divnonan);
DEFINE_BINARY_OP(Pow, vml::pow);
DEFINE_BINARY_OP(SquaredDifference, vml::sqdiff);
DEFINE_BINARY_OP(RsqrtGrad, vml::rsqrt_grad);
DEFINE_BINARY_OP(Minimum, vml::minimum);
DEFINE_BINARY_OP(Maximum, vml::maximum);
DEFINE_BINARY_OP(Equal, vml::equal);
DEFINE_BINARY_OP(NotEqual, vml::notEqual);
DEFINE_BINARY_OP(Less, vml::less);
DEFINE_BINARY_OP(LessEqual, vml::lessEqual);
DEFINE_BINARY_OP(Greater, vml::greater);
DEFINE_BINARY_OP(GreaterEqual, vml::greaterEqual);
