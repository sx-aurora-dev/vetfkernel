#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <omp.h>
#include "log.h"
#include "kernel.h"
#include "types.h"
#include "vml.h"

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/libvetfkernel.h"
#endif

#define DEFINE_UNARY_OP(NAME, FUNC) \
int vml::NAME(vml::Tensor const& out, vml::Tensor const& in) { \
  return unary_op_wrapper(out, in, FUNC<float, float>); \
}

namespace {

inline int unary_op_wrapper(vml::Tensor const& out, vml::Tensor const& in,
                            int (*func_f32_f32)(float*, float const*, size_t))
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";
  LOG(LOG_PARAM) << __FUNCTION__ << ": in.dtype=" << in.dtype
    << " out.dtype=" << out.dtype;

  int ret = 1;
  if (in.dtype == DT_FLOAT && out.dtype == DT_FLOAT) {
    float* po = reinterpret_cast<float*>(out.addr);
    float const* pi = reinterpret_cast<float const*>(in.addr);
    if( in.nelems >= 2048 ) {
#pragma omp parallel
      {
        int64_t nthreads = omp_get_num_threads() ;
        int64_t threadid = omp_get_thread_num() ;

        int64_t chunkSize = in.nelems / nthreads ;
        int64_t remain    = in.nelems % nthreads ;

        int64_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
        int64_t myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

        int64_t offset    = chunkBegin ;

        if( myChunk > 0 ) {
          ret = func_f32_f32(po + offset, pi + offset, myChunk);
        }
      }
    }
    else {
      ret = func_f32_f32(po, pi, in.nelems);
    }
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

} // namespace

//
// Abs
//

int vml::abs(vml::Tensor const& out, vml::Tensor const& in)
{
  if (in.dtype == DT_FLOAT) {
    float* po = out.ptr<float*>();
    float const* pi = in.ptr<float const*>();
    for (size_t i = 0; i < in.nelems; ++i)
      po[i] = fabsf(pi[i]);
    return 0;
  } else if (in.dtype == DT_DOUBLE) {
    double* po = out.ptr<double*>();
    double const* pi = in.ptr<double const*>();
    for (size_t i = 0; i < in.nelems; ++i)
      po[i] = fabs(pi[i]);
    return 0;
#if 0 // do int32 type's abs in CPU.
  } else if (in.dtype == DT_INT32) {
    int32_t* po = out.ptr<int32_t*>();
    int32_t const* pi = in.ptr<int32_t const*>();
    for (size_t i = 0; i < in.nelems; ++i)
      po[i] = ::abs(pi[i]);
    return 0;
#endif
  } else if (in.dtype == DT_INT64) {
    int64_t* po = out.ptr<int64_t*>();
    int64_t const* pi = in.ptr<int64_t const*>();
    for (size_t i = 0; i < in.nelems; ++i)
      po[i] = labs(pi[i]);
    return 0;
  }

  return 1;
}

//
// Exp
//

template<typename Tout, typename Tin>
int op_exp(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::exp(pi[i]) ;
  }

  return 0;
}

DEFINE_UNARY_OP(exp, op_exp);

//
// Floor
//

template<typename Tout, typename Tin>
int op_floor(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::floor(pi[i]);
  }

  return 0;
}

DEFINE_UNARY_OP(floor, op_floor);

//
// Log
//

template<typename Tout, typename Tin>
int op_log(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::log(pi[i]) ;
  }

  return 0;
}

DEFINE_UNARY_OP(log, op_log);

//
// Neg
//

template<typename Tout, typename Tin>
inline int neg(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = - pi[i];
  }

  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<>
inline int neg<float, float>(float* out, float const* in, size_t nelems)
{
  return neg_f32(out, in, nelems);
}
#endif

DEFINE_UNARY_OP(neg, ::neg);

//
// Reciprocal
//

template<typename Tout, typename Tin>
int op_reciprocal(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = Tin(1) / pi[i] ;
  }

  return 0;
}

DEFINE_UNARY_OP(reciprocal, op_reciprocal);

//
// Rsqrt
//

template<typename Tout, typename Tin>
inline int rsqrt(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = Tin(1) / std::sqrt(pi[i]);
  }
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<> 
inline int rsqrt<float, float>(float* po, float const* pi, size_t n)
{
  return rsqrt_f32(po, pi, n);
}
#endif

DEFINE_UNARY_OP(rsqrt, ::rsqrt);

//
// Sigmoid
//

template<typename Tout, typename Tin>
int op_sigmoid(Tout* po, Tin const* pi, size_t nelems)
{
#if 0 // original
  for (int64_t i = 0; i < nelems; ++i) {
    const Tout One = Tout(1.) ;
    po[i] = One / (One + std::exp(-pi[i])) ;
  }
#else
  // ncc's vectorized-exp causes nan.

  for (int64_t i = 0; i < nelems; ++i) {
    const Tout One = Tout(1.) ;
    const Tin  vi  = pi[i] ;

    po[i] = vi < Tin(-88.) ? Tout(0.) : One / (One + std::exp(-vi)) ;
  }
#endif
  return 0;
}

DEFINE_UNARY_OP(sigmoid, op_sigmoid);

//
// Sqrt
//

template<typename Tout, typename Tin>
int sqrt_(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::sqrt(pi[i]);
  }
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<>
int sqrt_<float, float>(float* out, float const* in, size_t nelems)
{
  return sqrt_f32(out, in, nelems);
}
#endif

DEFINE_UNARY_OP(sqrt, sqrt_);

//
// Square
//

template<typename Tout, typename Tin>
int square(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = pi[i] * pi[i];
  }
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<>
int square<float, float>(float* out, float const* in, size_t nelems)
{
  return square_f32(out, in, nelems);
}
#endif

DEFINE_UNARY_OP(square, ::square);

//
// Tanh
// 

template<typename Tout, typename Tin>
int op_tanh(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::tanh(pi[i]) ;
  }
  return 0;
}

DEFINE_UNARY_OP(tanh, op_tanh);

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
  vml::Tensor in;
  vml::Tensor out;
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
      ret = func(p->out, p->in);
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
REGISTER_UNARY_OP(Exp, vml::exp);
REGISTER_UNARY_OP(Floor, vml::floor);
REGISTER_UNARY_OP(Neg, vml::neg);
REGISTER_UNARY_OP(Log, vml::log);
REGISTER_UNARY_OP(Reciprocal, vml::reciprocal);
REGISTER_UNARY_OP(Rsqrt, vml::rsqrt);
REGISTER_UNARY_OP(Sigmoid, vml::sigmoid);
REGISTER_UNARY_OP(Sqrt, vml::sqrt);
REGISTER_UNARY_OP(Square, vml::square);
REGISTER_UNARY_OP(Tanh, vml::tanh);

