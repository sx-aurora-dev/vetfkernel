#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <omp.h>
#include "log.h"
#include "types.h"
#include "vml.h"

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/intrinsic.h"
#endif

#define DEFINE_UNARY_OP(NAME, FUNC) \
int vml::NAME(vml::Tensor const& out, vml::Tensor const& in) { \
  return unary_op_wrapper(out, in, FUNC<float, float>, \
                          FUNC<double, double>, \
                          FUNC<int64_t, int64_t>); \
}
#if 0
#define DEFINE_UNARY_OP3(NAME, FUNC1, FUNC2, FUNC3) \
int vml::NAME(vml::Tensor const& out, vml::Tensor const& in) { \
  return unary_op_wrapper(out, in, FUNC1, FUNC2, FUNC3); \
}
#endif

namespace {

inline int unary_op_wrapper(vml::Tensor const& out, vml::Tensor const& in,
                            int (*func_f32_f32)(float*, float const*, size_t),
                            int (*func_f64_f64)(double*, double const*, size_t),
                            int (*func_i64_i64)(int64_t*, int64_t const*, size_t))
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
  } else if (in.dtype == DT_DOUBLE && out.dtype == DT_DOUBLE) {
    double* po = reinterpret_cast<double*>(out.addr);
    double const* pi = reinterpret_cast<double const*>(in.addr);
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
          ret = func_f64_f64(po + offset, pi + offset, myChunk);
        }
      }
    }
    else {
      ret = func_f64_f64(po, pi, in.nelems);
    }
  } else if (in.dtype == DT_INT64 && out.dtype == DT_INT64) {
    int64_t* po = reinterpret_cast<int64_t*>(out.addr);
    int64_t const* pi = reinterpret_cast<int64_t const*>(in.addr);
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
          ret = func_i64_i64(po + offset, pi + offset, myChunk);
        }
      }
    }
    else {
      ret = func_i64_i64(po, pi, in.nelems);
    }
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

} // namespace

//
// Abs
//

template<typename Tout, typename Tin>
int op_abs(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::abs(pi[i]) ;
  }

  return 0;
}

/// Abs. 
DEFINE_UNARY_OP(abs, op_abs);

//
// Sign
//

template<typename Tout, typename Tin>
int op_sign(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = (Tout)((pi[i] > (Tin)0) - (pi[i] < (Tin)0));
  }

  return 0;
}

/// Sign
DEFINE_UNARY_OP(sign, op_sign);

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

/// Exp
DEFINE_UNARY_OP(exp, op_exp);

//
// Expm1
//

template<typename Tout, typename Tin>
int op_expm1(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::exp(pi[i]) - 1.0;
  }

  return 0;
}

/// `X = exp(Y) - 1.0`
DEFINE_UNARY_OP(expm1, op_expm1);

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

/// Floor
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

/// Log
DEFINE_UNARY_OP(log, op_log);

//
// Log1p
//

template<typename Tout, typename Tin>
int op_log1p(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::log1p(pi[i]) ;
  }

  return 0;
}

/// Log1p
DEFINE_UNARY_OP(log1p, op_log1p);

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

/// Neg
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

/// Reciprocal.
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

/// Rsqrt.
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

/// Sigmoid.
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

/// Sqrt
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

/// Square
DEFINE_UNARY_OP(square, ::square);

//
// Sin
// 

template<typename Tout, typename Tin>
int op_sin(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::sin(pi[i]) ;
  }
  return 0;
}

/// Sin
DEFINE_UNARY_OP(sin, op_sin);

//
// Cos
// 

template<typename Tout, typename Tin>
int op_cos(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::cos(pi[i]) ;
  }
  return 0;
}

/// Cos
DEFINE_UNARY_OP(cos, op_cos);

//
// Tan
// 

template<typename Tout, typename Tin>
int op_tan(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::tan(pi[i]) ;
  }
  return 0;
}

/// Tan
DEFINE_UNARY_OP(tan, op_tan);

//
// Sinh
// 

template<typename Tout, typename Tin>
int op_sinh(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::sinh(pi[i]) ;
  }
  return 0;
}

/// Sinh
DEFINE_UNARY_OP(sinh, op_sinh);

//
// Cosh
// 

template<typename Tout, typename Tin>
int op_cosh(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::cosh(pi[i]) ;
  }
  return 0;
}

/// Cosh
DEFINE_UNARY_OP(cosh, op_cosh);

//
// Tanh
// 

template<typename Tout, typename Tin>
int op_tanh(Tout* po, Tin const* pi, size_t nelems)
{
  /* B/F is small, so use packed_vector  */
#pragma _NEC packed_vector
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::tanh(pi[i]) ;
  }
  return 0;
}

/// Tanh
DEFINE_UNARY_OP(tanh, op_tanh);

//
// Asinh
// 

template<typename Tout, typename Tin>
int op_asinh(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::asinh(pi[i]) ;
  }
  return 0;
}

/// Asinh
DEFINE_UNARY_OP(asinh, op_asinh);

//
// Acosh
// 

template<typename Tout, typename Tin>
int op_acosh(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::acosh(pi[i]) ;
  }
  return 0;
}

/// Acosh
DEFINE_UNARY_OP(acosh, op_acosh);

//
// Atanh
// 

template<typename Tout, typename Tin>
int op_atanh(Tout* po, Tin const* pi, size_t nelems)
{
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::atanh(pi[i]) ;
  }
  return 0;
}

/// Atanh
DEFINE_UNARY_OP(atanh, op_atanh);


//
// Erf
//

template<typename Tout, typename Tin>
int op_erf(Tout* po, Tin const* pi, size_t nelems)
{
  /* B/F is small, so use packed_vector  */
#pragma _NEC packed_vector
  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::erf(pi[i]) ;
  }
  return 0;
}

/// Erf
DEFINE_UNARY_OP(erf, op_erf);
