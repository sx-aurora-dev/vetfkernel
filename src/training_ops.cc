#include "types.h"
#include "log.h"
#include "ve_ops_common.h"
#include "vml.h"

#include <omp.h>

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/intrinsic.h"
#endif


//
// ApplyGradientDescent
//

namespace {

template <typename T>
void apply_gradient_descent(
    const vml::Tensor* var_tensor,
    const vml::Tensor* delta_tensor,
    const T alpha
)
{
    const size_t nelems = var_tensor->nelems ;

    T* pVar          = reinterpret_cast<T*>(var_tensor->addr) ;
    T* pDelta        = reinterpret_cast<T*>(delta_tensor->addr) ;

    for (size_t i = 0; i < nelems; ++i) {
      pVar[i] -= alpha * pDelta[i] ;
    }
}
}

int vml::ApplyGradientDescent(
    vml::Tensor const& var,
    vml::Tensor const& alpha,	// scalar
    vml::Tensor const& delta
)
{
  if( alpha.nelems != 1 )
      return 1;

  if( var.nelems != delta.nelems )
      return 1 ;

  if( var.dtype != alpha.dtype ||  var.dtype != delta.dtype )
      return 1 ;

  if (var.dtype == DT_FLOAT ) {
    apply_gradient_descent<float>(
	  &var,
	  &delta,
	  reinterpret_cast<const float*>(alpha.addr)[0]
    );
  } else {
      return 1;
  }

  return 0;
}

namespace {
int op_ApplyGradientDescent(const VEOpArgs& args)
{
    if (args.nArguments() != 3)
        return 1;

    const vml::Tensor* var   = args.arg<vml::Tensor>(0) ;
    const vml::Tensor* alpha = args.arg<vml::Tensor>(1) ; // scalar
    const vml::Tensor* delta = args.arg<vml::Tensor>(2) ;

    if (!var || !alpha || !delta)
        return 1;

    LOG(LOG_PARAM)
        << __FUNCTION__  << ":"
	<< " var="     << var
	<< " alpha="   << alpha
	<< " delta="   << delta ;

    return vml::ApplyGradientDescent(*var, *alpha, *delta) ;
}

} // namespace

DEFINE_KERNEL(ApplyGradientDescent, op_ApplyGradientDescent);


//
// ApplyAdaDelta
//

namespace {

template <typename T>
void apply_adadelta(
    const vml::Tensor* var_tensor,
    const vml::Tensor* accum_tensor,
    const vml::Tensor* accum_update_tensor,
    const vml::Tensor* grad_tensor,
    const T lr,
    const T rho,
    const T epsilon
)
{
#if 0	// Eigen Impl
  accum.device(d) =
      accum * rho() + grad.square() * (static_cast<T>(1) - rho());
  const auto update =
      (accum_update + epsilon()).sqrt() * (accum + epsilon()).rsqrt() * grad;
  var.device(d) -= update * lr();
  accum_update.device(d) =
      accum_update * rho() + update.square() * (static_cast<T>(1) - rho());
#endif

    const size_t nelems = var_tensor->nelems ;

    T* pVar          = reinterpret_cast<T*>(var_tensor->addr) ;
    T* pAccum        = reinterpret_cast<T*>(accum_tensor->addr) ;
    T* pAccumUpdate  = reinterpret_cast<T*>(accum_update_tensor->addr) ;
    const T* pGrad   = reinterpret_cast<T*>(grad_tensor->addr) ;

    for (size_t i = 0; i < nelems; ++i) {
      const T grad   = pGrad[i] ;
      T accum        = pAccum[i] ;
      T accum_update = pAccumUpdate[i] ;
      T var          = pVar[i] ;


      accum =  accum * rho + grad * grad * (T(1.)-rho);

      const T update = std::sqrt(accum_update + epsilon) / std::sqrt(accum + epsilon) * grad ;
      accum_update = accum_update * rho + update * update * (T(1.)-rho);
      var-= update * lr ;

      pAccum[i] = accum ;
      pAccumUpdate[i] = accum_update ;
      pVar[i] = var ;
    }
}
}

int vml::ApplyAdadelta(
    vml::Tensor const& var,
    vml::Tensor const& accum,
    vml::Tensor const& accum_update,
    vml::Tensor const& lr,		// scalar
    vml::Tensor const& rho,		// scalar
    vml::Tensor const& epsilon,		// scalar
    vml::Tensor const& grad
)
{
  if( lr.nelems != 1 || rho.nelems != 1 || epsilon.nelems != 1 )
      return 1;

  if( var.nelems != accum.nelems
	  || var.nelems != accum_update.nelems
	  || var.nelems != grad.nelems )
      return 1 ;

  if( var.dtype != accum.dtype
	  ||  var.dtype != accum_update.dtype
	  ||  var.dtype != lr.dtype
	  ||  var.dtype != rho.dtype
	  ||  var.dtype != epsilon.dtype
	  ||  var.dtype != grad.dtype )
      return 1 ;

  if (var.dtype == DT_FLOAT ) {
    apply_adadelta<float>(
	  &var,
	  &accum,
	  &accum_update,
	  &grad,
	  reinterpret_cast<const float*>(lr.addr)[0],
	  reinterpret_cast<const float*>(rho.addr)[0],
	  reinterpret_cast<const float*>(epsilon.addr)[0]
    );
  } else {
      return 1;
  }

  return 0;
}

namespace {
int op_ApplyAdadelta(const VEOpArgs& args)
{
    if (args.nArguments() != 7)
        return 1;

    const vml::Tensor* var          = args.arg<vml::Tensor>(0) ;
    const vml::Tensor* accum        = args.arg<vml::Tensor>(1) ;
    const vml::Tensor* accum_update = args.arg<vml::Tensor>(2) ;
    const vml::Tensor* lr           = args.arg<vml::Tensor>(3) ; // scalar
    const vml::Tensor* rho          = args.arg<vml::Tensor>(4) ; // scalar
    const vml::Tensor* epsilon      = args.arg<vml::Tensor>(5) ; // scalar
    const vml::Tensor* grad         = args.arg<vml::Tensor>(6) ;


    if (!var || !accum || !accum_update || !lr || !rho || !epsilon || !grad)
        return 1;

    LOG(LOG_PARAM)
        << __FUNCTION__  << ":"
	<< " var="     << var
	<< " accum="   << accum
	<< " accum_update=" << accum_update
	<< " lr="      << lr
	<< " rho="     << rho
	<< " epsilon=" << epsilon
	<< " grad="    << grad ;

    return vml::ApplyAdadelta(
	*var, *accum, *accum_update,
	*lr, *rho, *epsilon,
	*grad ) ;
}

} // namespace

DEFINE_KERNEL(ApplyAdadelta, op_ApplyAdadelta);


//
// ApplyMomentum
//

namespace {

template <typename T>
void apply_momentum(
    const vml::Tensor* var_tensor,
    const vml::Tensor* accum_tensor,
    const vml::Tensor* grad_tensor,
    const T lr,
    const T momentum,
    const bool use_nesterov
)
{
    const size_t nelems = var_tensor->nelems ;

    T* pVar          = reinterpret_cast<T*>(var_tensor->addr) ;
    T* pAccum        = reinterpret_cast<T*>(accum_tensor->addr) ;
    const T* pGrad   = reinterpret_cast<T*>(grad_tensor->addr) ;

    const T one = T(1.) ;

    for(int64_t i=0; i<nelems; i++) {
      pAccum[i] = pAccum[i] * momentum + pGrad[i] ;
    }
    if (use_nesterov) {
      for(int64_t i=0; i<nelems; i++) {
	pVar[i] -= pGrad[i] * lr + pAccum[i] * momentum * lr ;
      }
    } else {
      for(int64_t i=0; i<nelems; i++) {
	pVar[i] -= pAccum[i] * lr ;
      }
    }
}
}


int vml::applyMomentum(
    vml::Tensor const& var,
    vml::Tensor const& accum,
    vml::Tensor const& lr,		// scalar
    vml::Tensor const& grad,
    vml::Tensor const& momentum,	// scalar
    const bool use_nesterov
)
{

  if( lr.nelems != 1 || momentum.nelems != 1 )
      return 1;

  if( var.nelems != accum.nelems
	  || var.nelems != grad.nelems )
      return 1 ;

  if( var.dtype != accum.dtype
	  ||  var.dtype != lr.dtype
	  ||  var.dtype != grad.dtype
	  ||  var.dtype != momentum.dtype )
      return 1 ;

  if (var.dtype == DT_FLOAT ) {
    apply_momentum<float>(
	  &var,
	  &accum,
	  &grad,
	  reinterpret_cast<const float*>(lr.addr)[0],
	  reinterpret_cast<const float*>(momentum.addr)[0],
	  use_nesterov
    );
  } else {
      return 1;
  }

  return 0;

}

namespace {
int op_ApplyMomentum(const VEOpArgs& args)
{
    if (args.nArguments() != 6)
        return 1;

    const vml::Tensor* var          = args.arg<vml::Tensor>(0) ;
    const vml::Tensor* accum        = args.arg<vml::Tensor>(1) ;
    const vml::Tensor* lr           = args.arg<vml::Tensor>(2) ; // scalar
    const vml::Tensor* grad         = args.arg<vml::Tensor>(3) ;
    const vml::Tensor* momentum     = args.arg<vml::Tensor>(4) ; // scalar
    const bool use_nesterov         = *args.arg<bool>(5) == 1 ? true : false ;

    if (!var || !accum || !lr || !grad || !momentum )
        return 1;

    LOG(LOG_PARAM)
        << __FUNCTION__  << ":"
	<< " var="     << var
	<< " accum="   << accum
	<< " lr="      << lr
	<< " grad="     << grad
	<< " momentum=" << momentum
	<< " use_nesterov ="    << use_nesterov  ;

    return vml::applyMomentum(*var, *accum, *lr, *grad, *momentum, use_nesterov ) ;
}

} // namespace

DEFINE_KERNEL(ApplyMomentum, op_ApplyMomentum);


//
// ApplyAdam
//


namespace {

template <typename T>
void apply_adam(
    const vml::Tensor* var_tensor,
    const vml::Tensor* m_tensor,
    const vml::Tensor* v_tensor,
    const vml::Tensor* grad_tensor,
    const T beta1_power,
    const T beta2_power,
    const T lr,
    const T beta1,
    const T beta2,
    const T epsilon,
    const bool use_nesterov
)
{
    const size_t nelems = var_tensor->nelems ;

    T* var        = reinterpret_cast<T*>(var_tensor->addr) ;
    T* m          = reinterpret_cast<T*>(m_tensor->addr) ;
    T* v          = reinterpret_cast<T*>(v_tensor->addr) ;
    const T* grad = reinterpret_cast<T*>(grad_tensor->addr) ;

    const T one = T(1.) ;

#if 1 // optimized

    const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;

#pragma omp parallel
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t eachNElement = nelems / nthreads ;
      int64_t remain       = nelems % nthreads ;

      int64_t elementBegin = eachNElement * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myElement    = eachNElement + ( threadid < remain ? 1 : 0 ) ;

      if( use_nesterov ) {
	for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
	  m[i] = m[i] + (one - beta1) * (grad[i] - m[i]) ;
	  v[i] = v[i] + (one - beta2) * (grad[i]*grad[i] - v[i]) ;
	  var[i] -= k * ( m[i] * beta1 + (one-beta1) * grad[i] ) / ( epsilon + std::sqrt(v[i])) ;
	}
      }
      else {
	for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
	  m[i] = m[i] + (one - beta1) * (grad[i] - m[i]) ;
	  v[i] = v[i] + (one - beta2) * (grad[i]*grad[i] - v[i]) ;
	  var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
	}
      }
    }
#else // original
    for(int64_t i=0; i<nelems; i++) {
      m[i] = m[i] + (one - beta1) * (grad[i] - m[i]) ;
    }
    for(int64_t i=0; i<nelems; i++) {
      v[i] = v[i] + (one - beta2) * (grad[i]*grad[i] - v[i]) ;
    }

    const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;
    if( use_nesterov ) {
      for(int64_t i=0; i<nelems; i++) {
	var[i] -= k * ( m[i] * beta1 + (one-beta1) * grad[i] ) / ( epsilon + std::sqrt(v[i])) ;
      }
    }
    else {
      for(int64_t i=0; i<nelems; i++) {
	var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
      }
    }
#endif
}

#ifdef LIBVETF_INTRINSIC
template<>
void apply_adam<float>(
    const vml::Tensor* var_tensor,
    const vml::Tensor* m_tensor,
    const vml::Tensor* v_tensor,
    const vml::Tensor* grad_tensor,
    const float beta1_power,
    const float beta2_power,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const bool use_nesterov
)
{
  const size_t nelems = var_tensor->nelems ;

  float* var        = reinterpret_cast<float*>(var_tensor->addr) ;
  float* m          = reinterpret_cast<float*>(m_tensor->addr) ;
  float* v          = reinterpret_cast<float*>(v_tensor->addr) ;
  const float* grad = reinterpret_cast<float*>(grad_tensor->addr) ;

  const float one = 1.f ;

  const float k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;

#pragma omp parallel
  {
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t eachNElement = nelems / nthreads ;
    int64_t remain       = nelems % nthreads ;

    int64_t elementBegin = eachNElement * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myElement    = eachNElement + ( threadid < remain ? 1 : 0 ) ;

    if( use_nesterov ) {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grad[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grad[i]*grad[i] - v[i]) ;
        var[i] -= k * ( m[i] * beta1 + (one-beta1) * grad[i] ) / ( epsilon + std::sqrt(v[i])) ;
      }
    }
    else {
      _apply_adam_f32(var+elementBegin, m+elementBegin, v+elementBegin,
                      beta1, beta2, epsilon, k, myElement, grad+elementBegin ) ;
    }
  }
}
#endif
}

int vml::applyAdam(
    vml::Tensor const& var,
    vml::Tensor const& m,
    vml::Tensor const& v,
    vml::Tensor const& beta1_power,	// scalar
    vml::Tensor const& beta2_power,	// scalar
    vml::Tensor const& lr,		// scalar
    vml::Tensor const& beta1,		// scalar
    vml::Tensor const& beta2,		// scalar
    vml::Tensor const& epsilon,		// scalar
    vml::Tensor const& grad,
    const bool use_nesterov
) {

  if( beta1_power.nelems != 1
	  || beta2_power.nelems !=1
	  || lr.nelems != 1
	  || beta1.nelems != 1
	  || beta2.nelems != 1
	  || epsilon.nelems !=1 )
      return 1 ;

  if( var.nelems != m.nelems
	  || var.nelems != v.nelems
	  || var.nelems != grad.nelems )
      return 1 ;

  if( var.dtype != m.dtype
	  ||  var.dtype != v.dtype
	  ||  var.dtype != beta1_power.dtype
	  ||  var.dtype != beta2_power.dtype
	  ||  var.dtype != lr.dtype
	  ||  var.dtype != beta1.dtype
	  ||  var.dtype != beta2.dtype
	  ||  var.dtype != epsilon.dtype
	  ||  var.dtype != grad.dtype )
      return 1 ;

  if (var.dtype == DT_FLOAT ) {
    apply_adam<float>(
	  &var,
	  &m,
	  &v,
	  &grad,
	  reinterpret_cast<const float*>(beta1_power.addr)[0],
	  reinterpret_cast<const float*>(beta2_power.addr)[0],
	  reinterpret_cast<const float*>(lr.addr)[0],
	  reinterpret_cast<const float*>(beta1.addr)[0],
	  reinterpret_cast<const float*>(beta2.addr)[0],
	  reinterpret_cast<const float*>(epsilon.addr)[0],
	  use_nesterov
    );
  } else {
      return 1;
  }

  return 0 ;

}

namespace {
int op_ApplyAdam(const VEOpArgs& args)
{
    if (args.nArguments() != 11)
        return 1;

    const vml::Tensor* var          = args.arg<vml::Tensor>(0) ;
    const vml::Tensor* m            = args.arg<vml::Tensor>(1) ;
    const vml::Tensor* v            = args.arg<vml::Tensor>(2) ;
    const vml::Tensor* beta1_power  = args.arg<vml::Tensor>(3) ; // scalar
    const vml::Tensor* beta2_power  = args.arg<vml::Tensor>(4) ; // scalar
    const vml::Tensor* lr           = args.arg<vml::Tensor>(5) ; // scalar
    const vml::Tensor* beta1        = args.arg<vml::Tensor>(6) ; // scalar
    const vml::Tensor* beta2        = args.arg<vml::Tensor>(7) ; // scalar
    const vml::Tensor* epsilon      = args.arg<vml::Tensor>(8) ; // scalar
    const vml::Tensor* grad         = args.arg<vml::Tensor>(9) ;
    const bool use_nesterov         = *args.arg<bool>(10) == 1 ? true : false ;


    if ( !var || !m || !v || !beta1_power || !beta2_power || !lr || !beta1 || !beta2 || !epsilon || !grad )
        return 1;

    LOG(LOG_PARAM)
        << __FUNCTION__  << ":"
  	<< " var="     << var
  	<< " m="       << m
  	<< " v="       << v
  	<< " beta1_power=" << beta1_power
  	<< " beta2_power=" << beta2_power
  	<< " lr="      << lr
  	<< " beta1="   << beta1
  	<< " beta2="   << beta2
  	<< " epsilon=" << epsilon
  	<< " grad="    << grad
  	<< " use_nesterov ="    << use_nesterov  ;

    return vml::applyAdam(
	*var, *m, *v,
	*beta1_power, *beta2_power,
	*lr, *beta1, *beta2, *epsilon,
	*grad,
	use_nesterov ) ;
}

} // namespace

DEFINE_KERNEL(ApplyAdam, op_ApplyAdam);
