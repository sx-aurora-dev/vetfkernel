#include "kernel.h"
#include "types.h"
#include "log.h"

#include "ve_ops_common.h"
#include "vml.h"

#include <omp.h>

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/intrinsic.h"
#endif

REGISTER_KERNEL("ApplyAdam", "op_ApplyAdam");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << (l1) << " expected but " << (l0); \
      return 1; \
  }

extern "C" {
  int op_ApplyAdam(const void* arg, size_t len);
}



//
// ApplyAdam
//

namespace {

template <typename T>
int apply_adam(bool use_nesterov, int64_t num_elements,
               uint64_t var_ptr, uint64_t m_ptr, uint64_t v_ptr,
               uint64_t beta1_power_ptr, uint64_t beta2_power_ptr,
               uint64_t lr_ptr,
               uint64_t beta1_ptr, uint64_t beta2_ptr, uint64_t epsilon_ptr,
               uint64_t grd_ptr )
{
  T* var = reinterpret_cast<T*>(var_ptr);
  T* m   = reinterpret_cast<T*>(m_ptr);
  T* v   = reinterpret_cast<T*>(v_ptr);

  const T* grd = reinterpret_cast<const T*>(grd_ptr);

  const T beta1_power = reinterpret_cast<const T*>(beta1_power_ptr)[0];
  const T beta2_power = reinterpret_cast<const T*>(beta2_power_ptr)[0];
  const T lr = reinterpret_cast<const T*>(lr_ptr)[0];
  const T beta1 = reinterpret_cast<const T*>(beta1_ptr)[0];
  const T beta2 = reinterpret_cast<const T*>(beta2_ptr)[0];
  const T epsilon = reinterpret_cast<const T*>(epsilon_ptr)[0];

  const T one = T(1.) ; 

#if 1 // optimized
 
  const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;

#pragma omp parallel
  { 
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t eachNElement = num_elements / nthreads ;
    int64_t remain       = num_elements % nthreads ;

    int64_t elementBegin = eachNElement * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myElement    = eachNElement + ( threadid < remain ? 1 : 0 ) ;

    if( use_nesterov ) {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
        var[i] -= k * ( m[i] * beta1 + (one-beta1) * grd[i] ) / ( epsilon + std::sqrt(v[i])) ;
      }
    }
    else {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
        var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
      }
    }
  }
#else // original
  for(int64_t i=0; i<num_elements; i++) {
    m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
  }
  for(int64_t i=0; i<num_elements; i++) {
    v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
  }
  
  const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;
  if( use_nesterov ) {
    for(int64_t i=0; i<num_elements; i++) {
      var[i] -= k * ( m[i] * beta1 + (one-beta1) * grd[i] ) / ( epsilon + std::sqrt(v[i])) ;
    }
  }
  else {
    for(int64_t i=0; i<num_elements; i++) {
      var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
    }
  }
#endif

  return 0 ;
}

#ifdef LIBVETF_INTRINSIC
template <>
int apply_adam<float>(bool use_nesterov, int64_t num_elements,
                      uint64_t var_ptr, uint64_t m_ptr, uint64_t v_ptr,
                      uint64_t beta1_power_ptr, uint64_t beta2_power_ptr,
                      uint64_t lr_ptr,
                      uint64_t beta1_ptr, uint64_t beta2_ptr, uint64_t epsilon_ptr,
                      uint64_t grd_ptr )
{
  float* var = reinterpret_cast<float*>(var_ptr);
  float* m   = reinterpret_cast<float*>(m_ptr);
  float* v   = reinterpret_cast<float*>(v_ptr);

  const float* grd = reinterpret_cast<const float*>(grd_ptr);

  const float beta1_power = reinterpret_cast<const float*>(beta1_power_ptr)[0];
  const float beta2_power = reinterpret_cast<const float*>(beta2_power_ptr)[0];
  const float lr = reinterpret_cast<const float*>(lr_ptr)[0];
  const float beta1 = reinterpret_cast<const float*>(beta1_ptr)[0];
  const float beta2 = reinterpret_cast<const float*>(beta2_ptr)[0];
  const float epsilon = reinterpret_cast<const float*>(epsilon_ptr)[0];

  const float one = 1.f ; 

  const float k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;

#pragma omp parallel
  { 
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t eachNElement = num_elements / nthreads ;
    int64_t remain       = num_elements % nthreads ;

    int64_t elementBegin = eachNElement * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myElement    = eachNElement + ( threadid < remain ? 1 : 0 ) ;

    if( use_nesterov ) {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
        var[i] -= k * ( m[i] * beta1 + (one-beta1) * grd[i] ) / ( epsilon + std::sqrt(v[i])) ;
      }
    }
    else {
      _apply_adam_f32(var+elementBegin, m+elementBegin, v+elementBegin,
                      beta1, beta2, epsilon, k, myElement, grd+elementBegin ) ;
    }
  }
  return 0 ;
}
#endif

}

int op_ApplyAdam(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    bool use_nesterov_ ;
    int64_t num_elements ;
    uint64_t var_ptr, m_ptr, v_ptr ;
    uint64_t beta1_power_ptr, beta2_power_ptr ;
    uint64_t lr ;
    uint64_t beta1_ptr, beta2_ptr, epsilon_ptr ;
    uint64_t grad_ptr;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype;

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    ret = apply_adam<float> (p->use_nesterov_, p->num_elements,
                             p->var_ptr, p->m_ptr, p->v_ptr,
                             p->beta1_power_ptr, p->beta2_power_ptr, p->lr,
                             p->beta1_ptr, p->beta2_ptr, p->epsilon_ptr,
                             p->grad_ptr ) ;
  }
  else if (p->dtype == DT_DOUBLE) {
    ret = apply_adam<double>(p->use_nesterov_, p->num_elements,
                             p->var_ptr, p->m_ptr, p->v_ptr,
                             p->beta1_power_ptr, p->beta2_power_ptr, p->lr,
                             p->beta1_ptr, p->beta2_ptr, p->epsilon_ptr,
                             p->grad_ptr ) ;
  }


  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}


//
// ApplyGradientDescent
//

namespace {

template <typename T>
void ApplyGradientDescent(
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

    if( alpha->nelems != 1 )
        return 1;

    if( var->nelems != delta->nelems )
        return 1 ;

    if( var->dtype != alpha->dtype ||  var->dtype != delta->dtype )
        return 1 ;

    if (var->dtype == DT_FLOAT ) {
      ApplyGradientDescent<float>(
	  var,
	  delta,
	  reinterpret_cast<const float*>(alpha->addr)[0]
      );
    } else {
        return 1;
    }

    return 0;
}

} // namespace

DEFINE_KERNEL(ApplyGradientDescent, op_ApplyGradientDescent);


//
// ApplyAdaDelta
//

namespace {

template <typename T>
void ApplyAdadelta(
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

    if( lr->nelems != 1 || rho->nelems != 1 || epsilon->nelems != 1 )
        return 1;

    if( var->nelems != accum->nelems
	  || var->nelems != accum_update->nelems
	  || var->nelems != grad->nelems )
        return 1 ;

    if( var->dtype != accum->dtype
	  ||  var->dtype != accum_update->dtype
	  ||  var->dtype != lr->dtype
	  ||  var->dtype != rho->dtype
	  ||  var->dtype != epsilon->dtype
	  ||  var->dtype != grad->dtype )
        return 1 ;

    if (var->dtype == DT_FLOAT ) {
      ApplyAdadelta<float>(
	  var,
	  accum,
	  accum_update,
	  grad,
	  reinterpret_cast<const float*>(lr->addr)[0],
	  reinterpret_cast<const float*>(rho->addr)[0],
	  reinterpret_cast<const float*>(epsilon->addr)[0]
      );
    } else {
        return 1;
    }

    return 0;
}

} // namespace

DEFINE_KERNEL(ApplyAdadelta, op_ApplyAdadelta);


//
// ApplyMomentum
//

namespace {

template <typename T>
void ApplyMomentum(
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

    if( lr->nelems != 1 || momentum->nelems != 1 )
        return 1;

    if( var->nelems != accum->nelems
	  || var->nelems != grad->nelems )
        return 1 ;

    if( var->dtype != accum->dtype
	  ||  var->dtype != lr->dtype
	  ||  var->dtype != grad->dtype
	  ||  var->dtype != momentum->dtype )
        return 1 ;

    if (var->dtype == DT_FLOAT ) {
      ApplyMomentum<float>(
	  var,
	  accum,
	  grad,
	  reinterpret_cast<const float*>(lr->addr)[0],
	  reinterpret_cast<const float*>(momentum->addr)[0],
	  use_nesterov
      );
    } else {
        return 1;
    }

    return 0;
}

} // namespace

DEFINE_KERNEL(ApplyMomentum, op_ApplyMomentum);
