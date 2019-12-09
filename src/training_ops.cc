#include "vml/types.h"
#include "vml/log.h"
#include "ve_ops_common.h"
#include "vml.h"

#include <omp.h>


//
// ApplyGradientDescent
//

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
