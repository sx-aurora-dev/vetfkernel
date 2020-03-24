#include <cstdint>
#include "ve_ops_common.h"
#include "vml.h"
#include "vml/types.h"

#include <vednn.h>

namespace {

int op_maxpool(const VEOpArgs& args)
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  //LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();

  int ret = 1;

  if (args.nArguments() != 8) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 8. But "
        << args.nArguments();
    goto error_exit;
  }

  {
    // dimensions are checked in TF
    vml::Tensor const* input  = args.arg<vml::Tensor>(0); // NCHW Tensor
    vml::Tensor const* output = args.arg<vml::Tensor>(1); // NCHW Tensor

    const int64_t row_window  = *args.arg<int64_t>(2) ;
    const int64_t col_window  = *args.arg<int64_t>(3) ;
    const int64_t row_stride  = *args.arg<int64_t>(4) ;
    const int64_t col_stride  = *args.arg<int64_t>(5) ;
    const int64_t row_padding = *args.arg<int64_t>(6) ;
    const int64_t col_padding = *args.arg<int64_t>(7) ;

#if 0
    const int data_layout     = *args.arg<int>(8) ;	// currently support NCHW only.
#endif

#define PT(T) \
    LOG(LOG_PARAM) << __FUNCTION__ << ": " #T "=" << *T

    PT(input) ;
    PT(output) ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": window="  << row_window  << "x" << col_window ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": stride="  << row_stride  << "x" << col_stride ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": padding=" << row_padding << "x" << col_padding ;

    if( input->dtype == DT_FLOAT )
    {
      void *pIn       = (void *) input->addr ;
      void *pOut      = (void *) output->addr ;

      vednnTensorParam_t ParamIn ;
      vednnTensorParam_t ParamOut ;

      vednnPoolingParam_t ParamPool ;

      ParamIn.dtype   = DTYPE_FLOAT ;
      ParamIn.batch   = input->dim_size[0] ;
      ParamIn.channel = input->dim_size[1] ;
      ParamIn.height  = input->dim_size[2] ;
      ParamIn.width   = input->dim_size[3] ;

      ParamOut.dtype   = DTYPE_FLOAT ;
      ParamOut.batch   = output->dim_size[0] ;
      ParamOut.channel = output->dim_size[1] ;
      ParamOut.height  = output->dim_size[2] ;
      ParamOut.width   = output->dim_size[3] ;

      ParamPool.windowWidth  = col_window ;
      ParamPool.windowHeight = row_window ;
      ParamPool.strideWidth  = col_stride ;
      ParamPool.strideHeight = row_stride ;
      ParamPool.padWidth     = col_padding ;
      ParamPool.padHeight    = row_padding ;

      ret = vednnMaxPoolingForward(&ParamIn,     pIn,
	                           &ParamOut,    pOut,
                     	           &ParamPool ) ;
    }
  }

error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}


int op_maxpoolgrad(const VEOpArgs& args)
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  //LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();

  int ret = 1;

  if (args.nArguments() != 10) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 8. But "
        << args.nArguments();
    goto error_exit;
  }

  {
    // dimensions are checked in TF
    vml::Tensor const* output_bp = args.arg<vml::Tensor>(0); // NCHW Tensor
    vml::Tensor const* output    = args.arg<vml::Tensor>(1);   // NCHW Tensor
    vml::Tensor const* input     = args.arg<vml::Tensor>(2);    // NCHW Tensor
    vml::Tensor const* input_bp  = args.arg<vml::Tensor>(3);    // NCHW Tensor

    const int64_t row_window  = *args.arg<int64_t>(4) ;
    const int64_t col_window  = *args.arg<int64_t>(5) ;
    const int64_t row_stride  = *args.arg<int64_t>(6) ;
    const int64_t col_stride  = *args.arg<int64_t>(7) ;
    const int64_t row_padding = *args.arg<int64_t>(8) ;
    const int64_t col_padding = *args.arg<int64_t>(9) ;

#if 0
    const int data_layout     = *args.arg<int>(10) ;	// currently support NCHW only.
#endif

#define PT(T) \
    LOG(LOG_PARAM) << __FUNCTION__ << ": " #T "=" << *T

    PT(input) ;
    PT(output) ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": window="  << row_window  << "x" << col_window ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": stride="  << row_stride  << "x" << col_stride ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": padding=" << row_padding << "x" << col_padding ;

    if( input->dtype == DT_FLOAT )
    {
      void *pGradOut  = (void *) output_bp->addr ;
      void *pOut      = (void *) output->addr ;
      void *pIn       = (void *) input->addr ;
      void *pGradIn   = (void *) input_bp->addr ;

      vednnTensorParam_t ParamGradOut ;
      vednnTensorParam_t ParamOut ;
      vednnTensorParam_t ParamIn ;
      vednnTensorParam_t ParamGradIn ;

      vednnPoolingParam_t ParamPool ;

      ParamGradOut.dtype   = ParamOut.dtype   = DTYPE_FLOAT ;
      ParamGradOut.batch   = ParamOut.batch   = output->dim_size[0] ;
      ParamGradOut.channel = ParamOut.channel = output->dim_size[1] ;
      ParamGradOut.height  = ParamOut.height  = output->dim_size[2] ;
      ParamGradOut.width   = ParamOut.width   = output->dim_size[3] ;

      ParamGradIn.dtype   = ParamIn.dtype   = DTYPE_FLOAT ;
      ParamGradIn.batch   = ParamIn.batch   = input->dim_size[0] ;
      ParamGradIn.channel = ParamIn.channel = input->dim_size[1] ;
      ParamGradIn.height  = ParamIn.height  = input->dim_size[2] ;
      ParamGradIn.width   = ParamIn.width   = input->dim_size[3] ;

      ParamPool.windowWidth  = col_window ;
      ParamPool.windowHeight = row_window ;
      ParamPool.strideWidth  = col_stride ;
      ParamPool.strideHeight = row_stride ;
      ParamPool.padWidth     = col_padding ;
      ParamPool.padHeight    = row_padding ;

      ret = vednnMaxPoolingBackward(&ParamGradOut,   pGradOut,
                       	            &ParamOut,       pOut,
                                    &ParamIn,        pIn,
                       	            &ParamGradIn,    pGradIn,
                     	            &ParamPool ) ;
    }
  }

error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}
} // namespace

DEFINE_KERNEL(MaxPool, op_maxpool);
DEFINE_KERNEL(MaxPoolGrad, op_maxpoolgrad);
