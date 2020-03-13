#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"

#include "ve_ops_common.h"
#include "vml.h"

namespace {

int depthwise_conv2d(VEOpArgs const& args)
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  //LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();

  int ret = 1;

  if (args.nArguments() != 6) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 6. But "
        << args.nArguments();
    goto error_exit;
  }

  {
    // dimensions are checked in TF
    vml::Tensor const* input  = args.arg<vml::Tensor>(0); // NCHW Tensor
    vml::Tensor const* filter = args.arg<vml::Tensor>(1); // NCHW Tensor (N=out_depth, C=1)
    vml::Tensor const* output = args.arg<vml::Tensor>(2); // NCHW Tensor

    const int64_t stride   = *args.arg<int64_t>(3) ;
    const int64_t pad_rows = *args.arg<int64_t>(4) ;
    const int64_t pad_cols = *args.arg<int64_t>(5) ;

#define PT(T) \
    LOG(LOG_PARAM) << __FUNCTION__ << ": " #T "=" << *T

    PT(input) ;
    PT(filter) ;
    PT(output) ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": stride="   << stride;
    LOG(LOG_PARAM) << __FUNCTION__ << ": pad_rows=" << pad_rows;
    LOG(LOG_PARAM) << __FUNCTION__ << ": pad_cols=" << pad_cols;

    vednnTensorParam_t ParamIn ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamOut ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = input->dim_size[0] ;
    ParamIn.channel = input->dim_size[1] ;
    ParamIn.height  = input->dim_size[2] ;
    ParamIn.width   = input->dim_size[3] ;

    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
    ParamFilter.inChannel  = filter->dim_size[1] ;
    ParamFilter.outChannel = filter->dim_size[0] ;
    ParamFilter.height     = filter->dim_size[2] ;
    ParamFilter.width      = filter->dim_size[3] ;

    ParamOut.dtype   = DTYPE_FLOAT ;
    ParamOut.batch   = output->dim_size[0] ;
    ParamOut.channel = output->dim_size[1] ;
    ParamOut.height  = output->dim_size[2] ;
    ParamOut.width   = output->dim_size[3] ;

    ParamConv.group          = input->dim_size[1] ; ;
    ParamConv.strideWidth    = stride ;
    ParamConv.strideHeight   = stride ;
    ParamConv.padHeight      = pad_rows ;
    ParamConv.padWidth       = pad_cols ;
    ParamConv.dilationWidth  = 1 ;
    ParamConv.dilationHeight = 1 ;

    float * pIn     = reinterpret_cast<float*>(input->addr) ;
    float * pOut    = reinterpret_cast<float*>(output->addr) ;
    float * pFilter = reinterpret_cast<float*>(filter->addr) ;

    ret = vednnConvolutionForward(&ParamIn,     (void*) pIn,
				  &ParamFilter, (void*) pFilter,
				  &ParamOut,    (void*) pOut,
				  &ParamConv,
				  VEDNN_CONV_ALGORITHM_DIRECT );
  }


error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}


int depthwise_conv2d_grad_data(VEOpArgs const& args)
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  //LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();

  int ret = 1;

  if (args.nArguments() != 6) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 6. But "
        << args.nArguments();
    goto error_exit;
  }

  {

    // dimensions are checked in TF
    vml::Tensor const* input_bp  = args.arg<vml::Tensor>(0); // NCHW Tensor
    vml::Tensor const* filter    = args.arg<vml::Tensor>(1); // NCHW Tensor (N=out_depth, C=1)
    vml::Tensor const* output_bp = args.arg<vml::Tensor>(2); // NCHW Tensor

    const int64_t stride   = *args.arg<int64_t>(3) ;
    const int64_t pad_rows = *args.arg<int64_t>(4) ;
    const int64_t pad_cols = *args.arg<int64_t>(5) ;

#define PT(T) \
    LOG(LOG_PARAM) << __FUNCTION__ << ": " #T "=" << *T

    PT(input_bp) ;
    PT(filter) ;
    PT(output_bp) ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": stride="   << stride;
    LOG(LOG_PARAM) << __FUNCTION__ << ": pad_rows=" << pad_rows;
    LOG(LOG_PARAM) << __FUNCTION__ << ": pad_cols=" << pad_cols;

    vednnTensorParam_t ParamIn ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamOut ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = input_bp->dim_size[0] ;
    ParamIn.channel = input_bp->dim_size[1] ;
    ParamIn.height  = input_bp->dim_size[2] ;
    ParamIn.width   = input_bp->dim_size[3] ;

    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
    ParamFilter.inChannel  = filter->dim_size[1] ;
    ParamFilter.outChannel = filter->dim_size[0] ;
    ParamFilter.height     = filter->dim_size[2] ;
    ParamFilter.width      = filter->dim_size[3] ;

    ParamOut.dtype   = DTYPE_FLOAT ;
    ParamOut.batch   = output_bp->dim_size[0] ;
    ParamOut.channel = output_bp->dim_size[1] ;
    ParamOut.height  = output_bp->dim_size[2] ;
    ParamOut.width   = output_bp->dim_size[3] ;

    ParamConv.group          = input_bp->dim_size[1] ; ;
    ParamConv.strideWidth    = stride ;
    ParamConv.strideHeight   = stride ;
    ParamConv.padHeight      = pad_rows ;
    ParamConv.padWidth       = pad_cols ;
    ParamConv.dilationWidth  = 1 ;
    ParamConv.dilationHeight = 1 ;

    float * pIn     = reinterpret_cast<float*>(input_bp->addr) ;
    float * pOut    = reinterpret_cast<float*>(output_bp->addr) ;
    float * pFilter = reinterpret_cast<float*>(filter->addr) ;

    ret = vednnConvolutionBackwardData(&ParamOut,    (void*) pOut,
				       &ParamFilter, (void*) pFilter,
				       &ParamIn,     (void*) pIn,
				       &ParamConv,
				       VEDNN_CONV_ALGORITHM_DIRECT );
  }


error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}


int depthwise_conv2d_grad_filter(VEOpArgs const& args)
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  //LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();

  int ret = 1;

  if (args.nArguments() != 6) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 6. But "
        << args.nArguments();
    goto error_exit;
  }

  {

    // dimensions are checked in TF
    vml::Tensor const* input     = args.arg<vml::Tensor>(0); // NCHW Tensor
    vml::Tensor const* filter_bp = args.arg<vml::Tensor>(1); // NCHW Tensor (N=out_depth, C=1)
    vml::Tensor const* output_bp = args.arg<vml::Tensor>(2); // NCHW Tensor

    const int64_t stride   = *args.arg<int64_t>(3) ;
    const int64_t pad_rows = *args.arg<int64_t>(4) ;
    const int64_t pad_cols = *args.arg<int64_t>(5) ;

#define PT(T) \
    LOG(LOG_PARAM) << __FUNCTION__ << ": " #T "=" << *T

    PT(input) ;
    PT(filter_bp) ;
    PT(output_bp) ;
    LOG(LOG_PARAM) << __FUNCTION__ << ": stride="   << stride;
    LOG(LOG_PARAM) << __FUNCTION__ << ": pad_rows=" << pad_rows;
    LOG(LOG_PARAM) << __FUNCTION__ << ": pad_cols=" << pad_cols;

    vednnTensorParam_t ParamIn ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamOut ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = input->dim_size[0] ;
    ParamIn.channel = input->dim_size[1] ;
    ParamIn.height  = input->dim_size[2] ;
    ParamIn.width   = input->dim_size[3] ;

    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
    ParamFilter.inChannel  = filter_bp->dim_size[1] ;
    ParamFilter.outChannel = filter_bp->dim_size[0] ;
    ParamFilter.height     = filter_bp->dim_size[2] ;
    ParamFilter.width      = filter_bp->dim_size[3] ;

    ParamOut.dtype   = DTYPE_FLOAT ;
    ParamOut.batch   = output_bp->dim_size[0] ;
    ParamOut.channel = output_bp->dim_size[1] ;
    ParamOut.height  = output_bp->dim_size[2] ;
    ParamOut.width   = output_bp->dim_size[3] ;

    ParamConv.group          = input->dim_size[1] ; ;
    ParamConv.strideWidth    = stride ;
    ParamConv.strideHeight   = stride ;
    ParamConv.padHeight      = pad_rows ;
    ParamConv.padWidth       = pad_cols ;
    ParamConv.dilationWidth  = 1 ;
    ParamConv.dilationHeight = 1 ;

    float * pIn     = reinterpret_cast<float*>(input->addr) ;
    float * pOut    = reinterpret_cast<float*>(output_bp->addr) ;
    float * pFilter = reinterpret_cast<float*>(filter_bp->addr) ;

    ret = vednnConvolutionBackwardFilter(&ParamIn,     (void*) pIn,
					 &ParamOut,    (void*) pOut,
					 &ParamFilter, (void*) pFilter,
					 &ParamConv,
					 VEDNN_CONV_ALGORITHM_DIRECT );
  }


error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}
} ;



DEFINE_KERNEL(DepthwiseConv2D, depthwise_conv2d);
DEFINE_KERNEL(DepthwiseConv2DGradData, depthwise_conv2d_grad_data);
DEFINE_KERNEL(DepthwiseConv2DGradFilter, depthwise_conv2d_grad_filter);
