#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"

#include "ve_ops_common.h"
#include "vml.h"

REGISTER_KERNEL("DepthwiseConv2DGradFilter", "depthwise_conv2d_grad_filter");

extern "C" {
    int depthwise_conv2d_grad_filter(const void* arg, size_t len);
}

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

} ;

DEFINE_KERNEL(DepthwiseConv2D, depthwise_conv2d);
DEFINE_KERNEL(DepthwiseConv2DGradData, depthwise_conv2d_grad_data);

int depthwise_conv2d_grad_filter(const void* arg, size_t len)
{
    LOG(LOG_TRACE) << __FUNCTION__ << ": begin";

    int rc = 1 ;

    struct Args {
      // Tensor Pointers
      uint64_t input_ptr ;
      uint64_t filter_ptr ;
      uint64_t output_ptr ;

      // Tensor Format
      int data_format ;
      int data_type ;

      // Input layer dimensions
      int batch;
      int in_rows;
      int in_cols;
      int in_depth;
      int filter_rows;
      int filter_cols;
      int depth_multiplier;
      int stride;
      int pad_rows;
      int pad_cols;

      // Output layer dimensions
      int out_rows;
      int out_cols;
      int out_depth;
    };

    assert(len == sizeof(Args));
    const Args& p = *(Args*)arg;

#ifdef _DEBUG
    fprintf(stderr, "depthwise_conv2d_grad_filter: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NCHW

    fprintf(stderr, "depthwise_conv2d_grad_filter: batch=%d\n", p.batch) ;
    fprintf(stderr, "depthwise_conv2d_grad_filter: in_rows=%d in_cols=%d in_depth=%d\n", p.in_rows, p.in_cols, p.in_depth) ;
    fprintf(stderr, "depthwise_conv2d_grad_filter: filter_rows=%d filter_cols=%d\n", p.filter_rows, p.filter_cols ) ;
    fprintf(stderr, "depthwise_conv2d_grad_filter: depth_multiplier=%d\n", p.depth_multiplier) ;
    fprintf(stderr, "depthwise_conv2d_grad_filter: stride=%d\n", p.stride) ;
    fprintf(stderr, "depthwise_conv2d_grad_filter: pad_rows=%d pad_cols=%d\n", p.pad_rows, p.pad_cols) ;
    fprintf(stderr, "depthwise_conv2d_grad_filter: out_rows=%d out_cols=%d out_depth=%d\n", p.out_rows, p.out_cols, p.out_depth) ;
#endif

#if 0	// naive-version
    {
      int64_t batch	= p.batch ;
      int64_t inChannel = p.in_depth ;
      int64_t inWidth	= p.in_cols ;
      int64_t inHeight	= p.in_rows ;
      int64_t outWidth	= p.out_cols ;
      int64_t outHeight = p.out_rows ;
      int64_t kernWidth	= p.filter_cols ;
      int64_t kernHeight= p.filter_rows ;

      int64_t depth_multiplier = p.depth_multiplier ;

      int64_t strideWidth	= p.stride ;
      int64_t strideHeight	= p.stride ;
      int64_t padWidth		= p.pad_cols ;
      int64_t padHeight		= p.pad_rows ;

      const float * __restrict pIn = (float * ) p.input_ptr;
      float * __restrict pKernel = (float *) p.filter_ptr ;
      const float * __restrict pOut = (float *) p.output_ptr;

      for(int64_t i=0; i<inChannel*depth_multiplier*kernHeight*kernWidth; i++) pKernel[i] = 0.0f ;

      for (int64_t n=0; n<batch; n++) {
	for (int64_t c=0; c<inChannel; c++) {
	  for (int64_t d=0; d<depth_multiplier; d++) {
	    int64_t k = c * depth_multiplier + d ;
	    for (int64_t y=0; y<outHeight; y++) {
	      int64_t i = y * strideHeight - padHeight;
	      for (int64_t x=0; x<outWidth; x++) {
		int64_t j = x * strideWidth - padWidth;
		int64_t outIndex  = ((n * (depth_multiplier * inChannel) + k) * outHeight + y) * outWidth + x;

		for (int64_t r=0; r<kernHeight; r++) {
		  for (int64_t s=0; s<kernWidth; s++) {
		      int64_t h = i + r ;
		      int64_t w = j + s ;
		      if (h < 0 || inHeight <= h) {
			  continue;
		      }
		      if (w < 0 || inWidth <= w) {
			  continue;
		      }
		      int64_t inputIndex  = ((n * inChannel + c) * inHeight + h) * inWidth + w;
		      int64_t kernelIndex = ((r* kernWidth + s) * inChannel + c ) * depth_multiplier + d; // HWCN_LAYOUT
		      pKernel[kernelIndex] += (pIn[inputIndex] * pOut[outIndex]);
		  } // kernWidth
		} // kernHeight
	      } // outWidth
	    } // outHeight
	  } // outChannel
	} // inChannel
      } // batch
    }
#else // vednn version


    const int64_t fsize = p.depth_multiplier * p.in_depth * p.filter_rows * p.filter_cols ;

    float * transformed_filter = NULL ;
    if( p.depth_multiplier > 1 || p.in_depth > 1 ) {
      const int N = p.depth_multiplier ;
      const int C = p.in_depth ;
      const int H = p.filter_rows ;
      const int W = p.filter_cols ;

      transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
    }

    {
      vednnTensorParam_t ParamIn ;
      vednnFilterParam_t ParamFilter ;
      vednnTensorParam_t ParamOut ;

      vednnConvolutionParam_t ParamConv ;

      ParamIn.dtype   = DTYPE_FLOAT ;
      ParamIn.batch   = p.batch ;
      ParamIn.channel = p.in_depth ;
      ParamIn.height  = p.in_rows ;
      ParamIn.width   = p.in_cols ;

      ParamFilter.dtype      = DTYPE_FLOAT ;
      ParamFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
      ParamFilter.inChannel  = 1 ;
      ParamFilter.outChannel = p.depth_multiplier ;
      ParamFilter.height     = p.filter_rows ;
      ParamFilter.width      = p.filter_cols ;

      ParamOut.dtype   = DTYPE_FLOAT ;
      ParamOut.batch   = p.batch ;
      ParamOut.channel = p.depth_multiplier * p.in_depth ;
      ParamOut.height  = p.out_rows ;
      ParamOut.width   = p.out_cols ;

      ParamConv.group          = p.in_depth ;
      ParamConv.strideWidth    = p.stride ;
      ParamConv.strideHeight   = p.stride ;
      ParamConv.padHeight      = p.pad_rows ;
      ParamConv.padWidth       = p.pad_cols ;
      ParamConv.dilationWidth  = 1 ;
      ParamConv.dilationHeight = 1 ;

      float * pIn  =  (float *) (p.input_ptr) ;
      float * pOut =  (float *) (p.output_ptr) ;
      float * pFilter = ((transformed_filter != NULL) ? transformed_filter : (float*) (p.filter_ptr)) ;
      vednnConvolutionBackwardFilter(&ParamIn,     (void*) pIn,
				     &ParamOut,    (void*) pOut,
				     &ParamFilter, (void*) pFilter,
				     &ParamConv,
				     VEDNN_CONV_ALGORITHM_DIRECT );
    }

    if( p.depth_multiplier > 1 || p.in_depth > 1 ) {
      const int N = p.depth_multiplier ;
      const int C = p.in_depth ;
      const int H = p.filter_rows ;
      const int W = p.filter_cols ;

      float * filter = (float *) p.filter_ptr ;

#if 0
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int h=0; h<H ; h++) {
            for(int w=0; w<W ; w++) {
              filter[((h*W+w)*C+c)*N+n] = transformed_filter[((n*C+c)*H+h)*W+w] ;
 	    }
          }
        }
      }
#else
#pragma omp parallel for
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int hw=0; hw<H*W ; hw++) {
            filter[((hw)*C+c)*N+n] = transformed_filter[((n*C+c)*H)*W+hw] ;
          }
        }
      }
#endif
      free(transformed_filter) ;
    }
#endif

    LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=" << rc ;

    return 0 ;
}
