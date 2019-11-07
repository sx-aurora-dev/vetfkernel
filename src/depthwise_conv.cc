#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"
#include "log.h"

REGISTER_KERNEL("DepthwiseConv2D",           "depthwise_conv2d");
REGISTER_KERNEL("DepthwiseConv2DGradData",   "depthwise_conv2d_grad_data");
REGISTER_KERNEL("DepthwiseConv2DGradFilter", "depthwise_conv2d_grad_filter");

extern "C" {
    int depthwise_conv2d(const void* arg, size_t len);
    int depthwise_conv2d_grad_data(const void* arg, size_t len);
    int depthwise_conv2d_grad_filter(const void* arg, size_t len);
}

int depthwise_conv2d(const void* arg, size_t len)
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
    fprintf(stderr, "depthwise_conv2d: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NCHW

    fprintf(stderr, "depthwise_conv2d: batch=%d\n", p.batch) ;
    fprintf(stderr, "depthwise_conv2d: in_rows=%d in_cols=%d in_depth=%d\n", p.in_rows, p.in_cols, p.in_depth) ;
    fprintf(stderr, "depthwise_conv2d: filter_rows=%d filter_cols=%d\n", p.filter_rows, p.filter_cols ) ;
    fprintf(stderr, "depthwise_conv2d: depth_multiplier=%d\n", p.depth_multiplier) ;
    fprintf(stderr, "depthwise_conv2d: stride=%d\n", p.stride) ;
    fprintf(stderr, "depthwise_conv2d: pad_rows=%d pad_cols=%d\n", p.pad_rows, p.pad_cols) ;
    fprintf(stderr, "depthwise_conv2d: out_rows=%d out_cols=%d out_depth=%d\n", p.out_rows, p.out_cols, p.out_depth) ;
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
      const float * __restrict pKernel = (float *) p.filter_ptr ;
      float * __restrict pOut = (float *) p.output_ptr;

      for (int64_t n=0; n<batch; n++) {
	for (int64_t c=0; c<inChannel; c++) {
	  for (int64_t d=0; d<depth_multiplier; d++) {
	    int64_t k = c * depth_multiplier + d ;
	    for (int64_t y=0; y<outHeight; y++) {
	      int64_t i = y * strideHeight - padHeight;
	      for (int64_t x=0; x<outWidth; x++) {
		int64_t j = x * strideWidth - padWidth;
		int64_t outIndex  = ((n * (depth_multiplier * inChannel) + k) * outHeight + y) * outWidth + x;
		float sum = 0.0;

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
		      sum += (pIn[inputIndex] * pKernel[kernelIndex]);
		  } // kernWidth
		} // kernHeight
		pOut[outIndex] = sum ;
	      } // outWidth
	    } // outHeight
	  } // outChannel
	} // inChannel
      } // batch
    }
#else // vednn version
    float * transformed_filter = NULL ;
    if( p.depth_multiplier > 1 || p.in_depth > 1 ) {
      const int N = p.depth_multiplier ;
      const int C = p.in_depth ;
      const int H = p.filter_rows ;
      const int W = p.filter_cols ;

      float * filter = (float *) p.filter_ptr ;

      transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
#if 0
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int h=0; h<H ; h++) {
            for(int w=0; w<W ; w++) {
              transformed_filter[((n*C+c)*H+h)*W+w] = filter[((h*W+w)*C+c)*N+n] ; 
 	    }
          }
        }
      }
#else
#pragma omp parallel for
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int hw=0; hw<H*W ; hw++) {
            transformed_filter[((n*C+c)*H)*W+hw] = filter[((hw)*C+c)*N+n] ; 
          }
        }
      }
#endif
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

      vednnConvolutionForward(&ParamIn,     (void*) pIn,
			      &ParamFilter, (void*) pFilter,
			      &ParamOut,    (void*) pOut,
			      &ParamConv,
			      VEDNN_CONV_ALGORITHM_DIRECT );
    }


    if( transformed_filter != NULL ) free(transformed_filter) ;
#endif

    LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=" << rc ;

    return 0 ;
}


int depthwise_conv2d_grad_data(const void* arg, size_t len)
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
    fprintf(stderr, "depthwise_conv2d_grad_data: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NCHW

    fprintf(stderr, "depthwise_conv2d_grad_data: batch=%d\n", p.batch) ;
    fprintf(stderr, "depthwise_conv2d_grad_data: in_rows=%d in_cols=%d in_depth=%d\n", p.in_rows, p.in_cols, p.in_depth) ;
    fprintf(stderr, "depthwise_conv2d_grad_data: filter_rows=%d filter_cols=%d\n", p.filter_rows, p.filter_cols ) ;
    fprintf(stderr, "depthwise_conv2d_grad_data: depth_multiplier=%d\n", p.depth_multiplier) ;
    fprintf(stderr, "depthwise_conv2d_grad_data: stride=%d\n", p.stride) ;
    fprintf(stderr, "depthwise_conv2d_grad_data: pad_rows=%d pad_cols=%d\n", p.pad_rows, p.pad_cols) ;
    fprintf(stderr, "depthwise_conv2d_grad_data: out_rows=%d out_cols=%d out_depth=%d\n", p.out_rows, p.out_cols, p.out_depth) ;
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

      float * __restrict pIn = (float * ) p.input_ptr;
      const float * __restrict pKernel = (float *) p.filter_ptr ;
      const float * __restrict pOut = (float *) p.output_ptr;

      for(int64_t i=0; i<batch*inChannel*inHeight*inWidth; i++) pIn[i] = 0.0f ;

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
		      pIn[inputIndex] += (pOut[outIndex] * pKernel[kernelIndex]);
		  } // kernWidth
		} // kernHeight
	      } // outWidth
	    } // outHeight
	  } // outChannel
	} // inChannel
      } // batch
    }
#else // vednn version
    float * transformed_filter = NULL ;
    if( p.depth_multiplier > 1 || p.in_depth > 1 ) {
      const int N = p.depth_multiplier ;
      const int C = p.in_depth ;
      const int H = p.filter_rows ;
      const int W = p.filter_cols ;

      float * filter = (float *) p.filter_ptr ;

      transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
#if 0
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int h=0; h<H ; h++) {
            for(int w=0; w<W ; w++) {
              transformed_filter[((n*C+c)*H+h)*W+w] = filter[((h*W+w)*C+c)*N+n] ;
 	    }
          }
        }
      }
#else
#pragma omp parallel for
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int hw=0; hw<H*W ; hw++) {
            transformed_filter[((n*C+c)*H)*W+hw] = filter[((hw)*C+c)*N+n] ;
          }
        }
      }
#endif
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
      vednnConvolutionBackwardData(&ParamOut,    (void*) pOut,
				   &ParamFilter, (void*) pFilter,
				   &ParamIn,     (void*) pIn,
				   &ParamConv,
				   VEDNN_CONV_ALGORITHM_DIRECT );
    }

    if( transformed_filter != NULL ) free(transformed_filter) ;
#endif

    LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=" << rc ;

    return 0 ;
}


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
