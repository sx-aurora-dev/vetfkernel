#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"
#include <vml.h>
#include <vector>
#include "vml/log.h"
#include "vml/types.h"

#define USE_VML

#ifndef USE_VML
#include "vml/profile.h"
#endif

//#define _DEBUG

REGISTER_KERNEL("Conv2D", "conv2d");

extern "C" {
    int conv2d(const void* arg, size_t len);
}

#if 0
// data layout must be compatible with vml::Tensor
struct Tensor {
  int32_t dtype;
  uint64_t addr;
  int32_t dims;
  int64_t nelems;
  int64_t dim_size[4];
} __attribute__((__packed__));
#endif

struct TensorParam {
    int w,h,c,n ;
};

struct ConvParam {
    uint64_t in;
    uint64_t filter;
    uint64_t out;
    TensorParam in_param;
    TensorParam filter_param;
    TensorParam out_param;

    int row_stride;
    int col_stride;
    int row_dilation;
    int col_dilation;
    int row_padding;
    int col_padding;

    int data_format;
    int data_type;
};

int conv2d(const void* arg, size_t len)
{
    LOG(LOG_TRACE) << __FUNCTION__ << ": begin";

#ifdef USE_VML
    // FIXME: use VEOpArgs
#ifdef _DEBUG
    fprintf(stderr, "[start]conv2d\n");
#endif
    assert(len == sizeof(ConvParam));
    const ConvParam& p = *(ConvParam*)arg;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p.data_type << " dformat=" << p.data_format;

#ifdef _DEBUG
    fprintf(stderr, "conv2d: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "conv2d: input   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_param.n, p.in_param.c, p.in_param.h, p.in_param.w ) ;
    fprintf(stderr, "conv2d: outnput (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_param.n, p.out_param.c, p.out_param.h, p.out_param.w ) ;
    fprintf(stderr, "conv2d: filter  (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.filter_param.n, p.filter_param.c, p.filter_param.h, p.filter_param.w ) ;

    fprintf(stderr, "conv2d: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            p.col_stride,   p.row_stride,
            p.col_dilation, p.row_dilation,
            p.col_padding,   p.row_padding);
#endif

    vml::TensorDesc<4> in;
    vml::TensorDesc<4> filter;
    vml::TensorDesc<4> out;
    std::vector<int> params(7);

#define COPY(t, ptr, param) \
    t.dtype = p.data_type; \
    t.addr = ptr; \
    t.dims = 4; \
    t.dim_size[0] = param.n; \
    t.dim_size[1] = param.c; \
    t.dim_size[2] = param.h; \
    t.dim_size[3] = param.w;

    COPY(in, p.in, p.in_param);
    COPY(filter, p.filter, p.filter_param);
    COPY(out, p.out, p.out_param);
    params[0] = p.col_stride;
    params[1] = p.row_stride;
    params[2] = p.col_dilation;
    params[3] = p.row_dilation;
    params[4] = p.col_padding;
    params[5] = p.row_padding;
    params[6] = p.data_format;

    return vml::conv2d(in, filter, out, params);
#else // USE_VML
    PROF_BEGIN(conv2d);

#ifdef _DEBUG
    fprintf(stderr, "[start]conv2d\n");
#endif
    assert(len == sizeof(ConvParam));
    const ConvParam& p = *(ConvParam*)arg;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p.data_type << " dformat=" << p.data_format;

#ifdef _DEBUG
    fprintf(stderr, "conv2d: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "conv2d: input   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_param.n, p.in_param.c, p.in_param.h, p.in_param.w ) ;
    fprintf(stderr, "conv2d: outnput (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_param.n, p.out_param.c, p.out_param.h, p.out_param.w ) ;
    fprintf(stderr, "conv2d: filter  (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.filter_param.n, p.filter_param.c, p.filter_param.h, p.filter_param.w ) ;

    fprintf(stderr, "conv2d: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            p.col_stride,   p.row_stride,
            p.col_dilation, p.row_dilation,
            p.col_padding,   p.row_padding);
#endif
     
    float * transformed_filter = NULL ;
    if( p.filter_param.n > 1 || p.filter_param.c > 1 ) {
      const int N = p.filter_param.n ;
      const int C = p.filter_param.c ;
      const int H = p.filter_param.h ;
      const int W = p.filter_param.w ;

      float * filter = (float *) p.filter ;     

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
    
    void *pIn     = (void *) p.in ;
    void *pOut    = (void *) p.out ;
    void *pFilter = (transformed_filter != NULL) ? (void*)transformed_filter : (void*)p.filter  ;
    
    vednnTensorParam_t ParamIn ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamOut ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = p.in_param.n ;
    ParamIn.channel = p.in_param.c ;
    ParamIn.height  = p.in_param.h ;
    ParamIn.width   = p.in_param.w ;

    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
    ParamFilter.inChannel  = p.in_param.c ;
    ParamFilter.outChannel = p.out_param.c ;
    ParamFilter.width      = p.filter_param.w ;
    ParamFilter.height     = p.filter_param.h ;
     
    ParamOut.dtype   = DTYPE_FLOAT ;
    ParamOut.batch   = p.out_param.n ;
    ParamOut.channel = p.out_param.c ;
    ParamOut.width   = p.out_param.w ;
    ParamOut.height  = p.out_param.h ;

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = p.col_stride ; 
    ParamConv.strideHeight   = p.row_stride ; 
    ParamConv.padWidth       = p.col_padding ;
    ParamConv.padHeight      = p.row_padding ;
    ParamConv.dilationWidth  = p.col_dilation ; 
    ParamConv.dilationHeight = p.row_dilation ; 

    vednnConvolutionForward(&ParamIn,     pIn,
                     	    &ParamFilter, pFilter,
                     	    &ParamOut,    pOut, 
                     	    &ParamConv,
                     	    VEDNN_CONV_ALGORITHM_DIRECT );
    

    if( transformed_filter != NULL ) free(transformed_filter) ;

    PROF_END(conv2d)
      << " in=[ " << p.in_param.n << " " << p.in_param.c << " " <<  p.in_param.h << " " <<  p.in_param.w << " ]"
      << " filter=[ " << p.filter_param.n << " " << p.filter_param.c << " " <<  p.filter_param.h << " " <<  p.filter_param.w << " ]"
      << " out=[ " << p.out_param.n << " " << p.out_param.c << " " <<  p.out_param.h << " " <<  p.out_param.w << " ]"
      << " stride=[" << p.col_stride << " " << p.row_stride << "]"
      << " padding=[" << p.col_padding << " " << p.row_padding << "]";

#ifdef _DEBUG
    fprintf(stderr, "[end]conv2d\n");
#endif
    LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=0";
    return 0;
#endif // USE_VML
}
