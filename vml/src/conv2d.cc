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
#include "vml/profile.h"

//#define _DEBUG

int vml::conv2d(vml::Tensor const& in,
                vml::Tensor const& filter,
                vml::Tensor const& out,
                std::vector<int> params) // stride[2],dilation[2],padding[2],data_format
{
    LOG(LOG_TRACE) << __FUNCTION__ << ": begin";

    PROF_BEGIN(conv2d);

#ifdef _DEBUG
    fprintf(stderr, "[start] vml::conv2d\n");
#endif

    assert(in.dims == 4);
    assert(filter.dims == 4);
    assert(out.dims == 4);
    assert(params.size() == 7);

    int data_format = params[6];
    LOG(LOG_PARAM) << __FUNCTION__
      << ": dtype=" << in.dtype << " dformat=" << data_format;

    if (data_format != FORMAT_NCHW)
      return 1;

#ifdef _DEBUG
    fprintf(stderr, "conv2d: data_format=%d data_type=%d\n",
            data_format, in.dtype);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "conv2d: input   (N,C,H,W) = (%d,%d,%d,%d)\n",
            in.dim_size[0], in.dim_size[1], in.dim_size[2], in.dim_size[3]);
    fprintf(stderr, "conv2d: outnput (N,C,H,W) = (%d,%d,%d,%d)\n",
            filter.dim_size[0], filter.dim_size[1], filter.dim_size[2], filter.dim_size[3]);
    fprintf(stderr, "conv2d: filter  (N,C,H,W) = (%d,%d,%d,%d)\n",
            out.dim_size[0], out.dim_size[1], out.dim_size[2], out.dim_size[3]);

    fprintf(stderr, "conv2d: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            params[0], params[1], params[2], params[3], params[4], params[5]);
#endif

    float * transformed_filter = NULL ;
    if( filter.dim_size[0] > 1 || filter.dim_size[1] > 1 ) {
      const int N = filter.dim_size[0];
      const int C = filter.dim_size[1];
      const int H = filter.dim_size[2];
      const int W = filter.dim_size[3];

      float * pfilter = filter.ptr<float*>();

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
            transformed_filter[((n*C+c)*H)*W+hw] = pfilter[((hw)*C+c)*N+n] ; 
          }
        }
      }
#endif
    }
    
    void *pIn     = in.ptr<void*>();
    void *pOut    = out.ptr<void*>();
    void *pFilter = (transformed_filter != NULL) ? (void*)transformed_filter : filter.ptr<void*>();
    
    vednnTensorParam_t ParamIn ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamOut ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = in.dim_size[0];
    ParamIn.channel = in.dim_size[1];
    ParamIn.height  = in.dim_size[2];
    ParamIn.width   = in.dim_size[3];

    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
    ParamFilter.inChannel  = in.dim_size[1];
    ParamFilter.outChannel = out.dim_size[1];
    ParamFilter.height     = filter.dim_size[2];
    ParamFilter.width      = filter.dim_size[3];
     
    ParamOut.dtype   = DTYPE_FLOAT ;
    ParamOut.batch   = out.dim_size[0];
    ParamOut.channel = out.dim_size[1];
    ParamOut.height  = out.dim_size[2];
    ParamOut.width   = out.dim_size[3];

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = params[0];
    ParamConv.strideHeight   = params[1];
    ParamConv.dilationWidth  = params[2];
    ParamConv.dilationHeight = params[3];
    ParamConv.padWidth       = params[4];
    ParamConv.padHeight      = params[5];

    vednnConvolutionForward(&ParamIn,     pIn,
                     	    &ParamFilter, pFilter,
                     	    &ParamOut,    pOut, 
                     	    &ParamConv,
                     	    VEDNN_CONV_ALGORITHM_DIRECT );
    

    if( transformed_filter != NULL ) free(transformed_filter) ;

#define FMT(t) "[ " << (t).dim_size[0] << " " << (t).dim_size[1] \
    << " " << (t).dim_size[2] << " " << (t).dim_size[3] << " ]"
    PROF_END(conv2d)
      << " in=" << FMT(in)
      << " filter=" << FMT(filter)
      << " out=" << FMT(out)
      << " stride=[" << params[0] << " " << params[1] << "]"
      << " padding=[" << params[4] << " " << params[5] << "]";
#undef FMT

#ifdef _DEBUG
    fprintf(stderr, "[end] vml::conv2d\n");
#endif
    LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=0";
    return 0;
}
