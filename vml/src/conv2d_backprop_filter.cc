//#undef NDEBUG

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

int vml::conv2d_backprop_filter(vml::Tensor const& in,
				vml::Tensor const& filter,
				vml::Tensor const& out_bp,
				std::vector<int> params) // stride[W,H],dilation[W,H],padding[W,H],data_format
{
    LOG(LOG_TRACE) << __FUNCTION__ << ": begin";

    PROF_BEGIN(conv2d_backprop_filter);

#ifdef _DEBUG
    fprintf(stderr, "[start] vml::conv2d_backprop_filter\n");
#endif

    assert(in.dims == 4);
    assert(filter.dims == 4);
    assert(out_bp.dims == 4);
    assert(params.size() == 7);

    int data_format = params[6];
    LOG(LOG_PARAM) << __FUNCTION__
		   << ": dtype=" << in.dtype
		   << " dformat=" << (data_format==FORMAT_NCHW?"NCHW":"NHWC");

    if (data_format != FORMAT_NCHW)
      return 1;

#ifdef _DEBUG
    fprintf(stderr, "conv2d_backprop_filter: data_format=%s data_type=%d\n",
            data_format==FORMAT_NCHW?"NCHW":"*ERR*", in.dtype);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NCHW

    fprintf(stderr, "conv2d_backprop_filter: input   (N,C,H,W) = (%d,%d,%d,%d)\n",
            in.dim_size[0], in.dim_size[1], in.dim_size[2], in.dim_size[3]);
    fprintf(stderr, "conv2d_backprop_filter: filter  (N,C,H,W) = (%d,%d,%d,%d)\n",
            filter.dim_size[0], filter.dim_size[1],
	    filter.dim_size[2], filter.dim_size[3]);
    fprintf(stderr, "conv2d_backprop_filter: output  (N,C,H,W) = (%d,%d,%d,%d)\n",
            out_bp.dim_size[0], out_bp.dim_size[1],
	    out_bp.dim_size[2], out_bp.dim_size[3]);

    fprintf(stderr, "conv2d_backprop_filter: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            params[0], params[1], params[2], params[3], params[4], params[5]);

    // dump tensor
#define DUMP_TENSOR(_TENSOR)						\
    {									\
      fprintf(stderr,							\
	      "conv2d_backprop_filter: Tensor-dump : %s\n", #_TENSOR);	\
      int dim0 = _TENSOR.dim_size[0];					\
      int dim1 = _TENSOR.dim_size[1];					\
      int dim2 = _TENSOR.dim_size[2];					\
      int dim3 = _TENSOR.dim_size[3];					\
      float *addr = _TENSOR.ptr<float*>();				\
      for(int i=0; i<dim0; i++) {					\
	for(int j=0; j<dim1; j++) {					\
	  for(int k=0; k<dim2; k++) {					\
	    for(int l=0; l<dim3; l++) {					\
	      int idx  = ((i*dim1+j)*dim2+k)*dim3+l;			\
	      fprintf(stderr,						\
		      "                      [%d][%d][%d][%d] %f",	\
			i,j,k,l,addr[idx]);				\
	    fprintf(stderr, "\n");					\
	    }								\
	  }								\
	}								\
      }									\
    }

    DUMP_TENSOR(in);
    DUMP_TENSOR(filter);
#endif


    void *pIn         = in.ptr<void*>();
    void *pGradOut    = out_bp.ptr<void*>();
    void *pGradFilter = filter.ptr<void*>();

    LOG(LOG_TRACE) << __FUNCTION__ << ": pIn         = "
		   << std::hex << pIn << std::dec;
    LOG(LOG_TRACE) << __FUNCTION__ << ": pGradOut    = "
		   << std::hex << pGradOut << std::dec;
    LOG(LOG_TRACE) << __FUNCTION__ << ": pGradFilter = "
		   << std::hex << pGradFilter << std::dec;

    vednnTensorParam_t ParamIn ;
    vednnTensorParam_t ParamGradOut ;
    vednnFilterParam_t ParamGradFilter ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = in.dim_size[0];
    ParamIn.channel = in.dim_size[1];
    ParamIn.height  = in.dim_size[2];
    ParamIn.width   = in.dim_size[3];

    ParamGradOut.dtype   = DTYPE_FLOAT ;
    ParamGradOut.batch   = out_bp.dim_size[0];
    ParamGradOut.channel = out_bp.dim_size[1];
    ParamGradOut.height  = out_bp.dim_size[2];
    ParamGradOut.width   = out_bp.dim_size[3];
    
    ParamGradFilter.dtype      = DTYPE_FLOAT ;
    ParamGradFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
    ParamGradFilter.inChannel  = in.dim_size[1];
    ParamGradFilter.outChannel = out_bp.dim_size[1];
    ParamGradFilter.height     = filter.dim_size[2];
    ParamGradFilter.width      = filter.dim_size[3];

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = params[0];  // col stride    W
    ParamConv.strideHeight   = params[1];  // row stride    H
    ParamConv.dilationWidth  = params[2];  // col dilation  W
    ParamConv.dilationHeight = params[3];  // row dilation  H
    ParamConv.padWidth       = params[4];  // col padding   W
    ParamConv.padHeight      = params[5];  // row padding   H

    vednnConvolutionBackwardFilter(&ParamIn,         pIn,
                     	           &ParamGradOut,    pGradOut, 
                     	           &ParamGradFilter, pGradFilter,
                     	           &ParamConv,
                     	           VEDNN_CONV_ALGORITHM_DIRECT );

#define FMT(t) "[ " << (t).dim_size[0] << " " << (t).dim_size[1] \
    << " " << (t).dim_size[2] << " " << (t).dim_size[3] << " ]"
    PROF_END(conv2d_backprop_filter)
      << " in=" << FMT(in)
      << " filter=" << FMT(filter)
      << " out_bp=" << FMT(out_bp)
      << " stride=[" << params[0] << " " << params[1] << "]"
      << " padding=[" << params[4] << " " << params[5] << "]";
#undef FMT

#ifdef _DEBUG
    DUMP_TENSOR(out_bp);
    fprintf(stderr, "[end] vml::conv2d_backprop_filter\n");
#endif

    LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=0";
    return 0;
}
