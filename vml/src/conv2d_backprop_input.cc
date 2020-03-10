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

int vml::conv2d_backprop_input(vml::Tensor const& out_bp,
			       vml::Tensor const& filter,
			       vml::Tensor const& in_bp,
			       std::vector<int> params) // stride[W,H],dilation[W,H],padding[W,H],data_format
{
    LOG(LOG_TRACE) << __FUNCTION__ << ": begin";

    PROF_BEGIN(conv2d_backprop_input);

#ifdef _DEBUG
    fprintf(stderr, "[start] vml::conv2d_backprop_input\n");
#endif

    assert(out_bp.dims == 4);
    assert(filter.dims == 4);
    assert(in_bp.dims == 4);
    assert(params.size() == 7);

    int data_format = params[6];
    LOG(LOG_PARAM) << __FUNCTION__
		   << ": dtype=" << in_bp.dtype
		   << " dformat=" << (data_format==FORMAT_NCHW?"NCHW":"*ERR*");

    if (data_format != FORMAT_NCHW)
      return 1;

#define GET_N(_T) (_T).dim_size[0]
#define GET_C(_T) (_T).dim_size[1]
#define GET_H(_T) (_T).dim_size[2]
#define GET_W(_T) (_T).dim_size[3]


#ifdef _DEBUG
    fprintf(stderr, "conv2d_backprop_input: data_format=%s data_type=%d\n",
            data_format==FORMAT_NCHW?"NCHW":"*ERR*", in_bp.dtype);
    // assert(data_type   == 1 ) ; // float
    // assert(data_format == 1 ) ; // NCHW

    fprintf(stderr, "conv2d_backprop_input: out_bp   (N,C,H,W) = (%d,%d,%d,%d)\n",
            GET_N(out_bp), GET_C(out_bp), GET_H(out_bp), GET_W(out_bp));
    fprintf(stderr, "conv2d_backprop_input: filter   (N,C,H,W) = (%d,%d,%d,%d)\n",
            GET_N(filter), GET_C(filter), GET_H(filter), GET_W(filter));
    fprintf(stderr, "conv2d_backprop_input: in_bp (N,C,H,W) = (%d,%d,%d,%d)\n",
            GET_N(in_bp), GET_C(in_bp), GET_H(in_bp), GET_W(in_bp));
    fprintf(stderr, "conv2d_backprop_input: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            params[0], params[1], params[2], params[3], params[4], params[5]);

    // dump tensor
#define DUMP_TENSOR(_TENSOR)						\
    {									\
      fprintf(stderr,							\
	      "conv2d_backprop_input: Tensor-dump : %s\n", #_TENSOR);	\
      int dim0 = GET_N(_TENSOR); /*.dim_size[0];*/			\
      int dim1 = GET_C(_TENSOR); /*.dim_size[1];*/			\
      int dim2 = GET_H(_TENSOR); /*.dim_size[2];*/			\
      int dim3 = GET_W(_TENSOR); /*.dim_size[3];*/			\
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

    //    DUMP_TENSOR(filter);
    //    DUMP_TENSOR(out_bp);
#endif


    void *pGradOut  = out_bp.ptr<void*>();
    void *pFilter   = filter.ptr<void*>();
    void *pGradIn   = in_bp.ptr<void*>();

    LOG(LOG_TRACE) << __FUNCTION__ << ": pGradOut = "
		   << std::hex << pGradOut << std::dec;
    LOG(LOG_TRACE) << __FUNCTION__ << ": pFilter  = "
		   << std::hex << pFilter << std::dec;
    LOG(LOG_TRACE) << __FUNCTION__ << ": pGradIn  = "
		   << std::hex << pGradIn << std::dec;

    vednnTensorParam_t ParamGradOut ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamGradIn ;

    vednnConvolutionParam_t ParamConv ;

    ParamGradOut.dtype   = DTYPE_FLOAT ;
    ParamGradOut.batch   = GET_N(out_bp);
    ParamGradOut.channel = GET_C(out_bp);
    ParamGradOut.height  = GET_H(out_bp);
    ParamGradOut.width   = GET_W(out_bp);
    
    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.layout     = VEDNN_FILTER_LAYOUT_NCHW ;
    ParamFilter.inChannel  = GET_C(in_bp);
    ParamFilter.outChannel = GET_C(out_bp);
    ParamFilter.height     = GET_H(filter);
    ParamFilter.width      = GET_W(filter);

    ParamGradIn.dtype   = DTYPE_FLOAT ;
    ParamGradIn.batch   = GET_N(in_bp);
    ParamGradIn.channel = GET_C(in_bp);
    ParamGradIn.height  = GET_H(in_bp);
    ParamGradIn.width   = GET_W(in_bp);

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = params[0];  // col stride    W
    ParamConv.strideHeight   = params[1];  // row stride    H
    ParamConv.dilationWidth  = params[2];  // col dilation  W
    ParamConv.dilationHeight = params[3];  // row dilation  H
    ParamConv.padWidth       = params[4];  // col padding   W
    ParamConv.padHeight      = params[5];  // row padding   H

    LOG(LOG_TRACE) << __FUNCTION__ << ": call vednn";
    vednnConvolutionBackwardData(&ParamGradOut,  pGradOut, 
                     	         &ParamFilter,   pFilter,
                                 &ParamGradIn,   pGradIn,
                     	         &ParamConv,
                     	         VEDNN_CONV_ALGORITHM_DIRECT );
    LOG(LOG_TRACE) << __FUNCTION__ << ": ret  vednn";

#define FMT(t) "[ " << GET_N((t)) << " " << GET_C((t))	\
		    << " " << GET_H((t)) << " " << GET_W((t)) << " ]"
    PROF_END(conv2d_backprop_input)
      << " in_bp=" << FMT(in_bp)
      << " filter=" << FMT(filter)
      << " out_bp=" << FMT(out_bp)
      << " stride=[" << params[0] << " " << params[1] << "]"
      << " padding=[" << params[4] << " " << params[5] << "]";
#undef FMT

#ifdef _DEBUG
    //    DUMP_TENSOR(in_bp);
    fprintf(stderr, "[end] vml::conv2d_backprop_input\n");
#endif

#undef GET_N
#undef GET_C
#undef GET_H
#undef GET_W

    LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=0";
    return 0;
}
