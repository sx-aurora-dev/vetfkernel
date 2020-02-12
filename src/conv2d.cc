#undef NDEBUG

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"
#include "ve_ops_common.h"
#include "vml.h"
#include "vml/log.h"
#include "vml/types.h"

#include "vml/profile.h"

//#define _DEBUG

namespace {

  struct TensorParam {
    int w, h, c, n;
  };

int op_Conv2d(VEOpArgs const& args) {

  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();
	
  int ret = 1;

  // get args
  uint64_t input           = *args.arg<uint64_t>(0);
  uint64_t filter          = *args.arg<uint64_t>(1);
  uint64_t output          = *args.arg<uint64_t>(2);
  TensorParam in_param     = *args.arg<TensorParam>(3);
  TensorParam filter_param = *args.arg<TensorParam>(4);
  TensorParam out_param    = *args.arg<TensorParam>(5);

  uint64_t temp_in         = *args.arg<uint64_t>(6);
  uint64_t temp_filter     = *args.arg<uint64_t>(7);
  uint64_t temp_out        = *args.arg<uint64_t>(8);
  
  int row_stride           = *args.arg<int>(9);
  int col_stride           = *args.arg<int>(10);
  int row_dilation         = *args.arg<int>(11);
  int col_dilation         = *args.arg<int>(12);
  int row_padding          = *args.arg<int>(13);
  int col_padding          = *args.arg<int>(14);

  int data_format          = *args.arg<int>(15);
  int data_type            = *args.arg<int>(16);

#ifdef _DEBUG
#define TVALUE(TENSOR)		\
  {									\
    LOG(LOG_PARAM) << "  --- " #TENSOR " = " << std::hex << (TENSOR);	\
    LOG(LOG_PARAM) << "  ---       .dtype  = " << (TENSOR)->dtype;	\
    LOG(LOG_PARAM) << "  ---       .addr   = "			\
                   << std::hex << (TENSOR)->addr;			\
    LOG(LOG_PARAM) << "  ---       .dims   = " << (TENSOR)->dims;	\
    LOG(LOG_PARAM) << "  ---       .nelems = " << (TENSOR)->nelems;	\
    std::stringstream ss;						\
    ss <<  "  ---       .shape  = ";					\
    for (int i = 0; i < (TENSOR)->dims; i++) {			\
      ss << (i==0?"(":",") << (TENSOR)->dim_size[i];			\
    }									\
    ss << ")";							\
    LOG(LOG_PARAM) << ss.str();					\
  }

  // dump param info
  LOG(LOG_PARAM) << "  --- input         = "
                 << std::hex << input << std::dec;                // arg(0)
  LOG(LOG_PARAM) << "  --- filter        = "
                 << std::hex << filter << std::dec;               // arg(1)
  LOG(LOG_PARAM) << "  --- output        = "
                 << std::hex << output << std::dec;               // arg(2)

  LOG(LOG_PARAM) << "  --- temp_in       = "
                 << std::hex << temp_in << std::dec;              // arg(6)
  LOG(LOG_PARAM) << "  --- temp_filter   = "
                 << std::hex << temp_filter << std::dec;          // arg(7)
  LOG(LOG_PARAM) << "  --- temp_out      = "
                 << std::hex << temp_out << std::dec;             // arg(8)

  LOG(LOG_PARAM) << "  --- row_stride    = " << row_stride;       // arg(9)
  LOG(LOG_PARAM) << "  --- col_stride    = " << col_stride;       // arg(10)
  LOG(LOG_PARAM) << "  --- row_dilation  = " << row_dilation;     // arg(11)
  LOG(LOG_PARAM) << "  --- col_dilation  = " << col_dilation;     // arg(12)
  LOG(LOG_PARAM) << "  --- row_padding   = " << row_padding;      // arg(13)
  LOG(LOG_PARAM) << "  --- col_padding   = " << col_padding;      // arg(14)

  LOG(LOG_PARAM) << "  --- data_format   = "
		 << (data_format==FORMAT_NHWC ? "NHWC" : "NCHW"); // arg(15)
  LOG(LOG_PARAM) << "  --- data_type     = " << data_type;        // arg(16)

#endif

  // call vml
  //   prepare args
  vml::TensorDesc<4> v_in;
  vml::TensorDesc<4> v_filter;
  vml::TensorDesc<4> v_out;
  std::vector<int> v_params(7);

#define COPY(t, ptr, param) \
  t.dtype = data_type;	    \
  t.addr = ptr;		     \
  t.dims = 4;		     \
  t.dim_size[0] = param.n;   \
  t.dim_size[1] = param.c;   \
  t.dim_size[2] = param.h;   \
  t.dim_size[3] = param.w;

  // if format is NHWC, transform indata to NCHW
  if (data_format == FORMAT_NHWC){
    float * transformed_in  = reinterpret_cast<float*>(temp_in);
    float * transformed_out = reinterpret_cast<float*>(temp_out);
    float * org_in          = reinterpret_cast<float*>(input);
    if( in_param.n > 1 || in_param.c > 1 ) {
      const int N = in_param.n ;
      const int C = in_param.c ;
      const int H = in_param.h ;
      const int W = in_param.w ;
      LOG(LOG_TRACE) << __FUNCTION__ << ": transform (in)";

      // transform NHWC to NCHW
      for(int n=0; n<N ; n++) {
	for(int c=0; c<C ; c++) {
	  for(int h=0; h<H ; h++) {
	    for(int w=0; w<W ; w++) {
	      transformed_in[((n*C+c)*H+h)*W+w] = org_in[((n*H+h)*W+w)*C+c]; 
	    }
	  }
	}
      }
      COPY(v_in, reinterpret_cast<uint64_t>(transformed_in), in_param);
      COPY(v_out, reinterpret_cast<uint64_t>(transformed_out), out_param);
    } else {
      LOG(LOG_TRACE) << __FUNCTION__ << ": not transform (in)";
      COPY(v_in, input, in_param);
      COPY(v_out, output, out_param);
    }
  } else {
    LOG(LOG_TRACE) << __FUNCTION__ << ": not transform : NCHW";
    COPY(v_in, input, in_param);
    COPY(v_out, output, out_param);
  }

  // transform filter to NCHW
  float * transformed_filter = reinterpret_cast<float*>(temp_filter);
  float * org_filter         = reinterpret_cast<float*>(filter);
  if( filter_param.n > 1 || filter_param.c > 1) {
    const int N = filter_param.n ;
    const int C = filter_param.c ;
    const int H = filter_param.h ;
    const int W = filter_param.w ;
    LOG(LOG_TRACE) << __FUNCTION__ << ": transform (filter)";

    // transform HWCN to NCHW
    for(int c=0; c<C ; c++) {
      for(int n=0; n<N ; n++) {
	for(int h=0; h<H ; h++) {
	  for(int w=0; w<W ; w++) {
	    transformed_filter[((n*C+c)*H+h)*W+w]
	      = org_filter[((h*W+w)*C+c)*N+n];
	  }
	}
      }
    }
    COPY(v_filter, reinterpret_cast<uint64_t>(transformed_filter), filter_param);
  } else {
    LOG(LOG_TRACE) << __FUNCTION__ << ": not transform (filter)";
    COPY(v_filter, filter, filter_param);
  }

  v_params[0] = col_stride;
  v_params[1] = row_stride;
  v_params[2] = col_dilation;
  v_params[3] = row_dilation;
  v_params[4] = col_padding;
  v_params[5] = row_padding;
  v_params[6] = FORMAT_NCHW;   // vml and vednn only suppot NCHW
  
  LOG(LOG_TRACE) << __FUNCTION__ << ": vml call(NCHW)";
  ret = vml::conv2d(v_in, v_filter, v_out, v_params);
  LOG(LOG_TRACE) << __FUNCTION__ << ": vml return";

  // if format is NHWC, transform outdata to NHWC
  if (data_format == FORMAT_NHWC){
    LOG(LOG_TRACE) << __FUNCTION__ << ": transpose data-format";

    float * transformed_out = reinterpret_cast<float*>(temp_out);
    float * org_out         = reinterpret_cast<float*>(output);
    
    if( in_param.n > 1 || in_param.c > 1 ) {
      const int N = out_param.n ;
      const int C = out_param.c ;
      const int H = out_param.h ;
      const int W = out_param.w ;
      LOG(LOG_TRACE) << __FUNCTION__ << ": transform (out)";

      // transform NCHW to NHWC
      for(int n=0; n<N ; n++) {
	for(int c=0; c<C ; c++) {
	  for(int h=0; h<H ; h++) {
	    for(int w=0; w<W ; w++) {
	      org_out[((n*H+h)*W+w)*C+c] = transformed_out[((n*C+c)*H+h)*W+w];
	    }
	  }
	}
      }
    } else {
      LOG(LOG_TRACE) << __FUNCTION__ << ": not transform (out)";
    }
  } else {  // NCHW
    LOG(LOG_TRACE) << __FUNCTION__ << ": not transpose : NCHW";
  }

error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;

}  // op_Conv2D

}  // namespace

DEFINE_KERNEL(Conv2D, op_Conv2d);
