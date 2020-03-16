#undef NDEBUG

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"
#include "ve_ops_common.h"
#include "vml.h"
#include "vml/log.h"
#include "vml/types.h"
#include <vector>

#include "vml/profile.h"

#include <iostream>

//#define _DEBUG
#ifdef _DEBUG
#define PRINT(str) fprintf(stderr, str);
#else
#define PRINT(str)
#endif

namespace {

int op_Conv2d_Backprop_Filter(VEOpArgs const& args) {

  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();
	
  if (args.nArguments() != 9)
    return 1;

  int ret = 1;

  const vml::Tensor* input  = args.arg<vml::Tensor>(0);
  const vml::Tensor* out_bp = args.arg<vml::Tensor>(1);
  const vml::Tensor* filter = args.arg<vml::Tensor>(2);

  int row_stride           = *args.arg<int>(3);
  int col_stride           = *args.arg<int>(4);
  int row_dilation         = *args.arg<int>(5);
  int col_dilation         = *args.arg<int>(6);
  int row_padding          = *args.arg<int>(7);
  int col_padding          = *args.arg<int>(8);

#if 0
  int data_layout          = *args.arg<int>(9) ;	// currently support NCHW only.
  int filter_layout        = *args.arg<int>(10) ;	// currently support NCHW only.
#endif

  std::vector<int> v_params(7);

  v_params[0] = col_stride;
  v_params[1] = row_stride;
  v_params[2] = col_dilation;
  v_params[3] = row_dilation;
  v_params[4] = col_padding;
  v_params[5] = row_padding;
  v_params[6] = FORMAT_NCHW;   // vml and vednn only suppot NCHW


  PRINT("call to vml\n");
  ret = vml::conv2d_backprop_filter(*input, *out_bp, *filter, v_params);
  PRINT("return from vml\n");

error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;

}  // op_Conv2D

}  // namespace


DEFINE_KERNEL(Conv2DBackpropFilter, op_Conv2d_Backprop_Filter);
