#undef NDEBUG
#include <iostream>
#include "ve_ops_common.h"
#include "vml/types.h"
#include "vml.h"

#include "vml/log.h"

namespace {

int op_Pad(VEOpArgs const& args) {

  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();
	
  int ret = 1;
 
  {
#define NUM_OF_FIXED_ARGS 6
    // pop param and logging
    int32_t dims = *args.arg<int32_t>(0);      // 0: fixed_dims
	
    if (args.nArguments() != NUM_OF_FIXED_ARGS+dims*2) { // check # of args
      LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be "
		     << NUM_OF_FIXED_ARGS+dims*2 << ". But "
		     << args.nArguments();
      goto error_exit;
    }

    int64_t Ttype = *args.arg<int64_t>(1);     // 1 :type of Tensor
    vml::Tensor const* x_input = args.arg<vml::Tensor>(2); 
                                               // 2 : input Tensor
    int64_t Tpadtype = *args.arg<int64_t>(3);  // 3 : type of padding
    void *pad_value_p = (void *)(args.arg<int32_t>(4));
                                               // 4 : padding value
    vml::Tensor const* y_output = args.arg<vml::Tensor>(5); 
                                               // 5 : output Tensor
    int32_t paddings[dims*2];
    for (int d = 0; d < dims*2 ; d+=2) {       // 6-: padding width
      paddings[d  ] = *args.arg<int32_t>(NUM_OF_FIXED_ARGS+d);
      paddings[d+1] = *args.arg<int32_t>(NUM_OF_FIXED_ARGS+d+1);
    }
#undef NUM_OF_FIXED_ARGS


#if 0 // for debug
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
    LOG(LOG_PARAM) << "  --- dims          = " << dims;         // arg(0)
    LOG(LOG_PARAM) << "  --- Ttype         = " << Ttype;        // arg(1)
    TVALUE(x_input);                                            // arg(2)
    LOG(LOG_PARAM) << "  --- Tpadtype      = " << Tpadtype;     // arg(3)
    LOG(LOG_PARAM) << "  --- pad_value_p   = " << std::hex
		   << pad_value_p << std::dec;                  // arg(4)
    TVALUE(y_output);                                           // arg(5)
    for (int d = 0; d < dims*2 ; d+=2) {
      LOG(LOG_PARAM) << "  ---       paddings[" << d
		     << ", 0] = " << paddings[d  ];
      LOG(LOG_PARAM) << "  ---       paddings[" << d
		     << ", 1] = " << paddings[d+1];
    }

#undef TVALUE
#endif

    if (Tpadtype != DT_INT32 && Tpadtype != DT_INT64) {
      LOG(LOG_ERROR) << __FUNCTION__ << ": Type of padding_value should be DT_INT32(3) or DT_INT64(9). But " << Ttype;
    goto error_exit;
    }

    // call pad op on vml
    switch(Ttype){
    case DT_FLOAT:
      ret = vml::pad(*y_output, *x_input,
		     *(reinterpret_cast<float*>(pad_value_p)),
		     (int32_t *)paddings);
      break;
    case DT_DOUBLE:
      ret = vml::pad(*y_output, *x_input,
		     *(reinterpret_cast<double*>(pad_value_p)),
		     (int32_t *)paddings);
      break;
    default:
      LOG(LOG_ERROR) << __FUNCTION__ << ": supported Tensor type is float or double only. Type is " << Ttype;
      break;
    }
  }

 error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}

} // namespace

DEFINE_KERNEL(Pad, op_Pad);
