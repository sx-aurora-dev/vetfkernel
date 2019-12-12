#undef NDEBUG
#include <iostream>
#include "ve_ops_common.h"
#include "vml/types.h"
#include "vml.h"

#include "vml/log.h"

namespace {

template <typename T>
void Operate2(vml::Tensor const& input,
	      int32_t dims,
	      int64_t asize,
	      uint32_t mult,
	      int32_t *paddings,
	      T *pad_value_p,
	      vml::Tensor const* output) {
  int i0_sz = input.dim_size[0];
  int i1_sz = input.dim_size[1];
  int i0_ps = paddings[0];       // padding low-side size for 1-dim 
  int i1_ps = paddings[2];       // padding low-side size for 2-dim 
  int o0_sz = output->dim_size[0];
  int o1_sz = output->dim_size[1];


#pragma _NEC ivdep
  for (int i = 0; i < asize; i++ ) { // initialize
    ((T *)output->addr)[i] = *pad_value_p;
  }
  for (int i0 = 0; i0 < i0_sz; i0++) {
    for (int i1 = 0; i1 < i1_sz; i1++) {

      int inidx  = i0*i1_sz + i1;
      int outidx = (i0+i0_ps)*o1_sz + (i1+i1_ps);

      ((T *)output->addr)[outidx] = ((T *)input.addr)[inidx];

#if 0 // for debug
      LOG(LOG_PARAM) << " input[" << i0 << "][" << i1 << "] = "
		     << ((T *)input.addr)[inidx];
      LOG(LOG_PARAM) << " output[" << i0+i0_ps << "][" << i1+i1_ps << "] = "
		     << ((T *)output->addr)[outidx];
#endif
    }
  }
}

template <typename T>
void Operate3(vml::Tensor const& input,
	      int32_t dims,
	      int64_t asize,
	      uint32_t mult,
	      int32_t *paddings,
	      T *pad_value_p,
	      vml::Tensor const* output) {
  int i0_sz = input.dim_size[0];
  int i1_sz = input.dim_size[1];
  int i2_sz = input.dim_size[2];
  int i0_ps = paddings[0];          // padding low-side size for 1-dim 
  int i1_ps = paddings[2];          //                           2-dim 
  int i2_ps = paddings[4];          //                           3-dim 
  int o0_sz = output->dim_size[0];
  int o1_sz = output->dim_size[1];
  int o2_sz = output->dim_size[2];


#pragma _NEC ivdep
  for (int i = 0; i < asize; i++ ) { // initialize
    ((T *)output->addr)[i] = *pad_value_p;
  }
  for (int i0 = 0; i0 < i0_sz; i0++) {
    for (int i1 = 0; i1 < i1_sz; i1++) {
      for (int i2 = 0; i2 < i2_sz; i2++) {

	int inidx  = i0*i1_sz*i2_sz + i1*i2_sz + i2;
	int outidx = (i0+i0_ps)*o1_sz*o2_sz + (i1+i1_ps)*o2_sz + i2+i2_ps;

	((T *)output->addr)[outidx] = ((T *)input.addr)[inidx];

#if 0 // for debug
	LOG(LOG_PARAM) << " input[" << i0 << "][" << i1 << "][" << i2 << "] = "
		       << ((T *)input.addr)[inidx];
	LOG(LOG_PARAM) << " output[" << i0+i0_ps << "][" << i1+i1_ps << "] = "
		       << ((T *)output->addr)[outidx];
#endif
      }
    }
  }
}


template <typename T>
void Operate(vml::Tensor const& input,
	     int32_t dims,
	     uint32_t mult,
	     int32_t *paddings,
	     T *pad_value_p,
	     vml::Tensor const* output) {

#if 0
  LOG(LOG_PARAM) << " ======== Operate ========";
  LOG(LOG_PARAM) << "         input       : " << std::hex << &input << std::dec;
  LOG(LOG_PARAM) << "         pad_value_p : " << std::hex << pad_value_p ;
  LOG(LOG_PARAM) << "         output      : " << std::hex << output << std::dec;

  for (int d = 0; d < dims*2 ; d+=2) {
    LOG(LOG_PARAM) << "  ---       paddings[" << d
		   << ", 0] = " << paddings[d  ];
    LOG(LOG_PARAM) << "  ---       paddings[" << d
		   << ", 1] = " << paddings[d+1];
  }
#endif

  int64_t asize = 1;  // number of element
  for (int i = 0; i < dims ; i++) {
    asize *= output->dim_size[i];
#if 1
    LOG(LOG_PARAM) << "           asize[" << i << "] = "
		   << output->dim_size[i];
#endif
  }
#if 0
  LOG(LOG_PARAM) << "           asize = " << asize;
#endif


#define CALLOPN(D)				\
  case D:								\
  {									\
    Operate ## D<T>(input, dims, asize, mult, paddings, pad_value_p, output); \
    break;								\
  }
  switch(dims) {
  CALLOPN(2);
  CALLOPN(3);
  default:
      LOG(LOG_ERROR) << __FUNCTION__ << ": Not supported dimension(only 2 or 3). Dim is "
		     << dims;
      break;
  }
#undef CALLOPN
}

template <typename T> int Pad(
			      int32_t fixed_dims,
			      vml::Tensor const& x_input,
                              uint32_t mult,
			      int32_t *paddings, // Tpaddingでtemplete化する
			      T *pad_value_p,
			      vml::Tensor const* y_output
			      )
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";

  int ret = 1;

  Operate<T>(x_input, fixed_dims, mult, paddings, pad_value_p, y_output);
  ret = 0;

  LOG(LOG_TRACE) << __FUNCTION__ << ": end, ret = " << ret;
  return ret;
}


int op_Pad(VEOpArgs const& args) {

  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();
	
  int ret = 1;
  uint32_t mult;
 
  {
    // 引数の取り出し(とLOG出力)
    int32_t dims = *args.arg<int32_t>(0);    // 0: fixed_dims
	
    if (args.nArguments() != 6+dims*2) {  // ★引数の数を確認する。
      LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be "
		     << 6+dims*2 << ". But " << args.nArguments();
      goto error_exit;
    }

    int64_t Ttype = *args.arg<int64_t>(1);     // 1:type of Tensor
    vml::Tensor const* x_input = args.arg<vml::Tensor>(2); 
                                               // 2: input Tensor
    int64_t Tpadtype = *args.arg<int64_t>(3);  // 3: type of padding
    void *pad_value_p = (void *)(args.arg<int32_t>(4)); // 4: padding value
    vml::Tensor const* y_output = args.arg<vml::Tensor>(5); 
                                               // 5: output Tensor
    int32_t paddings[dims*2];
    for (int d = 0; d < dims*2 ; d+=2) {
      paddings[d  ] = *args.arg<int32_t>(6+d);
      paddings[d+1] = *args.arg<int32_t>(7+d);
    }

    switch(Ttype){
    case DT_INT8:
    case DT_UINT8:
      mult = 1;
      break;
    case DT_FLOAT:
    case DT_INT32:
    case DT_UINT32:
      mult = 4;
      break;
    case DT_DOUBLE:
    case DT_INT64:
    case DT_UINT64:
      mult = 8;
      break;
    default:
      LOG(LOG_ERROR) << __FUNCTION__ << ": Type of Tensor should be DT_FLOAT(1)/DT_DOUBLE(2)/DT_INT32(3)/DT_UINT8(4)/DT_INT16(5)/DT_INT8(6)/DT_COMPLEX64(8)/DT_INT64(9). But "
		   << Ttype;
      goto error_exit;      
    }

#if 0 // for debug
#define TVALUE(TENSOR)		\
    {									\
      LOG(LOG_PARAM) << "  --- " #TENSOR " = " << std::hex << (TENSOR);	\
      LOG(LOG_PARAM) << "  ---       .dtype  = " << (TENSOR)->dtype;	\
      LOG(LOG_PARAM) << "  ---       .addr   = " << std::hex << (TENSOR)->addr; \
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
      LOG(LOG_ERROR) << __FUNCTION__ << ": Type of padding_value should be DT_INT32(3) or DT_INT64(9). But "
		   << Ttype;
    goto error_exit;
    }

    // 演算(Pad)の呼出
#define CALLPAD(T)				\
    ret = Pad<T>(dims, *x_input, mult, (int32_t*)paddings, (T *)pad_value_p, y_output);

    switch(Ttype){
    case DT_FLOAT:
      CALLPAD(float);
      break;
    case DT_DOUBLE:
      CALLPAD(double);
      break;
    case DT_UINT8:
      CALLPAD(uint8_t);
      break;
    case DT_INT16:
      CALLPAD(int16_t);
      break;
    case DT_INT8:
      CALLPAD(int8_t);
      break;
    case DT_STRING:
      LOG(LOG_ERROR) << __FUNCTION__ << ": string Tensor type is not supported on VE device.";
      break;
    case DT_COMPLEX64:
      LOG(LOG_ERROR) << __FUNCTION__ << ": complex64 Tensor type is not supported on VE device.";
      break;
    case DT_INT64:
      LOG(LOG_ERROR) << __FUNCTION__ << ": int64 Tensor type should be executed on CPU device.";
      break;
    case DT_BOOL:
      LOG(LOG_ERROR) << __FUNCTION__ << ": bool Tensor type is not supported on VE device.";
      break;
    case DT_QINT8:
    case DT_QUINT8:
    case DT_QINT32:
    case DT_QINT16:
    case DT_QUINT16:
      LOG(LOG_ERROR) << __FUNCTION__ << ": qintX Tensor type is not supported on VE device.";
      break;
    case DT_UINT16:
      CALLPAD(uint16_t);
      break;
    case DT_COMPLEX128:
      LOG(LOG_ERROR) << __FUNCTION__ << ": complex128 Tensor type is not supported on VE device.";
      break;
    case DT_HALF:
      LOG(LOG_ERROR) << __FUNCTION__ << ": half Tensor type is not supported on VE device.";
      break;
    case DT_UINT32:
      CALLPAD(uint32_t);
      break;
    case DT_UINT64:
      CALLPAD(uint64_t);
      break;
    case DT_INT32:
      LOG(LOG_ERROR) << __FUNCTION__ << ": int32 Tensor type should be executed on CPU device.";
      break;
    default:
      LOG(LOG_ERROR) << __FUNCTION__ << ": Type of Tensor is not fit. Type is "
		   << Ttype;
      break;
    }
#undef CALLPAD
  }

 error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}

} // namespace

DEFINE_KERNEL(Pad, op_Pad);
