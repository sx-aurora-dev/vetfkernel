#undef NDEBUG
#include <iostream>
#include "ve_ops_common.h"
#include "types.h"
#include "vml.h"
#include "log.h"

namespace {

template <typename T>
void Operate1(
	      vml::Tensor const& input,
	      int32_t *paddings,
	      T pad_value,
	      vml::Tensor const* output
	      )
{
  int64_t i0_sz = input.dim_size[0];
  int64_t i0_ps = paddings[0];       // padding low-side size for 1-dim
  int64_t o0_sz = output->dim_size[0];


  for (int64_t i = 0; i < output->nelems; i++ ) { // initialize
    (reinterpret_cast<T *>(output->addr))[i] = pad_value;
  }
  for (int64_t i0 = 0; i0 < i0_sz; i0++) {

      int64_t inidx  = i0 ;
      int64_t outidx = i0+i0_ps ;

      (reinterpret_cast<T *>(output->addr))[outidx]
	= (reinterpret_cast<T *>(input.addr))[inidx];
  }
}

template <typename T>
void Operate2(
	      vml::Tensor const& input,
	      int32_t *paddings,
	      T pad_value,
	      vml::Tensor const* output
	      )
{
  int64_t i0_sz = input.dim_size[0];
  int64_t i1_sz = input.dim_size[1];
  int64_t i0_ps = paddings[0];       // padding low-side size for 1-dim
  int64_t i1_ps = paddings[2];       // padding low-side size for 2-dim
  int64_t o0_sz = output->dim_size[0];
  int64_t o1_sz = output->dim_size[1];


#pragma _NEC ivdep
  for (int64_t i = 0; i < output->nelems; i++ ) { // initialize
    (reinterpret_cast<T *>(output->addr))[i] = pad_value;
  }
  for (int64_t i0 = 0; i0 < i0_sz; i0++) {
    for (int64_t i1 = 0; i1 < i1_sz; i1++) {

      int64_t inidx  = i0*i1_sz + i1;
      int64_t outidx = (i0+i0_ps)*o1_sz + (i1+i1_ps);

      (reinterpret_cast<T *>(output->addr))[outidx]
	= (reinterpret_cast<T *>(input.addr))[inidx];

#if 0 // for debug
      LOG(LOG_PARAM) << " input[" << i0 << "][" << i1 << "] = "
		     << (reinterpret_cast<T *>(input.addr))[inidx];
      LOG(LOG_PARAM) << " output[" << i0+i0_ps << "][" << i1+i1_ps << "] = "
		     << (reinterpret_cast<T *>(output->addr))[outidx];
#endif
    }
  }
}

template <typename T>
void Operate3(
	      vml::Tensor const& input,
	      int32_t *paddings,
	      T pad_value,
	      vml::Tensor const* output
	      )
{
  int64_t i0_sz = input.dim_size[0];
  int64_t i1_sz = input.dim_size[1];
  int64_t i2_sz = input.dim_size[2];
  int64_t i0_ps = paddings[0];          // padding low-side size for 1-dim
  int64_t i1_ps = paddings[2];          //                           2-dim
  int64_t i2_ps = paddings[4];          //                           3-dim
  int64_t o0_sz = output->dim_size[0];
  int64_t o1_sz = output->dim_size[1];
  int64_t o2_sz = output->dim_size[2];


#pragma _NEC ivdep
  for (int64_t i = 0; i < output->nelems; i++ ) { // initialize
    (reinterpret_cast<T *>(output->addr))[i] = pad_value;
  }
  for (int64_t i0 = 0; i0 < i0_sz; i0++) {
    for (int64_t i1 = 0; i1 < i1_sz; i1++) {
      for (int64_t i2 = 0; i2 < i2_sz; i2++) {

	int64_t inidx  = i0*i1_sz*i2_sz + i1*i2_sz + i2;
	int64_t outidx = (i0+i0_ps)*o1_sz*o2_sz + (i1+i1_ps)*o2_sz + i2+i2_ps;

	(reinterpret_cast<T *>(output->addr))[outidx]
	  = (reinterpret_cast<T *>(input.addr))[inidx];

#if 0 // for debug
	LOG(LOG_PARAM) << " input[" << i0 << "][" << i1 << "][" << i2 << "] = "
		       << (reinterpret_cast<T *>(input.addr))[inidx];
	LOG(LOG_PARAM) << " output[" << i0+i0_ps << "][" << i1+i1_ps << "] = "
		       << (reinterpret_cast<T *>(output->addr))[outidx];
#endif
      }
    }
  }
}

template <typename T>
void Operate4(
	      vml::Tensor const& input,
	      int32_t *paddings,
	      T pad_value,
	      vml::Tensor const* output
	      )
{
  int64_t i0 = input.dim_size[0];
  int64_t i1 = input.dim_size[1];
  int64_t i2 = input.dim_size[2];
  int64_t i3 = input.dim_size[3];
  int64_t p0 = paddings[0];          // padding low-side size for 1-dim
  int64_t p1 = paddings[2];          //                           2-dim
  int64_t p2 = paddings[4];          //                           3-dim
  int64_t p3 = paddings[6];          //                           4-dim
  int64_t o0 = output->dim_size[0];
  int64_t o1 = output->dim_size[1];
  int64_t o2 = output->dim_size[2];
  int64_t o3 = output->dim_size[3];


#pragma _NEC ivdep
  for (int64_t i = 0; i < output->nelems; i++ ) { // initialize
    (reinterpret_cast<T *>(output->addr))[i] = pad_value;
  }

  for (int64_t d0 = 0; d0 < i0; d0++) {
    for (int64_t d1 = 0; d1 < i1; d1++) {
      for (int64_t d2 = 0; d2 < i2; d2++) {
	for (int64_t d3 = 0; d3 < i3; d3++) {

	  int64_t inidx  = ((d0*i1+d1)*i2+d2)*i3+d3;
	  int64_t outidx = (((d0+p0)*o1 +(d1+p1))*o2 +(d2+p2))*o3+(d3+p3);

	  (reinterpret_cast<T *>(output->addr))[outidx]
	    = (reinterpret_cast<T *>(input.addr))[inidx];
	}
      }
    }
  }
}


template <typename T>
int Operate(
	     vml::Tensor const& input,
	     int32_t *paddings,
	     T pad_value,
	     vml::Tensor const* output)
{

#if 0
  LOG(LOG_PARAM) << " ======== Operate ========";
  LOG(LOG_PARAM) << "         input       : " << std::hex << &input << std::dec;
  LOG(LOG_PARAM) << "         pad_value   : " << pad_value ;
  LOG(LOG_PARAM) << "         output      : " << std::hex << output << std::dec;

  for (int d = 0; d < input.dims*2 ; d+=2) {
    LOG(LOG_PARAM) << "  ---       paddings[" << d
		   << ", 0] = " << paddings[d  ];
    LOG(LOG_PARAM) << "  ---       paddings[" << d
		   << ", 1] = " << paddings[d+1];
}
#endif


#define CALLOPN(D)						\
  case D:							\
  {								\
    Operate ## D<T>(input, paddings, pad_value, output);	\
    return 0 ;							\
  }
  switch(input.dims) {
  CALLOPN(1);
  CALLOPN(2);
  CALLOPN(3);
  CALLOPN(4);
  default:
      LOG(LOG_ERROR) << __FUNCTION__ << ": Not supported dimension. Dim is "
		     << input.dims;
      return 1 ;
  }
}
} // namespace

//  pad for float
int vml::pad(
    vml::Tensor const& out,
    vml::Tensor const& in,
    float pad_value,          // 
    int32_t *padding          // padding range
)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "(float): begin";

  int rc = Operate<float>(in, padding, pad_value, &out);

  LOG(LOG_TRACE) << __FUNCTION__ << ": end, ret = " << 0;
  return rc ;
}

//  pad for double
int vml::pad(
    vml::Tensor const& out,
    vml::Tensor const& in,
    double pad_value,         // 
    int32_t *padding          // padding range
)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "(double): begin";

  int rc = Operate<double>(in, padding, pad_value, &out);

  LOG(LOG_TRACE) << __FUNCTION__ << ": end, ret = " << 0;
  return rc ;
}
