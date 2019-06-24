#include <cstdint>
#include <sstream>
#include "types.h"
#include "ve_ops_common.h"


//
// Reverse
//
template<typename T>
int reverse_d1(Tensor const *input,
               Tensor const *output,
	       const int64_t *axes )
{

  T* pi = reinterpret_cast<T*>(input->addr);
  T* po = reinterpret_cast<T*>(output->addr);

  const int64_t a0 = axes[0] ;
  const int64_t d0 = input->dim_size[0] ;
  const int64_t s0 = a0 ? -1 : 1 ;

  T* pi0 = pi + a0 * d0 - a0 ;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    po0[i0] = pi0[s0*i0] ;
  }

  return 0 ;
}

template<typename T>
int reverse_d2(Tensor const *input,
               Tensor const *output,
	       const int64_t *axes )
{

  T* pi = reinterpret_cast<T*>(input->addr);
  T* po = reinterpret_cast<T*>(output->addr);

  const int64_t a0 = axes[0] ;
  const int64_t d0 = input->dim_size[0] ;
  const int64_t s0 = a0 ? -1 : 1 ;

  const int64_t a1 = axes[1] ;
  const int64_t d1 = input->dim_size[1] ;
  const int64_t s1 = a1 ? -1 : 1 ;

  T* pi0 = pi + (a0 * d0 - a0) * d1;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    T* pi1 = pi0 + a1 * d1 - a1 ;
    T* po1 = po0 ;
    for(int64_t i1=0; i1<d1; ++i1) {
      po1[i1] = pi1[s1*i1] ;
    }
    pi0 += s0*d1;
    po0 += d1 ;
  }

  return 0 ;
}

template<typename T>
int reverse_d3(Tensor const *input,
               Tensor const *output,
	       const int64_t *axes )
{

  T* pi = reinterpret_cast<T*>(input->addr);
  T* po = reinterpret_cast<T*>(output->addr);

  const int64_t a0 = axes[0] ;
  const int64_t d0 = input->dim_size[0] ;
  const int64_t s0 = a0 ? -1 : 1 ;

  const int64_t a1 = axes[1] ;
  const int64_t d1 = input->dim_size[1] ;
  const int64_t s1 = a1 ? -1 : 1 ;

  const int64_t a2 = axes[2] ;
  const int64_t d2 = input->dim_size[2] ;
  const int64_t s2 = a2 ? -1 : 1 ;

  T* pi0 = pi + (a0 * d0 - a0) * d1 * d2 ;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    T* pi1 = pi0 + (a1 * d1 - a1) * d2 ;
    T* po1 = po0 ;
    for(int64_t i1=0; i1<d1; ++i1) {
      T* pi2 = pi1 + a2 * d2 - a2 ;
      T* po2 = po1 ;
      for(int64_t i2=0; i2<d2; ++i2) {
	po2[i2] = pi2[s2*i2] ;
      }
      pi1 += s1*d2 ;
      po1 += d2;
    }
    pi0 += s0*d1*d2;
    po0 += d1*d2 ;
  }

  return 0 ;
}

template<typename T>
int reverse(Tensor const *input,
            Tensor const *output,
	    const int64_t *axes )
{

  switch(input->dims) {
  case 1 :
    return reverse_d1<T>(input,output,axes) ;
  case 2 :
    return reverse_d2<T>(input,output,axes) ;
  case 3 :
    return reverse_d3<T>(input,output,axes) ;
  case 4 :
    return 1 ;
  case 5 :
    return 1 ;
  case 6 :
    return 1 ;
  case 7 :
    return 1 ;
  default :
    return 1 ;
  }

}

namespace {
int op_Reverse(const VEOpArgs& args)
{
    LOG(2) << __FUNCTION__ << " begin";

    if (args.nVariables() != 10)
        return 1 ;

    int ret=1;

    const Tensor *input_tensor  = args.arg<Tensor>(0) ;
    const Tensor *output_tensor = args.arg<Tensor>(1) ;

    int64_t axes[8] ;
    axes[0] = *args.arg<int64_t>(2) ;
    axes[1] = *args.arg<int64_t>(3) ;
    axes[2] = *args.arg<int64_t>(4) ;
    axes[3] = *args.arg<int64_t>(5) ;
    axes[4] = *args.arg<int64_t>(6) ;
    axes[5] = *args.arg<int64_t>(7) ;
    axes[6] = *args.arg<int64_t>(8) ;
    axes[7] = *args.arg<int64_t>(9) ;

    const int dtype  = input_tensor->dtype ;
    const int indims = input_tensor->dims ;

    if( dtype == DT_FLOAT ) {
      ret = reverse<float>(input_tensor, output_tensor, axes) ;
    }
    else if ( dtype == DT_DOUBLE ) {
      ret = reverse<double>(input_tensor, output_tensor, axes) ;
    }

    LOG(2) << __FUNCTION__ << " end. ret=" << ret;

    return ret;
}
}

DEFINE_KERNEL(Reverse, op_Reverse);
