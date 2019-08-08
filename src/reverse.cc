#include <cstdint>
#include <sstream>
#include "types.h"
#include "ve_ops_common.h"
#include "log.h"
#include "vml.h"


//
// Reverse
//
template<typename T>
int reverse_d1(vml::Tensor const *input,
               vml::Tensor const *output,
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
int reverse_d2(vml::Tensor const *input,
               vml::Tensor const *output,
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
int reverse_d3(vml::Tensor const *input,
               vml::Tensor const *output,
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
int reverse_d4(vml::Tensor const *input,
               vml::Tensor const *output,
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

  const int64_t a3 = axes[3] ;
  const int64_t d3 = input->dim_size[3] ;
  const int64_t s3 = a3 ? -1 : 1 ;

  T* pi0 = pi + (a0 * d0 - a0) * d1 * d2 * d3 ;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    T* pi1 = pi0 + (a1 * d1 - a1) * d2 * d3;
    T* po1 = po0 ;
    for(int64_t i1=0; i1<d1; ++i1) {
      T* pi2 = pi1 + (a2 * d2 - a2) * d3 ;
      T* po2 = po1 ;
      for(int64_t i2=0; i2<d2; ++i2) {
	T* pi3 = pi2 + (a3 * d3 - a3) ;
	T* po3 = po2 ;
	for(int64_t i3=0; i3<d3; ++i3) {
	  po3[i3] = pi3[s3*i3] ;
	}
	pi2 += s2*d3 ;
	po2 += d3;
      }
      pi1 += s1*d2*d3 ;
      po1 += d2*d3;
    }
    pi0 += s0*d1*d2*d3;
    po0 += d1*d2*d3;
  }

  return 0 ;
}


template<typename T>
int reverse_d5(vml::Tensor const *input,
               vml::Tensor const *output,
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

  const int64_t a3 = axes[3] ;
  const int64_t d3 = input->dim_size[3] ;
  const int64_t s3 = a3 ? -1 : 1 ;

  const int64_t a4 = axes[4] ;
  const int64_t d4 = input->dim_size[4] ;
  const int64_t s4 = a4 ? -1 : 1 ;

  T* pi0 = pi + (a0 * d0 - a0) * d1 * d2 * d3 * d4;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    T* pi1 = pi0 + (a1 * d1 - a1) * d2 * d3 * d4;
    T* po1 = po0 ;
    for(int64_t i1=0; i1<d1; ++i1) {
      T* pi2 = pi1 + (a2 * d2 - a2) * d3 * d4;
      T* po2 = po1 ;
      for(int64_t i2=0; i2<d2; ++i2) {
	T* pi3 = pi2 + (a3 * d3 - a3) * d4 ;
	T* po3 = po2 ;
	for(int64_t i3=0; i3<d3; ++i3) {
	  T* pi4 = pi3 + (a4 * d4 - a4) ;
	  T* po4 = po3 ;
	  for(int64_t i4=0; i4<d4; ++i4) {
	    po4[i4] = pi4[s4*i4] ;
	  }
	  pi3 += s3*d4 ;
	  po3 += d4;
	}
	pi2 += s2*d3*d4 ;
	po2 += d3*d4;
      }
      pi1 += s1*d2*d3*d4 ;
      po1 += d2*d3*d4;
    }
    pi0 += s0*d1*d2*d3*d4;
    po0 += d1*d2*d3*d4;
  }

  return 0 ;
}


template<typename T>
int reverse_d6(vml::Tensor const *input,
               vml::Tensor const *output,
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

  const int64_t a3 = axes[3] ;
  const int64_t d3 = input->dim_size[3] ;
  const int64_t s3 = a3 ? -1 : 1 ;

  const int64_t a4 = axes[4] ;
  const int64_t d4 = input->dim_size[4] ;
  const int64_t s4 = a4 ? -1 : 1 ;

  const int64_t a5 = axes[5] ;
  const int64_t d5 = input->dim_size[5] ;
  const int64_t s5 = a5 ? -1 : 1 ;

  T* pi0 = pi + (a0 * d0 - a0) * d1 * d2 * d3 * d4 * d5;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    T* pi1 = pi0 + (a1 * d1 - a1) * d2 * d3 * d4 * d5;
    T* po1 = po0 ;
    for(int64_t i1=0; i1<d1; ++i1) {
      T* pi2 = pi1 + (a2 * d2 - a2) * d3 * d4 * d5;
      T* po2 = po1 ;
      for(int64_t i2=0; i2<d2; ++i2) {
	T* pi3 = pi2 + (a3 * d3 - a3) * d4 * d5;
	T* po3 = po2 ;
	for(int64_t i3=0; i3<d3; ++i3) {
	  T* pi4 = pi3 + (a4 * d4 - a4) * d5;
	  T* po4 = po3 ;
	  for(int64_t i4=0; i4<d4; ++i4) {
	    T* pi5 = pi4 + (a5 * d5 - a5) ;
	    T* po5 = po4 ;
	    for(int64_t i5=0; i5<d5; ++i5) {
	      po5[i5] = pi5[s5*i5] ;
	    }
	    pi4 += s4*d5 ;
	    po4 += d5;
	  }
	  pi3 += s3*d4*d5 ;
	  po3 += d4*d5;
	}
	pi2 += s2*d3*d4*d5 ;
	po2 += d3*d4*d5;
      }
      pi1 += s1*d2*d3*d4*d5 ;
      po1 += d2*d3*d4*d5;
    }
    pi0 += s0*d1*d2*d3*d4*d5;
    po0 += d1*d2*d3*d4*d5;
  }

  return 0 ;
}


template<typename T>
int reverse_d7(vml::Tensor const *input,
               vml::Tensor const *output,
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

  const int64_t a3 = axes[3] ;
  const int64_t d3 = input->dim_size[3] ;
  const int64_t s3 = a3 ? -1 : 1 ;

  const int64_t a4 = axes[4] ;
  const int64_t d4 = input->dim_size[4] ;
  const int64_t s4 = a4 ? -1 : 1 ;

  const int64_t a5 = axes[5] ;
  const int64_t d5 = input->dim_size[5] ;
  const int64_t s5 = a5 ? -1 : 1 ;

  const int64_t a6 = axes[6] ;
  const int64_t d6 = input->dim_size[6] ;
  const int64_t s6 = a6 ? -1 : 1 ;

  T* pi0 = pi + (a0 * d0 - a0) * d1 * d2 * d3 * d4 * d5 * d6;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    T* pi1 = pi0 + (a1 * d1 - a1) * d2 * d3 * d4 * d5 * d6;
    T* po1 = po0 ;
    for(int64_t i1=0; i1<d1; ++i1) {
      T* pi2 = pi1 + (a2 * d2 - a2) * d3 * d4 * d5 * d6;
      T* po2 = po1 ;
      for(int64_t i2=0; i2<d2; ++i2) {
	T* pi3 = pi2 + (a3 * d3 - a3) * d4 * d5 * d6;
	T* po3 = po2 ;
	for(int64_t i3=0; i3<d3; ++i3) {
	  T* pi4 = pi3 + (a4 * d4 - a4) * d5 * d6;
	  T* po4 = po3 ;
	  for(int64_t i4=0; i4<d4; ++i4) {
	    T* pi5 = pi4 + (a5 * d5 - a5) * d6;
	    T* po5 = po4 ;
	    for(int64_t i5=0; i5<d5; ++i5) {
	      T* pi6 = pi5 + (a6 * d6 - a6) ;
	      T* po6 = po5 ;
	      for(int64_t i6=0; i6<d6; ++i6) {
		po6[i6] = pi6[s6*i6] ;
	      }
	      pi5 += s5*d6 ;
	      po5 += d6;
	    }
	    pi4 += s4*d5*d6 ;
	    po4 += d5*d6;
	  }
	  pi3 += s3*d4*d5*d6 ;
	  po3 += d4*d5*d6;
	}
	pi2 += s2*d3*d4*d5*d6 ;
	po2 += d3*d4*d5*d6;
      }
      pi1 += s1*d2*d3*d4*d5*d6 ;
      po1 += d2*d3*d4*d5*d6;
    }
    pi0 += s0*d1*d2*d3*d4*d5*d6;
    po0 += d1*d2*d3*d4*d5*d6;
  }

  return 0 ;
}


template<typename T>
int reverse_d8(vml::Tensor const *input,
               vml::Tensor const *output,
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

  const int64_t a3 = axes[3] ;
  const int64_t d3 = input->dim_size[3] ;
  const int64_t s3 = a3 ? -1 : 1 ;

  const int64_t a4 = axes[4] ;
  const int64_t d4 = input->dim_size[4] ;
  const int64_t s4 = a4 ? -1 : 1 ;

  const int64_t a5 = axes[5] ;
  const int64_t d5 = input->dim_size[5] ;
  const int64_t s5 = a5 ? -1 : 1 ;

  const int64_t a6 = axes[6] ;
  const int64_t d6 = input->dim_size[6] ;
  const int64_t s6 = a6 ? -1 : 1 ;

  const int64_t a7 = axes[7] ;
  const int64_t d7 = input->dim_size[7] ;
  const int64_t s7 = a7 ? -1 : 1 ;

  T* pi0 = pi + (a0 * d0 - a0) * d1 * d2 * d3 * d4 * d5 * d6 * d7;
  T* po0 = po ;
  for(int64_t i0=0; i0<d0; ++i0) {
    T* pi1 = pi0 + (a1 * d1 - a1) * d2 * d3 * d4 * d5 * d6 * d7 ;
    T* po1 = po0 ;
    for(int64_t i1=0; i1<d1; ++i1) {
      T* pi2 = pi1 + (a2 * d2 - a2) * d3 * d4 * d5 * d6 * d7;
      T* po2 = po1 ;
      for(int64_t i2=0; i2<d2; ++i2) {
	T* pi3 = pi2 + (a3 * d3 - a3) * d4 * d5 * d6 * d7;
	T* po3 = po2 ;
	for(int64_t i3=0; i3<d3; ++i3) {
	  T* pi4 = pi3 + (a4 * d4 - a4) * d5 * d6 * d7;
	  T* po4 = po3 ;
	  for(int64_t i4=0; i4<d4; ++i4) {
	    T* pi5 = pi4 + (a5 * d5 - a5) * d6 * d7;
	    T* po5 = po4 ;
	    for(int64_t i5=0; i5<d5; ++i5) {
	      T* pi6 = pi5 + (a6 * d6 - a6) * d7;
	      T* po6 = po5 ;
	      for(int64_t i6=0; i6<d6; ++i6) {
		T* pi7 = pi6 + (a7 * d7 - a7) ;
		T* po7 = po6 ;
		for(int64_t i7=0; i7<d7; ++i7) {
		  po7[i7] = pi7[s7*i7] ;
		}
		pi6 += s6*d7 ;
		po6 += d7;
	      }
	      pi5 += s5*d6*d7 ;
	      po5 += d6*d7;
	    }
	    pi4 += s4*d5*d6*d7 ;
	    po4 += d5*d6*d7;
	  }
	  pi3 += s3*d4*d5*d6*d7 ;
	  po3 += d4*d5*d6*d7;
	}
	pi2 += s2*d3*d4*d5*d6*d7 ;
	po2 += d3*d4*d5*d6*d7;
      }
      pi1 += s1*d2*d3*d4*d5*d6*d7 ;
      po1 += d2*d3*d4*d5*d6*d7;
    }
    pi0 += s0*d1*d2*d3*d4*d5*d6*d7;
    po0 += d1*d2*d3*d4*d5*d6*d7;
  }

  return 0 ;
}

template<typename T>
int reverse(vml::Tensor const *input,
            vml::Tensor const *output,
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
    return reverse_d4<T>(input,output,axes) ;
  case 5 :
    return reverse_d5<T>(input,output,axes) ;
  case 6 :
    return reverse_d6<T>(input,output,axes) ;
  case 7 :
    return reverse_d7<T>(input,output,axes) ;
  case 8 :
    return reverse_d8<T>(input,output,axes) ;
  default :
    return 1 ;
  }

}

namespace {
int op_Reverse(const VEOpArgs& args)
{
    LOG(LOG_TRACE) << __FUNCTION__ << " begin";

    if (args.nVariables() != 10)
        return 1 ;

    int ret=1;

    const vml::Tensor *input_tensor  = args.arg<vml::Tensor>(0) ;
    const vml::Tensor *output_tensor = args.arg<vml::Tensor>(1) ;

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

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

    if( dtype == DT_FLOAT ) {
      ret = reverse<float>(input_tensor, output_tensor, axes) ;
    }
    else if ( dtype == DT_DOUBLE ) {
      ret = reverse<double>(input_tensor, output_tensor, axes) ;
    }

    LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;

    return ret;
}
}

DEFINE_KERNEL(Reverse, op_Reverse);
