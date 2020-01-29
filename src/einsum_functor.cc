#include "ve_ops_common.h"
#include "vml.h"
#include "vml/types.h"


//
// Stride functor for Einsum
//

template <typename T>
int stride1(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;

  const int64_t o0 = out.dim_size[0] ;

  const int64_t s0 = strides[0] ;

  for(int64_t d0=0; d0<o0; d0++) {
    po[d0] = pi[d0*s0] ;
  }

  return 0 ;
}

template <typename T>
int stride2(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;

  for(int64_t d0=0; d0<o0; d0++) {
    for(int64_t d1=0; d1<o1; d1++) {
      po[d0*o1+d1] = pi[(d0*s0)*i1+(d1*s1)] ;
    }
  }

  return 0 ;
}

template <typename T>
int stride3(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;

  for(int64_t d0=0; d0<o0; d0++) {
    for(int64_t d1=0; d1<o1; d1++) {
      for(int64_t d2=0; d2<o2; d2++) {
	po[(d0*o1+d1)*o2+d2] = pi[((d0*s0)*i1+(d1*s1))*i2+(d2*s2)] ;
      }
    }
  }

  return 0 ;
}

template <typename T>
int stride4(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;
  const int64_t i3 = in.dim_size[3] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;
  const int64_t o3 = out.dim_size[3] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;
  const int64_t s3 = strides[3] ;

  for(int64_t d0=0; d0<o0; d0++) {
    for(int64_t d1=0; d1<o1; d1++) {
      for(int64_t d2=0; d2<o2; d2++) {
	for(int64_t d3=0; d3<o3; d3++) {
	  po[((d0*o1+d1)*o2+d2)*o3+d3]
	     = pi[(((d0*s0)*i1+(d1*s1))*i2+(d2*s2))*i3+(d3*s3)] ;
	}
      }
    }
  }

  return 0 ;
}

template <typename T>
int stride5(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;
  const int64_t i3 = in.dim_size[3] ;
  const int64_t i4 = in.dim_size[4] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;
  const int64_t o3 = out.dim_size[3] ;
  const int64_t o4 = out.dim_size[4] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;
  const int64_t s3 = strides[3] ;
  const int64_t s4 = strides[4] ;

  for(int64_t d0=0; d0<o0; d0++) {
    for(int64_t d1=0; d1<o1; d1++) {
      for(int64_t d2=0; d2<o2; d2++) {
	for(int64_t d3=0; d3<o3; d3++) {
	  for(int64_t d4=0; d4<o4; d4++) {
	    po[(((d0*o1+d1)*o2+d2)*o3+d3)*o4+d4]
	       = pi[((((d0*s0)*i1+(d1*s1))*i2+(d2*s2))*i3+(d3*s3))*i4+(d4*s4)] ;
	  }
	}
      }
    }
  }

  return 0 ;
}

template <typename T>
int stride6(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;
  const int64_t i3 = in.dim_size[3] ;
  const int64_t i4 = in.dim_size[4] ;
  const int64_t i5 = in.dim_size[5] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;
  const int64_t o3 = out.dim_size[3] ;
  const int64_t o4 = out.dim_size[4] ;
  const int64_t o5 = out.dim_size[5] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;
  const int64_t s3 = strides[3] ;
  const int64_t s4 = strides[4] ;
  const int64_t s5 = strides[5] ;

  for(int64_t d0=0; d0<o0; d0++) {
    for(int64_t d1=0; d1<o1; d1++) {
      for(int64_t d2=0; d2<o2; d2++) {
	for(int64_t d3=0; d3<o3; d3++) {
	  for(int64_t d4=0; d4<o4; d4++) {
	    for(int64_t d5=0; d5<o5; d5++) {
	      po[((((d0*o1+d1)*o2+d2)*o3+d3)*o4+d4)*o5+d5]
		 = pi[(((((d0*s0)*i1+(d1*s1))*i2+(d2*s2))*i3+(d3*s3))*i4+(d4*s4))*i5+(d5*s5)] ;
	    }
	  }
	}
      }
    }
  }

  return 0 ;
}

template <typename T>
int stride(vml::Tensor const & out, const vml::Tensor & in, int64_t *strides, int64_t ndim)
{

  int rc = 1 ;

  switch(ndim) {
  case 1 :
    rc = stride1<T>(out,in,strides) ;
    break ;
  case 2 :
    rc = stride2<T>(out,in,strides) ;
    break ;
  case 3 :
    rc = stride3<T>(out,in,strides) ;
    break ;
  case 4 :
    rc = stride4<T>(out,in,strides) ;
    break ;
  case 5 :
    rc = stride5<T>(out,in,strides) ;
    break ;
  case 6 :
    rc = stride6<T>(out,in,strides) ;
    break ;
  default :
    break ;
  }

  return rc ;
}


namespace {

int einsum_stride_functor(const VEOpArgs& args)
{
  if (args.nArguments() <= 3)
    return 1;

  const int64_t ndim     = *args.arg<int64_t>(0);
  const vml::Tensor* in  = args.arg<vml::Tensor>(1);
  const vml::Tensor* out = args.arg<vml::Tensor>(2);

  LOG(LOG_PARAM) << "ndim = " << ndim
                 << ", in=" << *in
		 << ", out=" << *out ;

  int64_t strides[ndim] ;
  for(int i=0; i<ndim; i++) {
    strides[i] = *args.arg<int64_t>(i+3) ;
  }

  int ret = 1 ;

  if ( in->dtype != out->dtype ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " mis-match dtype of tensors.";
    return 1 ;
  }

  switch(in->dtype) {
  case DT_FLOAT :
    ret = stride<float>(*out, *in, strides, ndim) ;
    break ;
  default :
    LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
    return 1 ;
  }

  return ret ;

  return 0 ;
}

} // namespace

DEFINE_KERNEL(EinsumStride, einsum_stride_functor);



//
// Inflate functor for Einsum
//

template <typename T>
int inflate1(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;

  const int64_t o0 = out.dim_size[0] ;

  const int64_t s0 = strides[0] ;

  // zero fill
  for(int64_t i=0; i<out.nelems; i++) po[i] = T(0.) ;

  // inflate
  for(int64_t d0=0; d0<i0; d0++) {
    po[d0*s0] = pi[d0] ;
  }

  return 0 ;
}

template <typename T>
int inflate2(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;

  // zero fill
  for(int64_t i=0; i<out.nelems; i++) po[i] = T(0.) ;

  // inflate
  for(int64_t d0=0; d0<i0; d0++) {
    for(int64_t d1=0; d1<i1; d1++) {
      po[(d0*s0)*o1+(d1*s1)] = pi[d0*i1+d1] ;
    }
  }

  return 0 ;
}

template <typename T>
int inflate3(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;

  // zero fill
  for(int64_t i=0; i<out.nelems; i++) po[i] = T(0.) ;

  // inflate
  for(int64_t d0=0; d0<i0; d0++) {
    for(int64_t d1=0; d1<i1; d1++) {
      for(int64_t d2=0; d2<i2; d2++) {
	po[((d0*s0)*o1+(d1*s1))*o2+(d2*s2)]
	   = pi[(d0*i1+d1)*i2+d2] ;
      }
    }
  }

  return 0 ;
}


template <typename T>
int inflate4(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;
  const int64_t i3 = in.dim_size[3] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;
  const int64_t o3 = out.dim_size[3] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;
  const int64_t s3 = strides[3] ;

  // zero fill
  for(int64_t i=0; i<out.nelems; i++) po[i] = T(0.) ;

  // inflate
  for(int64_t d0=0; d0<i0; d0++) {
    for(int64_t d1=0; d1<i1; d1++) {
      for(int64_t d2=0; d2<i2; d2++) {
	for(int64_t d3=0; d3<i3; d3++) {
	  po[(((d0*s0)*o1+(d1*s1))*o2+(d2*s2))*o3+(d3*s3)]
	     = pi[((d0*i1+d1)*i2+d2)*i3+d3] ;
	}
      }
    }
  }

  return 0 ;
}

template <typename T>
int inflate5(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;
  const int64_t i3 = in.dim_size[3] ;
  const int64_t i4 = in.dim_size[4] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;
  const int64_t o3 = out.dim_size[3] ;
  const int64_t o4 = out.dim_size[4] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;
  const int64_t s3 = strides[3] ;
  const int64_t s4 = strides[4] ;

  // zero fill
  for(int64_t i=0; i<out.nelems; i++) po[i] = T(0.) ;

  // inflate
  for(int64_t d0=0; d0<i0; d0++) {
    for(int64_t d1=0; d1<i1; d1++) {
      for(int64_t d2=0; d2<i2; d2++) {
	for(int64_t d3=0; d3<i3; d3++) {
	  for(int64_t d4=0; d4<i4; d4++) {
	    po[((((d0*s0)*o1+(d1*s1))*o2+(d2*s2))*o3+(d3*s3))*o4+(d4*s4)]
	       = pi[(((d0*i1+d1)*i2+d2)*i3+d3)*i4+d4] ;
	  }
	}
      }
    }
  }

  return 0 ;
}

template <typename T>
int inflate6(vml::Tensor const & out, const vml::Tensor & in, int64_t* strides)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  T* po = reinterpret_cast<T*>(out.addr);

  const int64_t i0 = in.dim_size[0] ;
  const int64_t i1 = in.dim_size[1] ;
  const int64_t i2 = in.dim_size[2] ;
  const int64_t i3 = in.dim_size[3] ;
  const int64_t i4 = in.dim_size[4] ;
  const int64_t i5 = in.dim_size[5] ;

  const int64_t o0 = out.dim_size[0] ;
  const int64_t o1 = out.dim_size[1] ;
  const int64_t o2 = out.dim_size[2] ;
  const int64_t o3 = out.dim_size[3] ;
  const int64_t o4 = out.dim_size[4] ;
  const int64_t o5 = out.dim_size[5] ;

  const int64_t s0 = strides[0] ;
  const int64_t s1 = strides[1] ;
  const int64_t s2 = strides[2] ;
  const int64_t s3 = strides[3] ;
  const int64_t s4 = strides[4] ;
  const int64_t s5 = strides[5] ;

  // zero fill
  for(int64_t i=0; i<out.nelems; i++) po[i] = T(0.) ;

  // inflate
  for(int64_t d0=0; d0<i0; d0++) {
    for(int64_t d1=0; d1<i1; d1++) {
      for(int64_t d2=0; d2<i2; d2++) {
	for(int64_t d3=0; d3<i3; d3++) {
	  for(int64_t d4=0; d4<i4; d4++) {
	    for(int64_t d5=0; d5<i5; d5++) {
	      po[(((((d0*s0)*o1+(d1*s1))*o2+(d2*s2))*o3+(d3*s3))*o4+(d4*s4))*o5+(d5*s5)]
		 = pi[((((d0*i1+d1)*i2+d2)*i3+d3)*i4+d4)*i5+d5] ;
	    }
	  }
	}
      }
    }
  }

  return 0 ;
}

template <typename T>
int inflate(vml::Tensor const & out, const vml::Tensor & in, int64_t *strides, int64_t ndim)
{

  int rc = 1 ;

  switch(ndim) {
  case 1 :
    rc = inflate1<T>(out,in,strides) ;
    break ;
  case 2 :
    rc = inflate2<T>(out,in,strides) ;
    break ;
  case 3 :
    rc = inflate3<T>(out,in,strides) ;
    break ;
  case 4 :
    rc = inflate4<T>(out,in,strides) ;
    break ;
  case 5 :
    rc = inflate5<T>(out,in,strides) ;
    break ;
  case 6 :
    rc = inflate6<T>(out,in,strides) ;
    break ;
  default :
    break ;
  }

  return rc ;
}


namespace {

int einsum_inflate_functor(const VEOpArgs& args)
{
  if (args.nArguments() <= 3)
    return 1;

  const int64_t ndim     = *args.arg<int64_t>(0);
  const vml::Tensor* in  = args.arg<vml::Tensor>(1);
  const vml::Tensor* out = args.arg<vml::Tensor>(2);

  LOG(LOG_PARAM) << "ndim = " << ndim
                 << ", in=" << *in
		 << ", out=" << *out ;

  int64_t strides[ndim] ;
  for(int i=0; i<ndim; i++) {
    strides[i] = *args.arg<int64_t>(i+3) ;
  }

  int ret = 1 ;

  if ( in->dtype != out->dtype ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " mis-match dtype of tensors.";
    return 1 ;
  }

  switch(in->dtype) {
  case DT_FLOAT :
    ret = inflate<float>(*out, *in, strides, ndim) ;
    break ;
  default :
    LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
    return 1 ;
  }

  return ret ;

  return 0 ;
}

} // namespace

DEFINE_KERNEL(EinsumInflate, einsum_stride_functor);
