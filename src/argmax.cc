#include "ve_ops_common.h"
#include "vml.h"
#include "vml/types.h"

#include <omp.h>

enum VEArgOpType { ARGMAX=0, ARGMIN=1 };

//
// ArgMax/ArgMin
//

namespace {

template <typename T, typename Index>
int argmax_d1(vml::Tensor const& in, vml::Tensor const& out, const int64_t d0)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);


  Index idx = 0 ;
  for(int64_t i=0; i<d0; i++) {
    if( pi[idx] < pi[i] ) idx = i ;
  }
  po[0] = idx ;

  return 0 ;
}

template <typename T, typename Index>
int argmax_d2a0(vml::Tensor const& in,
                vml::Tensor const& out,
                const int64_t d0,
		const int64_t d1)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);

#pragma _NEC novector
  for(int64_t i1=0; i1<d1; i1++) {
    Index idx = 0 ;
#pragma _NEC vector
    for(int64_t i0=0; i0<d0; i0++) {
      if( pi[idx*d1+i1] < pi[i0*d1+i1] ) idx = i0 ;
    }
    po[i1] = idx ;
  }

  return 0 ;
}

template <typename T, typename Index>
int argmax_d2a1(vml::Tensor const& in,
                vml::Tensor const& out,
                const int64_t d0,
		const int64_t d1)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);

#pragma _NEC novector
  for(int64_t i0=0; i0<d0; i0++) {
    Index idx = 0 ;
#pragma _NEC vector
    for(int64_t i1=0; i1<d1; i1++) {
      if( pi[i0*d1+idx] < pi[i0*d1+i1] ) idx = i1 ;
    }
    po[i0] = idx ;
  }

  return 0 ;
}


template <typename T, typename Index>
int argmax_d3a1(vml::Tensor const& in,
                vml::Tensor const& out,
		const int64_t d0,
		const int64_t d1,
		const int64_t d2)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);

  for(int64_t i0=0; i0<d0; i0++) {
    for(int64_t i2=0; i2<d2; i2++) {
      Index idx = 0 ;
#pragma _NEC vector
      for(int64_t i1=0; i1<d1; i1++) {
	if( pi[(i0*d1*d2+i2)+idx*d2] < pi[(i0*d1*d2+i2)+i1*d2] ) idx = i1 ;
      }
      po[i0*d2+i2] = idx ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int argmax_handler(vml::Tensor const& in,
	           vml::Tensor const& out,
                   int64_t axis)
{

  int ret = 1 ;

  if( in.dims == 1 ) {
    if( axis == 0 ) {
      ret = argmax_d1<T, Index>(in, out, in.dim_size[0]) ;
    }
  }
  else if( in.dims == 2 ) {
    if( axis == 0 ) {
      ret = argmax_d2a0<T, Index>(in, out, in.dim_size[0], in.dim_size[1]) ;
    }
    else if( axis == 1 ) {
      ret = argmax_d2a1<T, Index>(in, out, in.dim_size[0], in.dim_size[1]) ;
    }
  }
  else if(in.dims == 3 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] ;
      ret = argmax_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] ;
      const int64_t d1 = in.dim_size[2] ;
      ret = argmax_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 4 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      ret = argmax_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      ret = argmax_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 5 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      ret = argmax_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] * in.dim_size[4] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      const int64_t d2 = in.dim_size[4] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 4 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      const int64_t d1 = in.dim_size[4] ;
      ret = argmax_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 6 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      ret = argmax_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      const int64_t d2 = in.dim_size[4] * in.dim_size[5] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 4 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      const int64_t d1 = in.dim_size[4] ;
      const int64_t d2 = in.dim_size[5] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 5 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      const int64_t d1 = in.dim_size[5] ;
      ret = argmax_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 7 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmax_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] * in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      const int64_t d2 = in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 4 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      const int64_t d1 = in.dim_size[4] ;
      const int64_t d2 = in.dim_size[5] * in.dim_size[6] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 5 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      const int64_t d1 = in.dim_size[5] ;
      const int64_t d2 = in.dim_size[6] ;
      ret = argmax_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 6 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      const int64_t d1 = in.dim_size[6] ;
      ret = argmax_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }

  return ret ;
}

template <typename T, typename Index>
int argmin_d1(vml::Tensor const& in, vml::Tensor const& out, const int64_t d0)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);


  Index idx = 0 ;
  for(int64_t i=0; i<d0; i++) {
    if( pi[idx] > pi[i] ) idx = i ;
  }
  po[0] = idx ;

  return 0 ;
}

template <typename T, typename Index>
int argmin_d2a0(vml::Tensor const& in,
                vml::Tensor const& out,
                const int64_t d0,
		const int64_t d1)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);

#pragma _NEC novector
  for(int64_t i1=0; i1<d1; i1++) {
    Index idx = 0 ;
#pragma _NEC vector
    for(int64_t i0=0; i0<d0; i0++) {
      if( pi[idx*d1+i1] > pi[i0*d1+i1] ) idx = i0 ;
    }
    po[i1] = idx ;
  }

  return 0 ;
}

template <typename T, typename Index>
int argmin_d2a1(vml::Tensor const& in,
                vml::Tensor const& out,
                const int64_t d0,
		const int64_t d1)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);

#pragma _NEC novector
  for(int64_t i0=0; i0<d0; i0++) {
    Index idx = 0 ;
#pragma _NEC vector
    for(int64_t i1=0; i1<d1; i1++) {
      if( pi[i0*d1+idx] > pi[i0*d1+i1] ) idx = i1 ;
    }
    po[i0] = idx ;
  }

  return 0 ;
}


template <typename T, typename Index>
int argmin_d3a1(vml::Tensor const& in,
                vml::Tensor const& out,
		const int64_t d0,
		const int64_t d1,
		const int64_t d2)
{
  const T* pi = reinterpret_cast<const T*>(in.addr);
  Index*   po = reinterpret_cast<Index*>(out.addr);

  for(int64_t i0=0; i0<d0; i0++) {
    for(int64_t i2=0; i2<d2; i2++) {
      Index idx = 0 ;
#pragma _NEC vector
      for(int64_t i1=0; i1<d1; i1++) {
	if( pi[(i0*d1*d2+i2)+idx*d2] > pi[(i0*d1*d2+i2)+i1*d2] ) idx = i1 ;
      }
      po[i0*d2+i2] = idx ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int argmin_handler(vml::Tensor const& in,
	           vml::Tensor const& out,
                   int64_t axis)
{
  int ret = 1 ;

  if( in.dims == 1 ) {
    if( axis == 0 ) {
      ret = argmin_d1<T, Index>(in, out, in.dim_size[0]) ;
    }
  }
  else if( in.dims == 2 ) {
    if( axis == 0 ) {
      ret = argmin_d2a0<T, Index>(in, out, in.dim_size[0], in.dim_size[1]) ;
    }
    else if( axis == 1 ) {
      ret = argmin_d2a1<T, Index>(in, out, in.dim_size[0], in.dim_size[1]) ;
    }
  }
  else if(in.dims == 3 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] ;
      ret = argmin_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] ;
      const int64_t d1 = in.dim_size[2] ;
      ret = argmin_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 4 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      ret = argmin_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      ret = argmin_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 5 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      ret = argmin_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] * in.dim_size[4] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      const int64_t d2 = in.dim_size[4] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 4 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      const int64_t d1 = in.dim_size[4] ;
      ret = argmin_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 6 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      ret = argmin_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      const int64_t d2 = in.dim_size[4] * in.dim_size[5] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 4 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      const int64_t d1 = in.dim_size[4] ;
      const int64_t d2 = in.dim_size[5] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 5 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      const int64_t d1 = in.dim_size[5] ;
      ret = argmin_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }
  else if(in.dims == 7 ) {
    if( axis == 0 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmin_d2a0<T, Index>(in, out, d0, d1) ;
    }
    else if( axis == 1 ) {
      const int64_t d0 = in.dim_size[0] ;
      const int64_t d1 = in.dim_size[1] ;
      const int64_t d2 = in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 2 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1];
      const int64_t d1 = in.dim_size[2] ;
      const int64_t d2 = in.dim_size[3] * in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 3 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] ;
      const int64_t d1 = in.dim_size[3] ;
      const int64_t d2 = in.dim_size[4] * in.dim_size[5] * in.dim_size[6] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 4 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] ;
      const int64_t d1 = in.dim_size[4] ;
      const int64_t d2 = in.dim_size[5] * in.dim_size[6] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 5 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] ;
      const int64_t d1 = in.dim_size[5] ;
      const int64_t d2 = in.dim_size[6] ;
      ret = argmin_d3a1<T, Index>(in, out, d0, d1, d2) ;
    }
    else if( axis == 6 ) {
      const int64_t d0 = in.dim_size[0] * in.dim_size[1] * in.dim_size[2] * in.dim_size[3] * in.dim_size[4] * in.dim_size[5] ;
      const int64_t d1 = in.dim_size[6] ;
      ret = argmin_d2a1<T, Index>(in, out, d0, d1) ;
    }
  }

  return ret ;
}



int argmax(vml::Tensor const& in, vml::Tensor const& out, const int64_t axis)
{
  int ret = 1 ;

  LOG(LOG_PARAM) << __FUNCTION__ << ": in="   << in ;
  LOG(LOG_PARAM) << __FUNCTION__ << ": out="  << out ;
  LOG(LOG_PARAM) << __FUNCTION__ << ": axis=" << axis ;

  if( in.dtype == DT_FLOAT ) {
    if( out.dtype == DT_INT64 ) {
      ret = argmax_handler<float,int64_t>(in, out, axis) ;
    }
    else if ( out.dtype == DT_INT32 ) {
      ret = argmax_handler<float,int32_t>(in, out, axis) ;
    }
  }

  return ret ;
}

int argmin(vml::Tensor const& in, vml::Tensor const& out, const int64_t axis)
{
  int ret = 1 ;

  LOG(LOG_PARAM) << __FUNCTION__ << ": in="   << in ;
  LOG(LOG_PARAM) << __FUNCTION__ << ": out="  << out ;
  LOG(LOG_PARAM) << __FUNCTION__ << ": axis=" << axis ;

  if( in.dtype == DT_FLOAT ) {
    if( out.dtype == DT_INT64 ) {
      ret = argmin_handler<float,int64_t>(in, out, axis) ;
    }
    else if ( out.dtype == DT_INT32 ) {
      ret = argmin_handler<float,int32_t>(in, out, axis) ;
    }
  }

  return ret ;
}


} ;

int Arg(VEOpArgs const& args)
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  //LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();

  int ret = 1;

  if (args.nArguments() != 4) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 4. But "
        << args.nArguments();
    goto error_exit;
  }

  {
    vml::Tensor const* input  = args.arg<vml::Tensor>(0);
    vml::Tensor const* output = args.arg<vml::Tensor>(1);

    const int64_t axis   = *args.arg<int64_t>(2) ;
    const int64_t op     = *args.arg<int64_t>(3) ;

    if( op == ARGMAX )
      ret = argmax(*input, *output, axis) ;
    else if( op == ARGMIN )
      ret = argmin(*input, *output, axis) ;
  }

error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}


DEFINE_KERNEL(Arg, Arg) ;
