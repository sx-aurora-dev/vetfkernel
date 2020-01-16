#include "ve_ops_common.h"
#include "vml.h"
#include "vml/types.h"

//
// Fill
//

template <typename T>
int fill(vml::Tensor const & out, const T s)
{
  T* p = reinterpret_cast<T*>(out.addr);

  for (size_t i = 0; i < out.nelems; ++i)
    p[i] = s;

  return 0 ;
}

namespace {

int fill_functor(const VEOpArgs& args)
{
  if (args.nArguments() != 2)
    return 1;

  const int64_t dtype    = *args.arg<int64_t>(0) ;
  const vml::Tensor* out = args.arg<vml::Tensor>(1);
  const vml::Tensor* in  = args.arg<vml::Tensor>(2);	// must be scalar

  int ret = 1 ;

  if ( dtype != in->dtype || dtype != out->dtype ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " mis-match dtype of tensors.";
    return 1 ;
  }

  switch(dtype) {
  case DT_FLOAT :
    ret = fill<float>(*out, reinterpret_cast<float*>(in->addr)[0]) ;
    break ;
  case DT_DOUBLE :
    ret = fill<double>(*out, reinterpret_cast<double*>(in->addr)[0]) ;
    break ;
  case DT_INT64 :
    ret = fill<double>(*out, reinterpret_cast<int64_t*>(in->addr)[0]) ;
    break ;
  default :
    LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
    return 1 ;
  }

  return ret ;
}

} // namespace

DEFINE_KERNEL(Fill, fill_functor);
