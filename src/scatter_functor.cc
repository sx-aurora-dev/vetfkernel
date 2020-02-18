#include "ve_ops_common.h"
#include "vml.h"
#include "vml/types.h"


enum class UpdateOp : int64_t { ASSIGN, ADD, SUB, MUL, DIV, MIN, MAX };

//
// Scatter
//

template <typename T, typename Index>
int scatter_assign(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T* pi = reinterpret_cast<const T*>(updates.addr);
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;
  const int64_t inner_size = updates.nelems / nindex ;

  for(int64_t j=0; j<nindex; j++) {
    const int64_t d = idx[j] ;
    for(int64_t i=0; i<inner_size; i++) {
      po[d*inner_size+i] = pi[j*inner_size+i] ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_add(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T* pi = reinterpret_cast<const T*>(updates.addr);
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;
  const int64_t inner_size = updates.nelems / nindex ;

  for(int64_t j=0; j<nindex; j++) {
    const int64_t d = idx[j] ;
    for(int64_t i=0; i<inner_size; i++) {
      po[d*inner_size+i] += pi[j*inner_size+i] ;
    }
  }

  return 0 ;
}


template <typename T, typename Index>
int scatter_sub(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T* pi = reinterpret_cast<const T*>(updates.addr);
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;
  const int64_t inner_size = updates.nelems / nindex ;

  for(int64_t j=0; j<nindex; j++) {
    const int64_t d = idx[j] ;
    for(int64_t i=0; i<inner_size; i++) {
      po[d*inner_size+i] -= pi[j*inner_size+i] ;
    }
  }

  return 0 ;
}


template <typename T, typename Index>
int scatter_mul(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T* pi = reinterpret_cast<const T*>(updates.addr);
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;
  const int64_t inner_size = updates.nelems / nindex ;

  for(int64_t j=0; j<nindex; j++) {
    const int64_t d = idx[j] ;
    for(int64_t i=0; i<inner_size; i++) {
      po[d*inner_size+i] *= pi[j*inner_size+i] ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_div(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T* pi = reinterpret_cast<const T*>(updates.addr);
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;
  const int64_t inner_size = updates.nelems / nindex ;

  for(int64_t j=0; j<nindex; j++) {
    const int64_t d = idx[j] ;
    for(int64_t i=0; i<inner_size; i++) {
      po[d*inner_size+i] /= pi[j*inner_size+i] ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_max(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T* pi = reinterpret_cast<const T*>(updates.addr);
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;
  const int64_t inner_size = updates.nelems / nindex ;

  for(int64_t j=0; j<nindex; j++) {
    const int64_t d = idx[j] ;
    for(int64_t i=0; i<inner_size; i++) {
      if(po[d*inner_size+i] < pi[j*inner_size+i])
	po[d*inner_size+i] = pi[j*inner_size+i] ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_min(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T* pi = reinterpret_cast<const T*>(updates.addr);
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;
  const int64_t inner_size = updates.nelems / nindex ;

  for(int64_t j=0; j<nindex; j++) {
    const int64_t d = idx[j] ;
    for(int64_t i=0; i<inner_size; i++) {
      if(po[d*inner_size+i] > pi[j*inner_size+i])
	po[d*inner_size+i] = pi[j*inner_size+i] ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_handle(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices,
    const UpdateOp op
)
{

  int rc = 1 ;

  switch(op) {
  case UpdateOp::ASSIGN :
    rc =scatter_assign<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::ADD :
    rc =scatter_add<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::SUB :
    rc =scatter_sub<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::MUL :
    rc =scatter_mul<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::DIV :
    rc =scatter_div<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::MAX :
    rc =scatter_max<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::MIN :
    rc =scatter_min<T,Index>(params, updates, indices) ;
    break ;
  default :
    break ;
  }

  return rc ;
}


namespace {

int Scatter(const VEOpArgs& args)
{
  if (args.nArguments() < 4)
    return 1;

  const vml::Tensor* params  = args.arg<vml::Tensor>(0);
  const vml::Tensor* updates = args.arg<vml::Tensor>(1);
  const vml::Tensor* indices = args.arg<vml::Tensor>(2);
  const UpdateOp op = *args.arg<UpdateOp>(3) ;

  LOG(LOG_PARAM) << "params = " << *params
                 << ", updates=" << *updates
		 << ", indices=" << *indices
		 << ", op=" << static_cast<int64_t>(op) ;

  int ret = 1 ;

  if ( params->dtype != updates->dtype ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " mis-match dtype of tensors.";
    return 1 ;
  }

  switch(params->dtype) {
  case DT_FLOAT :
    if( indices->dtype == DT_INT32) {
      ret = scatter_handle<float,int32_t>(*params, *updates, *indices, op) ;
    }
    else if( indices->dtype == DT_INT64 ) {
      ret = scatter_handle<float,int64_t>(*params, *updates, *indices, op) ;
    }
    break ;
  default :
    LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
    return 1 ;
  }

  return ret ;

}

} // namespace

DEFINE_KERNEL(Scatter, Scatter);


//
// ScatterScalar
//

template <typename T, typename Index>
int scatter_scalar_assign(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T si = reinterpret_cast<const T*>(updates.addr)[0];
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;

  for(int64_t j=0; j<nindex; j++) {
      po[idx[j]] = si ;
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_scalar_add(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T si = reinterpret_cast<const T*>(updates.addr)[0];
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;

  for(int64_t j=0; j<nindex; j++) {
      po[idx[j]] += si ;
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_scalar_sub(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T si = reinterpret_cast<const T*>(updates.addr)[0];
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;

  for(int64_t j=0; j<nindex; j++) {
      po[idx[j]] -= si ;
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_scalar_mul(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T si = reinterpret_cast<const T*>(updates.addr)[0];
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;

  for(int64_t j=0; j<nindex; j++) {
      po[idx[j]] *= si ;
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_scalar_div(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T si = reinterpret_cast<const T*>(updates.addr)[0];
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;

  for(int64_t j=0; j<nindex; j++) {
      po[idx[j]] /= si ;
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_scalar_max(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T si = reinterpret_cast<const T*>(updates.addr)[0];
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;

  for(int64_t j=0; j<nindex; j++) {
    if( po[idx[j]] < si ) po[idx[j]] = si ;
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_scalar_min(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices
)
{
  const T si = reinterpret_cast<const T*>(updates.addr)[0];
  const Index* idx = reinterpret_cast<const Index*>(indices.addr);
  T* po = reinterpret_cast<T*>(params.addr);

  const int64_t nindex = indices.nelems ;

  for(int64_t j=0; j<nindex; j++) {
    if( po[idx[j]] > si ) po[idx[j]] = si ;
  }

  return 0 ;
}

template <typename T, typename Index>
int scatter_scalar_handle(
    vml::Tensor const & params,
    const vml::Tensor & updates,
    const vml::Tensor & indices,
    const UpdateOp op
)
{

  int rc = 1 ;

  switch(op) {
  case UpdateOp::ASSIGN :
    rc =scatter_scalar_assign<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::ADD :
    rc =scatter_scalar_add<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::SUB :
    rc =scatter_scalar_sub<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::MUL :
    rc =scatter_scalar_mul<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::DIV :
    rc =scatter_scalar_div<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::MAX :
    rc =scatter_scalar_max<T,Index>(params, updates, indices) ;
    break ;
  case UpdateOp::MIN :
    rc =scatter_scalar_min<T,Index>(params, updates, indices) ;
    break ;
  default :
    break ;
  }

  return rc ;
}


namespace {

int ScatterScalar(const VEOpArgs& args)
{
  if (args.nArguments() < 4)
    return 1;

  const vml::Tensor* params  = args.arg<vml::Tensor>(0);
  const vml::Tensor* update  = args.arg<vml::Tensor>(1); // scalar
  const vml::Tensor* indices = args.arg<vml::Tensor>(2);
  const UpdateOp op = *args.arg<UpdateOp>(3) ;

  LOG(LOG_PARAM) << "params = " << *params
                 << ", update=" << *update
		 << ", indices=" << *indices
		 << ", op=" << static_cast<int64_t>(op) ;

  int ret = 1 ;

  if ( params->dtype != update->dtype ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " mis-match dtype of tensors.";
    return 1 ;
  }

  switch(params->dtype) {
  case DT_FLOAT :
    if( indices->dtype == DT_INT32) {
      ret = scatter_scalar_handle<float,int32_t>(*params, *update, *indices, op) ;
    }
    else if( indices->dtype == DT_INT64 ) {
      ret = scatter_scalar_handle<float,int64_t>(*params, *update, *indices, op) ;
    }
    break ;
  default :
    LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
    return 1 ;
  }

  return ret ;

}

} // namespace

DEFINE_KERNEL(ScatterScalar, ScatterScalar);

