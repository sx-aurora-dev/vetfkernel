#include "ve_ops_common.h"
#include "vml.h"
#include "vml/types.h"



//
// BroadcastTo
//


// out.nelems = n, in.nelems = 1
template<typename T>
int broadcast_to_n1(uint64_t out, uint64_t in, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi = reinterpret_cast<const T*>(in);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi[0] ;
  }

  return 0 ;
}

// out.dim[0] = n0, in.dim[0] == n0
// out.dim[1] = n1, in.dim[1] ==  1
template <typename T>
int broadcast_to_dim2_nn_n1(uint64_t out,
                            uint64_t in,
                            size_t n0,
                            size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi = reinterpret_cast<const T*>(in);

#pragma omp parallel for
  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi[i];
    }
  }
  return 0;
}


template<typename T>
int broadcast_to_dimn(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_TRACE) << __FUNCTION__ << ": general broadcast kernel is called";

    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);

    size_t dims = X.dims;

    size_t stX[dims];
    stX[dims - 1] = 1;
#pragma _NEC novector
    for (int dim = dims - 2; dim >= 0; --dim) {
      stX[dim] = stX[dim + 1] * X.dim_size[dim + 1];
    }

    for (size_t ix = 0; ix < X.nelems; ++ix) {
      size_t tmp = ix;
      size_t iy = 0;
      size_t iz = 0;
#pragma _NEC novector
      for (size_t dim = 0; dim < dims; ++dim) {
        size_t tmp1 = tmp / stX[dim];
        iy = (iy * Y.dim_size[dim]) + tmp1 % Y.dim_size[dim];
        tmp = tmp % stX[dim];
      }
      px[ix] = py[iy] ;
    }

    return 0;
}

template <typename T>
int broadcast_to(
    vml::Tensor const & output,
    const vml::Tensor & input
)
{
  if (input.nelems == 1) {
    return broadcast_to_n1<T>(output.addr, input.addr, output.nelems ) ;
  }

  if (output.dims == 2) {
    if (output.dim_size[0] == input.dim_size[0]  &&  input.dim_size[1] == 1) {
      // (X0,X1) <-- (X0, 1)
      return broadcast_to_dim2_nn_n1<T>(output.addr, input.addr, output.dim_size[0], output.dim_size[1]);
    }
  }

  // [todo] general kernel is slow, add parameter-dedicated kernels.

  return broadcast_to_dimn<T>(output, input) ;
}

namespace {

int BroadcastTo(const VEOpArgs& args)
{
  if (args.nArguments() < 2)
    return 1;

  const vml::Tensor* output  = args.arg<vml::Tensor>(0);
  const vml::Tensor* input = args.arg<vml::Tensor>(1);

  LOG(LOG_PARAM) << "output = " << *output
                 << ", input=" << *input ;

  int ret = 1 ;

  if ( output->dtype != input->dtype ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " mis-match dtype of tensors.";
    return 1 ;
  }

  // In/Out Tensors must be reshaped to same rank at TensorFlow side.
  if ( output->dims != input->dims ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " mis-match rank of tensors.";
    return 1 ;
  }

  switch(output->dtype) {
  case DT_FLOAT :
    if( output->dtype == DT_FLOAT) {
      ret = broadcast_to<float>(*output, *input) ;
    }
    break ;
  default :
    LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
    return 1 ;
  }

  return ret ;

}

} // namespace

DEFINE_KERNEL(BroadcastTo, BroadcastTo);

