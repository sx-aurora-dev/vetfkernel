#include <cstring>
#include <cmath>
#include "ve_ops_common.h"
#include "vml/types.h"
#include "vml.h"

//#define _DEBUG


template <typename T, typename U> int Operate(
        vml::Tensor const& x_input,
        vml::Tensor const& scale_input,
        vml::Tensor const& offset_input,
        vml::Tensor const& estimated_mean_input,
        vml::Tensor const& estimated_variance_input,
        vml::Tensor const& y_output,
        vml::Tensor const& batch_mean_output,
        vml::Tensor const& saved_mean_output,
        vml::Tensor const& batch_var_output,
        vml::Tensor const& saved_var_output,
        U epsilon,
        bool is_training)
{
  // NCHW

  size_t batch_size = x_input.dim_size[0];
  size_t depth = x_input.dim_size[1]; // C
  size_t size = x_input.nelems;
  size_t rest_size = size / depth;
  size_t sizeCHW = size / batch_size; // C*H*W
  size_t sizeHW = sizeCHW / depth; // H*W

  T* Y = reinterpret_cast<T*>(y_output.addr);
  T const* X = reinterpret_cast<T*>(x_input.addr);
  T const* mean;
  T const* var;
  T const* scale = reinterpret_cast<T*>(scale_input.addr);
  T const* offset = reinterpret_cast<T*>(offset_input.addr);

  if (is_training) {
    // TODO: should be U?
    T* mean = reinterpret_cast<T*>(batch_mean_output.addr);
    T* var = reinterpret_cast<T*>(batch_var_output.addr);
    T* saved_mean = reinterpret_cast<T*>(saved_mean_output.addr);
    T* saved_var = reinterpret_cast<T*>(saved_var_output.addr);

    const int rest_size_minus_one = rest_size > 1 ? rest_size - 1 : 1;
    U rest_size_adjust
        = static_cast<U>(rest_size) / static_cast<U>(rest_size_minus_one);

#pragma omp parallel for
    for (size_t c = 0; c < depth; ++c) {
      T _mean= T(0);
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        T const* x = X + b * sizeCHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          _mean += x[c * sizeHW + i];
        }
      }
      _mean /= rest_size ;
      saved_mean[c] = mean[c] = _mean ;

      T _var = T(0) ;
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        T* y = Y + b * sizeCHW;
        T const* x = X + b * sizeCHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          T tmp = x[c * sizeHW + i] - _mean ;
          y[c * sizeHW + i] = tmp;
          _var += tmp * tmp;
        }
      }
      _var /= rest_size ;
      var[c] = _var * rest_size_adjust;
      saved_var[c] = _var ;

      const T scaling_factor = scale[c] / std::sqrt(_var+epsilon) ;
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        T* y = Y + b * sizeCHW;
        T const* x = X + b * sizeCHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          y[c * sizeHW + i] 
              = y[c * sizeHW + i] * scaling_factor
              + offset[c];
        }
      }
    }

  } else { // !is_training
    T const* mean = reinterpret_cast<T*>(estimated_mean_input.addr);
    T const* var = reinterpret_cast<T*>(estimated_variance_input.addr);

#pragma omp parallel for
    for (size_t c = 0; c < depth; ++c) {
      const T rsqrt_var = T(1) / std::sqrt(var[c]+epsilon) ;
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        T* y = Y + b * sizeCHW;
        T const* x = X + b * sizeCHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          y[c * sizeHW + i] 
              = (x[c * sizeHW + i] - mean[c]) * rsqrt_var * scale[c] + offset[c];
        }
      }
    }
  }

  return 0;
}

int vml::fused_batch_norm(vml::Tensor const& x_input,
                     vml::Tensor const& scale_input,
                     vml::Tensor const& offset_input,
                     vml::Tensor const& estimated_mean_input,
                     vml::Tensor const& estimated_variance_input,
                     vml::Tensor const& y_output,
                     vml::Tensor const& batch_mean_output,
                     vml::Tensor const& saved_mean_output,
                     vml::Tensor const& batch_var_output,
                     vml::Tensor const& saved_var_output,
                     float epsilon,
                     bool is_training)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "(float): begin";

  Operate<float, float>(x_input,
			scale_input,
			offset_input,
			estimated_mean_input,
			estimated_variance_input,
			y_output,
			batch_mean_output,
			saved_mean_output,
			batch_var_output,
			saved_var_output,
			epsilon,
			is_training);

  LOG(LOG_TRACE) << __FUNCTION__ << "(float): end, ret = " << 0;
  return 0;
}

