#include <cstring>
#include <cmath>
#include "ve_ops_common.h"
#include "vml/types.h"
#include "vml.h"

namespace {

template <typename T, typename U> int FusedBatchNorm(
        vml::Tensor const* x_input,
        vml::Tensor const* scale_input,
        vml::Tensor const* offset_input,
        vml::Tensor const* estimated_mean_input,
        vml::Tensor const* estimated_variance_input,
        vml::Tensor const* y_output,
        vml::Tensor const* batch_mean_output,
        vml::Tensor const* saved_mean_output,
        vml::Tensor const* batch_var_output,
        vml::Tensor const* saved_var_output,
        U epsilon,
        bool is_training)
{
  // NCHW

  size_t batch_size = x_input->dim_size[0];
  size_t depth = x_input->dim_size[1]; // C
  size_t size = x_input->nelems;
  size_t rest_size = size / depth;
  size_t sizeCHW = size / batch_size; // C*H*W
  size_t sizeHW = sizeCHW / depth; // H*W

  T* Y = reinterpret_cast<T*>(y_output->addr);
  T const* X = reinterpret_cast<T*>(x_input->addr);
  T const* mean;
  T const* var;
  T const* scale = reinterpret_cast<T*>(scale_input->addr);
  T const* offset = reinterpret_cast<T*>(offset_input->addr);

  if (is_training) {
    // TODO: should be U?
    T* mean = reinterpret_cast<T*>(batch_mean_output->addr);
    T* var = reinterpret_cast<T*>(batch_var_output->addr);
    T* saved_mean = reinterpret_cast<T*>(saved_mean_output->addr);
    T* saved_var = reinterpret_cast<T*>(saved_var_output->addr);

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
    T const* mean = reinterpret_cast<T*>(estimated_mean_input->addr);
    T const* var = reinterpret_cast<T*>(estimated_variance_input->addr);

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


int op_FusedBatchNorm(VEOpArgs const& args) 
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";
  //LOG(LOG_PARAM) << __FUNCTION__ << ": args.nArguments=" << args.nArguments();

  int ret = 1;

  if (args.nArguments() != 14) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 14. But "
        << args.nArguments();
    goto error_exit;
  }

  {  
    // dimensions are checked in TF
    vml::Tensor const* x_input = args.arg<vml::Tensor>(0); // 4D
    vml::Tensor const* scale_input = args.arg<vml::Tensor>(1); // 1D
    vml::Tensor const* offset_input = args.arg<vml::Tensor>(2); // 1D
    vml::Tensor const* estimated_mean_input = args.arg<vml::Tensor>(3); // 1D
    vml::Tensor const* estimated_variance_input = args.arg<vml::Tensor>(4); // 1D
    vml::Tensor const* y_output = args.arg<vml::Tensor>(5); // 4D
    vml::Tensor const* batch_mean_output = args.arg<vml::Tensor>(6); // 1D
    vml::Tensor const* saved_mean_output = args.arg<vml::Tensor>(7); // 1D
    vml::Tensor const* batch_var_output = args.arg<vml::Tensor>(8); // 1D
    vml::Tensor const* saved_var_output = args.arg<vml::Tensor>(9); // 1D
    // epsilon(10)
    bool is_training = *args.arg<bool>(11);
    int Ttype = *args.arg<int64_t>(12);
    int Utype = *args.arg<int64_t>(13);

#define PT(T) \
    LOG(LOG_PARAM) << __FUNCTION__ << ": " #T "=" << T
    PT(x_input);
    PT(scale_input);
    PT(offset_input);
    PT(estimated_mean_input);
    PT(estimated_variance_input);
    PT(y_output);
    PT(batch_mean_output);
    PT(saved_mean_output);
    PT(batch_var_output);
    PT(saved_var_output);
    LOG(LOG_PARAM) << __FUNCTION__ << ": is_training=" << is_training;
    LOG(LOG_PARAM) << __FUNCTION__ << ": Ttype=" << Ttype;
    LOG(LOG_PARAM) << __FUNCTION__ << ": Utype=" << Utype;

    if (Ttype == DT_FLOAT && Utype == DT_FLOAT) {
      float epsilon = *args.arg<float>(10);
      LOG(LOG_PARAM) << __FUNCTION__ << ": epsilon=" << epsilon;
      ret = FusedBatchNorm<float, float>(
            x_input, scale_input, offset_input,
            estimated_mean_input, estimated_variance_input,
            y_output,
            batch_mean_output, saved_mean_output,
            batch_var_output, saved_var_output,
            epsilon, is_training);
    }
  }

 error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}
} // namespace

DEFINE_KERNEL(FusedBatchNorm, op_FusedBatchNorm);

//
// FusedBatchNormGrad
//

namespace {

template <typename T, typename U> int FusedBatchNormGrad_NCHW(
        vml::Tensor const* y_backprop_input,
        vml::Tensor const* x_input,
        vml::Tensor const* scale_input,
        vml::Tensor const* mean_input,
        vml::Tensor const* variance_input,
        vml::Tensor const* x_backprop_output,
        vml::Tensor const* scale_backprop_output,
        vml::Tensor const* offset_backprop_output,
        U epsilon,
        bool is_training)
{
  T const* y_backprop = reinterpret_cast<T const*>(y_backprop_input->addr);
  T const* x = reinterpret_cast<T const*>(x_input->addr);
  T const* scale = reinterpret_cast<T const*>(scale_input->addr);
  T const* mean = reinterpret_cast<T const*>(mean_input->addr);
  T const* variance = reinterpret_cast<T const*>(variance_input->addr);

  T* x_backprop = reinterpret_cast<T*>(x_backprop_output->addr);
  T* scale_backprop = reinterpret_cast<T*>(scale_backprop_output->addr);
  T* offset_backprop = reinterpret_cast<T*>(offset_backprop_output->addr);

  size_t batch_size = x_input->dim_size[0]; // N
  const int depth = x_input->dim_size[1]; // C
  const int size = x_input->nelems;
  const int rest_size = size / depth;
  size_t sizeCHW = size / batch_size; // C*H*W
  size_t sizeHW = sizeCHW / depth; // H*W

#if 0
  LOG(LOG_PARAM) << __FUNCTION__ << ": "
      << " y_backprop=" << y_backprop
      << " x=" << x
      << " scale=" << scale
      << " mean=" << mean
      << " variance=" << variance
      << " x_backprop=" << x_backprop
      << " scale_backprop=" << scale_backprop
      << " offset_backprop=" << offset_backprop;
#endif

  if (is_training) {
    // Note: the following formulas are used to compute the gradients for
    // back propagation.
    // x_backprop = scale * rsqrt(variance + epsilon) *
    //              [y_backprop - mean(y_backprop) - (x - mean(x)) *
    //              mean(y_backprop * (x - mean(x))) / (variance + epsilon)]
    // scale_backprop = sum(y_backprop *
    //                  (x - mean(x)) * rsqrt(variance + epsilon))
    // offset_backprop = sum(y_backprop)

#if 0
    LOG(LOG_PARAM) << __FUNCTION__ << ":"
        << " depth=" << depth;
#endif
#if 1 // optimized version
#pragma omp parallel for
    for (size_t c = 0; c < depth; ++c) {

      T sum_y_backprop = T(0);
      T sum_y_backprop_x_centered = T(0) ;
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        size_t offset = b * sizeCHW + c * sizeHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          sum_y_backprop += y_backprop[offset + i];
          sum_y_backprop_x_centered +=
              y_backprop[offset + i] * (x[offset + i] - mean[c]) ;
        }
      }

      const T mean_y_backprop = sum_y_backprop / rest_size ;
      const T mean_y_backprop_x_centered = sum_y_backprop_x_centered / rest_size ;

      const T rsqrt_variance = T(1.0) /std::sqrt(variance[c] + epsilon) ;
      const T coef1 = scale[c] * rsqrt_variance ;
      const T coef2 = mean_y_backprop_x_centered / (variance[c] + epsilon)  ;
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        size_t offset = b * sizeCHW + c * sizeHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          x_backprop[offset +i] =
              coef1
	       * ( y_backprop[offset + i]
		   - mean_y_backprop
                   - (x[offset +i] - mean[c]) * coef2 ) ;
        }
      }

      offset_backprop[c] = sum_y_backprop ;
      scale_backprop[c]  = sum_y_backprop_x_centered * rsqrt_variance ;
    }
#else // original version
#pragma omp parallel for
    for (size_t c = 0; c < depth; ++c) {
      offset_backprop[c] = T(0);
      scale_backprop[c] = T(0);
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        size_t offset = b * sizeCHW + c * sizeHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          offset_backprop[c] += y_backprop[offset + i];                        // offset_backprop = sum(y_backprop)
          scale_backprop[c] += y_backprop[offset + i]                          // scale_backprop = sum(y_backprop *
              * (x[offset + i] - mean[c]) / std::sqrt(variance[c] + epsilon);  //                  (x - mean(x)) * rsqrt(variance + epsilon))
        }
      }
    }

#pragma omp parallel for
    for (size_t c = 0; c < depth; ++c) {
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        size_t offset = b * sizeCHW + c * sizeHW;
        for (size_t i = 0; i < sizeHW; ++i) {

          x_backprop[offset +i] =                                              // x_backprop =
              scale[c]                                                         //    scale
              * (T(1.0) /std::sqrt(variance[c] + epsilon))	               //  * rsqrt(variance + epsilon)
              * (                                                              //  * [
        	  y_backprop[offset + i]                                       //      y_backprop
                  - offset_backprop[c] / rest_size                             //      - mean(y_backprop)
                  - (x[offset +i] - mean[c])       	                       //      - (x - mean(x))
		     * scale_backprop[c] * std::sqrt(variance[c] + epsilon) / rest_size
		                                                               //         * mean(y_backprop * (x - mean(x)))
        	     / (variance[c] + epsilon)                                 //        / (variance + epsilon)
                 ) ;                                                           //    ]
        }
      }
    }
#endif
  } else {
    // offset_backprop  = sum(y_backprop)
    // scale_backprop = y_backprop * ((x - pop_mean) * rsqrt(pop_var + epsilon))
    // x_backprop = y_backprop * (scale * rsqrt(pop_var + epsilon))

#pragma omp parallel for
    for (size_t c = 0; c < depth; ++c) {
      offset_backprop[c] = T(0);
      scale_backprop[c] = T(0);
#pragma _NEC novector
      for (size_t b = 0; b < batch_size; ++b) {
        size_t offset = b * sizeCHW + c * sizeHW;
        for (size_t i = 0; i < sizeHW; ++i) {
          offset_backprop[c] += y_backprop[offset + i];
          scale_backprop[c] += y_backprop[offset + i]
              * (x[offset + i] - mean[c]) / std::sqrt(variance[c] + epsilon);
          x_backprop[offset + c]
              = y_backprop[offset + i]
              * scale[c] / std::sqrt(variance[c] + epsilon);
        }
      }
    }
  }

  return 0;
}

int op_FusedBatchNormGrad(VEOpArgs const& args)
{
  LOG(LOG_TRACE) << __FUNCTION__ << ": begin";

  int ret = 1;

  if (args.nArguments() != 15) {
    LOG(LOG_ERROR) << __FUNCTION__ << ": nArguments should be 15. But "
        << args.nArguments();
    goto error_exit;
  }

  {
    // dimensions are checked in TF
    // input tensors
    vml::Tensor const* y_backprop = args.arg<vml::Tensor>(0); // 4D
    vml::Tensor const* x = args.arg<vml::Tensor>(1); // 4D
    vml::Tensor const* scale = args.arg<vml::Tensor>(2); // 1D
    vml::Tensor const* saved_mean_or_pop_mean = args.arg<vml::Tensor>(3); // 1D
    vml::Tensor const* saved_maybe_inv_var_or_pop_var = args.arg<vml::Tensor>(4); // 1D
    // output tensors
    vml::Tensor const* x_backprop = args.arg<vml::Tensor>(5); // 4D
    vml::Tensor const* scale_backprop = args.arg<vml::Tensor>(6); // 1D
    vml::Tensor const* offset_backprop = args.arg<vml::Tensor>(7); // 1D
    vml::Tensor const* placeholder_1 = args.arg<vml::Tensor>(8);
    vml::Tensor const* placeholder_2 = args.arg<vml::Tensor>(9);
    // epsilon(10)
    int32_t tensor_format = *args.arg<int32_t>(11);
    bool is_training = *args.arg<bool>(12);
    int Ttype = *args.arg<int64_t>(13);
    int Utype = *args.arg<int64_t>(14);

#define PT(T) \
    LOG(LOG_PARAM) << __FUNCTION__ << ": " #T "=" << T
    PT(y_backprop);
    PT(x);
    PT(scale);
    PT(saved_mean_or_pop_mean);
    PT(saved_maybe_inv_var_or_pop_var);
    PT(x_backprop);
    PT(scale_backprop);
    PT(offset_backprop);
    PT(placeholder_1);
    PT(placeholder_2);
    LOG(LOG_PARAM) << __FUNCTION__ << ": tensor_format=" << tensor_format;
    LOG(LOG_PARAM) << __FUNCTION__ << ": is_training=" << is_training;
    LOG(LOG_PARAM) << __FUNCTION__ << ": Ttype=" << Ttype;
    LOG(LOG_PARAM) << __FUNCTION__ << ": Utype=" << Utype;

    if (Ttype == DT_FLOAT && Utype == DT_FLOAT && tensor_format == FORMAT_NCHW) {
      float epsilon = *args.arg<float>(10);
      LOG(LOG_PARAM) << __FUNCTION__ << ": epsilon=" << epsilon;

      memset(reinterpret_cast<void*>(placeholder_1->addr), 0,
	     sizeof(float) * placeholder_1->nelems);
      memset(reinterpret_cast<void*>(placeholder_2->addr), 0,
	     sizeof(float) * placeholder_2->nelems);

      ret = FusedBatchNormGrad_NCHW<float, float>(
            y_backprop, x, scale, 
            saved_mean_or_pop_mean, saved_maybe_inv_var_or_pop_var,
            x_backprop, scale_backprop, offset_backprop,
            epsilon, is_training);
    }
  }

 error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << ": end";
  return ret;
}

} // namespace

DEFINE_KERNEL(FusedBatchNormGrad, op_FusedBatchNormGrad);
