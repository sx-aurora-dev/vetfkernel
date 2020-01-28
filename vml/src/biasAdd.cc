#include "vml.h"
#include "vml/types.h"
#include <cstdint>
#include <omp.h>

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/intrinsic.h"
#endif

namespace {

template<typename T>
int BiasAdd_NHWC(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(out);
  const T* pin = reinterpret_cast<const T*>(in);
  const T* pbias = reinterpret_cast<const T*>(bias);

  for (int b = 0; b < batch; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channel; ++c) {
          int i
            = b * height * width * channel
            + y * width * channel
            + x * channel;
          pout[i + c] = pin[i + c] + pbias[c];
        }
      }
    }
  }

  // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__ << " done";
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<>
inline int BiasAdd_NHWC<float>(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
  BiasAdd_NHWC_f32(out, in, bias, batch, width, height, channel) ;
}
#endif

template<typename T>
int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(out);
  const T* pin = reinterpret_cast<const T*>(in);
  const T* pbias = reinterpret_cast<const T*>(bias);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int xy = 0; xy < width*height; ++xy) {
        int i 
          = b * height * width * channel
          + c * height * width ;
        pout[i + xy] = pin[i + xy] + pbias[c];
      }
    }
  }

  // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__ << " done";
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<>
inline int BiasAdd_NCHW<float>(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
  BiasAdd_NCHW_f32(out, in, bias, batch, width, height, channel) ;
}
#endif


template <typename T>
int biasAddTmpl(vml::Tensor const& out, 
                vml::Tensor const& in, 
                vml::Tensor const& bias,
                int data_format)
{
  if (out.dims != 4 || in.dims != 4 || bias.dims != 1)
    return 1;

  int batch, height, width, channel;
  int (*func)(uint64_t, uint64_t, uint64_t, int, int , int, int);
  if (data_format == FORMAT_NHWC) {
    batch = in.dim_size[0];
    height = in.dim_size[1];
    width = in.dim_size[2];
    channel = in.dim_size[3];
    func = BiasAdd_NHWC<T>;
  } else if (data_format == FORMAT_NCHW) {
    batch = in.dim_size[0];
    height = in.dim_size[2];
    width = in.dim_size[3];
    channel = in.dim_size[1];
    func = BiasAdd_NCHW<T>;
  } else {
    return 1;
  }

  int ret = 1;
  if ( batch > 1 && batch * channel * height * width > 8192 ) {
    ret = 0 ;
#pragma omp parallel reduction(|:ret)
    {
      int64_t nthreads = omp_get_num_threads() ;
      int64_t threadid = omp_get_thread_num() ;

      int64_t chunkSize = batch / nthreads ;
      int64_t remain    = batch % nthreads ;

      int64_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
      int64_t myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

      int64_t offset    = chunkBegin * sizeof(T) *  width * height * channel ;

      if( myChunk > 0 ) {
        ret |= func(out.addr + offset, in.addr + offset, bias.addr,
                    myChunk, width, height, channel);
      }
    }
  }
  else {
    ret = func(out.addr, in.addr, bias.addr, batch, width, height, channel);
  }

  return ret;
}

} // namespace

namespace vml {

/// biasAdd
int biasAdd(Tensor const& out, Tensor const& in, Tensor const& bias,
            int data_format)
{
  if (in.dtype == DT_FLOAT) {
    return biasAddTmpl<float>(out, in, bias, data_format);
  } else if (in.dtype == DT_DOUBLE) {
    return biasAddTmpl<double>(out, in, bias, data_format);
  }
  return 1;
}

} // namespace vml

