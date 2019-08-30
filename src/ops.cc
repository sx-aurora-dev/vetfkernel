#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <complex.h>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

#include "vednn.h"

#ifdef LIBVETF_INTRINSIC
#include "libvetfkernel.h"
#endif

#define ADD_
#include <cblas_f77.h>
#undef ADD_

REGISTER_KERNEL("Fill", "op_fill");
REGISTER_KERNEL("ZerosLike", "op_ZerosLike");
REGISTER_KERNEL("AddN", "op_AddN");
REGISTER_KERNEL("BiasAdd", "op_BiasAdd");
REGISTER_KERNEL("BiasAddGrad", "op_BiasAddGrad");
REGISTER_KERNEL("Relu", "op_Relu");
REGISTER_KERNEL("ReluGrad", "op_ReluGrad");
REGISTER_KERNEL("Snapshot", "op_Snapshot")
REGISTER_KERNEL("Transpose", "op_Transpose");
REGISTER_KERNEL("MatMul", "op_MatMul");
REGISTER_KERNEL("Pack", "op_Pack");
REGISTER_KERNEL("Slice", "op_Slice");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << (l1) << " expected but " << (l0); \
      return 1; \
  }

extern "C" {
  int op_fill(const void* arg, size_t len);
  int op_ZerosLike(const void* arg, size_t len);
  int op_AddN(const void* arg, size_t len);
  int op_BiasAdd(const void* arg, size_t len);
  int op_BiasAddGrad(const void* arg, size_t len);
  int op_Relu(const void* arg, size_t len);
  int op_ReluGrad(const void* arg, size_t len);
  int op_Snapshot(const void* arg, size_t len);
  int op_Transpose(const void* arg, size_t len);
  int op_MatMul(const void* arg, size_t len);
  int op_Pack(const void* arg, size_t len);
  int op_Slice(const void* arg, size_t len);
}

//
// Fill
//

namespace {
template <typename T>
  int fill(const void* args, size_t len)
  {
    // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__;
    struct Args {
      int data_type;
      uint64_t in;
      uint64_t out;
      size_t num_elems;
    } const* p;

    if (len != sizeof(*p)) {
      LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << sizeof(*p) << " expected but " << len;
      return 1;
    }

    p = reinterpret_cast<const Args*>(args);
#if 0
    fprintf(stderr, "%s: data_type=%d out=%016lx num_elems=%lu\n",
          __FUNCTION__, p->data_type, p->out, p->num_elems);
#endif

    const T* in = (T*)p->in;
    T* out = (T*)p->out;

    // LOG(LOG_DETAIL) << __FUNCTION__ << ": value=" << *in;

    for (size_t i = 0; i < p->num_elems; ++i)
      out[i] = *in;

    // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__ << ": done";
    return 0;
  }
}

#define DATA_TYPE(p) *(int*)(p)

int op_fill(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  int dtype = DATA_TYPE(args);
  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

  int ret = 1;
  if (dtype == DT_FLOAT) {
    ret = fill<float>(args, len);
  }
  else if (dtype == DT_DOUBLE) {
    ret = fill<double>(args, len);
  }
  else if (dtype == DT_INT64) {
    ret = fill<int64_t>(args, len);
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

//
// ZerosLike
//

namespace {
template <typename T>
  int zeroslike(const void* args, size_t len)
  {
    // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__;
    struct Args {
      int data_type;
      uint64_t out;
      size_t num_elems;
    } const* p;

    if (len != sizeof(*p)) {
      LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << sizeof(*p) << " expected but " << len;
      return 1;
    }

    p = reinterpret_cast<const Args*>(args);

    T* out = (T*)p->out;

    // LOG(LOG_DETAIL) << __FUNCTION__ ;

    for (size_t i = 0; i < p->num_elems; ++i)
      out[i] = T(0.);

    // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__ << ": done";
    return 0;
  }
}

#define DATA_TYPE(p) *(int*)(p)

int op_ZerosLike(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  int dtype = DATA_TYPE(args);
  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

  int ret = 1;
  if (dtype == DT_FLOAT) {
    ret = zeroslike<float>(args, len);
  } else if (dtype == DT_DOUBLE) {
    ret = zeroslike<double>(args, len);
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

namespace {
template <typename T>
int AddNOp(T* out, T** in, size_t num_elems, size_t num_inputs)
{
  switch( num_inputs ) {
  case 0 :
     memset(out, 0, sizeof(T) * num_elems);
     break ;
  case 1 :
     for (size_t i = 0; i < num_elems; ++i) {
       out[i] = in[0][i];
     }
     break ;
  case 2 :
     for (size_t i = 0; i < num_elems; ++i) {
       out[i] = in[0][i] + in[1][i] ;
     }
     break ;
  default :
     for (size_t i = 0; i < num_elems; ++i) {
       out[i] = in[0][i];
     }
     for (size_t j = 1; j < num_inputs; ++j) {
       for (size_t i = 0; i < num_elems; ++i) {
         out[i] += in[j][i];
       }
     }
     break ;
  }

  return 0;
}
};

int op_AddN(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int output_type;
    uint64_t out;
    size_t num_elems;
    size_t num_inputs;
    uint64_t in[1];	// variable length
  } const* p;

  int ret = 1;

  p = reinterpret_cast<const Args*>(args);

  size_t expectedSize = sizeof(struct Args) + sizeof(uint64_t) * (p->num_inputs - 1) ;
  if (len != expectedSize) {
      LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << expectedSize << " expected but " << len;
      goto error_exit;
  }

  LOG(LOG_PARAM) << __FUNCTION__
	 << ": dtype=" << p->output_type
	 << " num_elems=" << p->num_elems << " num_inputs=" << p->num_inputs;

  if (p->output_type == DT_FLOAT) {
    ret = AddNOp<float>((float*)p->out, (float**)p->in, p->num_elems, p->num_inputs);
  }

 error_exit:
  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

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
};

template<typename T>
int op_BiasAdd(const void* args, size_t len)
{
  // LOG(LOG_DETAIL) << __FUNCTION__ << " begin";
  struct Args {
    int dtype;
    int data_format;
    uint64_t in;
    uint64_t bias;
    uint64_t out;
    int batch;
    int width;
    int height;
    int channel;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));

  p = reinterpret_cast<const Args*>(args);

  //LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " dformat=" << p->data_format;

  int ret = 1 ;

  if (p->data_format == FORMAT_NHWC) {
    if ( p->batch > 1 && p->batch * p->channel * p->height * p->width > 8192 ) {
      ret = 0 ;
#pragma omp parallel reduction(|:ret)
      {
	int64_t nthreads = omp_get_num_threads() ;
	int64_t threadid = omp_get_thread_num() ;

	int64_t chunkSize = p->batch / nthreads ;
	int64_t remain    = p->batch % nthreads ;

	int64_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
	int64_t myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

	int64_t offset    = chunkBegin * sizeof(T) *  p->width * p->height * p->channel ;

	if( myChunk > 0 ) {
	  ret |= BiasAdd_NHWC<T>(p->out+offset, p->in+offset, p->bias, myChunk, p->width, p->height, p->channel);
	}
      }
    }
    else {
      ret = BiasAdd_NHWC<T>(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
    }

  } else if (p->data_format == FORMAT_NCHW) {
    if ( p->batch > 1 && p->batch * p->channel * p->height * p->width > 8192 ) {
      ret = 0 ;
#pragma omp parallel reduction(|:ret)
      {
	int64_t nthreads = omp_get_num_threads() ;
	int64_t threadid = omp_get_thread_num() ;

	int64_t chunkSize = p->batch / nthreads ;
	int64_t remain    = p->batch % nthreads ;

	int64_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
	int64_t myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

	int64_t offset    = chunkBegin * sizeof(T) *  p->width * p->height * p->channel ;

	if( myChunk > 0 ) {
	  ret |= BiasAdd_NCHW<T>(p->out+offset, p->in+offset, p->bias, myChunk, p->width, p->height, p->channel);
	}
      }
    }
    else {
      ret = BiasAdd_NCHW<T>(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
    }
  }

  // LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

int op_BiasAdd(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";
  struct Args {
    int dtype;
    int data_format;
    uint64_t in;
    uint64_t bias;
    uint64_t out;
    int batch;
    int width;
    int height;
    int channel;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));

  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " dformat=" << p->data_format;

  int ret = 1 ;

  if (p->dtype == DT_FLOAT) {
    ret = op_BiasAdd<float>(args, len);
  } else if (p->dtype == DT_DOUBLE) {
    ret = op_BiasAdd<double>(args, len);
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

#ifndef LIBVETF_INTRINSIC
namespace {

template<typename T>
int BiasAddGrad_NHWC(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(output);
  const T* pin = reinterpret_cast<const T*>(output_backprop);

  memset(pout, 0, sizeof(T) * channel);

  for (int b = 0; b < batch; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channel; ++c) {
          int i
            = b * height * width * channel
            + y * width * channel
            + x * channel;
          pout[c] += pin[i + c];
        }
      }
    }
  }

  // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__ << " done";
  return 0;
}


template<typename T>
int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(output);
  const T* pin = reinterpret_cast<const T*>(output_backprop);

  memset(pout, 0, sizeof(T) * channel);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int i = 0; i < width * height; ++i) {
        pout[c] += pin[b * channel * height * width + c * height * width + i];
      }
    }
  }

  // LOG(LOG_DETAIL) << __PRETTY_FUNCTION__ << ": done";
  return 0;
}

};
#endif
int op_BiasAddGrad(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args{
    int dtype;
    int data_format;
    uint64_t output_backprop;
    uint64_t output;
    int batch;
    int width;
    int height;
    int channel;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));

  p = reinterpret_cast<const Args*>(args);

#if 0
  fprintf(stderr, "%s dtype=%d data_format=%d batch=%d width=%d height=%d channel=%d\n", 
          __FUNCTION__, p->dtype, p->data_format, p->batch, p->width, p->height, p->channel);
#endif

  int ret = 1;

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " dformat=" << p->data_format;

#ifndef LIBVETF_INTRINSIC
  if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NHWC) {
    ret = BiasAddGrad_NHWC<float>(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  } else if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NCHW) {
    ret = BiasAddGrad_NCHW<float>(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  }
#else
  if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NHWC) {
    ret = BiasAddGrad_NHWC(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  } else if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NCHW) {
    ret =  BiasAddGrad_NCHW(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  }
#endif

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

//
// Relu and ReluGrad
//

int op_Relu(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    uint64_t in;
    uint64_t out;
    uint64_t num_elems;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int dtype = p->dtype;
  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

  int ret = 1;

  if (dtype == DT_FLOAT) {
    ret = vednnActivationForward(VEDNN_ACTIVATION_RELU, (void*)(p->in), (void*)(p->out), p->num_elems);
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

int op_ReluGrad(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    uint64_t g;
    uint64_t a;
    uint64_t output;
    uint64_t num_elems;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int dtype = p->dtype;
  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    ret = vednnActivationBackward(VEDNN_ACTIVATION_RELU, (void*)(p->g), (void*)(p->a), (void*)(p->output), p->num_elems);
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}


int op_Snapshot(const void* arg, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Arg {
    uint64_t dst;
    uint64_t src;
    size_t size;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Arg));
  p = reinterpret_cast<const Arg*>(arg);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dst=" << p->dst << " src=" << p->src << " size=" << p->size; 

  memcpy(reinterpret_cast<void*>(p->dst),
         reinterpret_cast<const void*>(p->src),
         p->size);

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=0";
  return 0;
}

//
// Transpose
//

namespace {


template<typename Tin, typename Tout = Tin>
int notranspose_copy(uint64_t out, uint64_t in, const int64_t nelems )
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = pi[i];
  }
  return 0;
}

template<typename Tin, typename Tout = Tin>
int transpose2_10(uint64_t out, uint64_t in, const int32_t* dim_size)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  const uint64_t d0 = dim_size[0] ;
  const uint64_t d1 = dim_size[1] ;

  for (int64_t i1 = 0; i1 < d1; ++i1) {
    for (int64_t i0 = 0; i0 < d0; ++i0) {
      po[i1*d0+i0] = pi[i0*d1+i1];
    }
  }

  return 0;
}

template<typename Tin, typename Tout = Tin>
int transpose3_102(uint64_t out, uint64_t in, const int32_t* dim_size)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  const uint64_t d0 = dim_size[0] ;
  const uint64_t d1 = dim_size[1] ;
  const uint64_t d2 = dim_size[2] ;


  if( d0 > d1 ) {
#pragma omp parallel for
    for (int64_t i0 = 0; i0 < d0; ++i0) {
      for (int64_t i1 = 0; i1 < d1; ++i1) {
	for (int64_t i2 = 0; i2 < d2; ++i2) {
	  po[(i1*d0+i0)*d2+i2] = pi[(i0*d1+i1)*d2+i2];
	}
      }
    }
  }
  else {
#pragma omp parallel for
    for (int64_t i1 = 0; i1 < d1; ++i1) {
      for (int64_t i0 = 0; i0 < d0; ++i0) {
	for (int64_t i2 = 0; i2 < d2; ++i2) {
	  po[(i1*d0+i0)*d2+i2] = pi[(i0*d1+i1)*d2+i2];
	}
      }
    }
  }

  return 0;
}

template<typename Tin, typename Tout = Tin>
int transpose3(uint64_t out, uint64_t in, const int32_t* dim_size, const int32_t* perm)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;

  int64_t stride_out[3] ;
  stride_out[perm[2]] = 1 ;
  for(int64_t d=1; d>=0; --d) {
    stride_out[perm[d]] = stride_out[perm[d+1]] * dim_size[perm[d+1]] ;
  }
  const int64_t s0 = stride_out[0] ;
  const int64_t s1 = stride_out[1] ;
  const int64_t s2 = stride_out[2] ;

  for (int64_t i0 = 0; i0 < d0; ++i0) {
    for (int64_t i1 = 0; i1 < d1; ++i1) {
      for (int64_t i2 = 0; i2 < d2; ++i2) {
	po[i0*s0+i1*s1+i2*s2] = pi[(i0*d1+i1)*d2+i2];
      }
    }
  }

  return 0;
}

template<typename Tin, typename Tout = Tin>
int transpose4_0231(uint64_t out, uint64_t in, const int32_t* dim_size)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  uint64_t si2 = dim_size[3];
  uint64_t si1 = si2 * dim_size[2];
  uint64_t si0 = si1 * dim_size[1];

  uint64_t so2 = dim_size[1];
  uint64_t so1 = so2 * dim_size[3];
  uint64_t so0 = so1 * dim_size[2];

  for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
    for (int64_t i1 = 0; i1 < dim_size[2]; ++i1) {
      for (int64_t i2 = 0; i2 < dim_size[3]; ++i2) {
	for (int64_t i3 = 0; i3 < dim_size[1]; ++i3) {
	  po[i0 * so0 + i1 * so1 + i2 * so2 + i3]
	    = pi[i0 * si0 + i1 * si2 + i2 + i3 * si1];
	}
      }
    }
  }

  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<>
inline  int transpose4_0231<float>(uint64_t out, uint64_t in, const int32_t* dim_size) {
  return transpose4_0231_f32(out, in, dim_size) ;
}
#endif

template<typename Tin, typename Tout = Tin>
int transpose4_0312(uint64_t out, uint64_t in, const int32_t* dim_size)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  uint64_t si2 = dim_size[3];
  uint64_t si1 = si2 * dim_size[2];
  uint64_t si0 = si1 * dim_size[1];

  uint64_t so2 = dim_size[2];
  uint64_t so1 = so2 * dim_size[1];
  uint64_t so0 = so1 * dim_size[3];

  for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
    for (int64_t i1 = 0; i1 < dim_size[3]; ++i1) {
      for (int64_t i2 = 0; i2 < dim_size[1]; ++i2) {
	for (int64_t i3 = 0; i3 < dim_size[2]; ++i3) {
	  po[i0 * so0 + i1 * so1 + i2 * so2 + i3]
	    = pi[i0 * si0 + i1 + i2 * si1 + i3 * si2];
	}
      }
    }
  }

  return 0;
}

template<typename Tin, typename Tout = Tin>
int transpose4_1023(uint64_t out, uint64_t in, const int32_t* dim_size)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  uint64_t si2 = dim_size[3];
  uint64_t si1 = si2 * dim_size[2];
  uint64_t si0 = si1 * dim_size[1];

  uint64_t so2 = dim_size[3];
  uint64_t so1 = so2 * dim_size[2];
  uint64_t so0 = so1 * dim_size[0];

  for (int64_t i0 = 0; i0 < dim_size[1]; ++i0) {
    for (int64_t i1 = 0; i1 < dim_size[0]; ++i1) {
      for (int64_t i2 = 0; i2 < dim_size[2]; ++i2) {
	for (int64_t i3 = 0; i3 < dim_size[3]; ++i3) {
	  po[i0 * so0 + i1 * so1 + i2 * so2 + i3]
	    = pi[i0 * si1 + i1 * si0 + i2 * si2 + i3];
	}
      }
    }
  }

  return 0;
}

#ifdef LIBVETF_INTRINSIC
template<>
inline  int transpose4_0312<float>(uint64_t out, uint64_t in, const int32_t* dim_size) {
  return transpose4_0312_f32(out, in, dim_size) ;
}
#endif

template<typename Tin, typename Tout = Tin>
int transpose4(uint64_t out, uint64_t in, const int32_t* dim_size, const int32_t* perm)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;
  const int64_t d3 = dim_size[3] ;

  int64_t stride_out[4] ;
  stride_out[perm[3]] = 1 ;
  for(int64_t d=2; d>=0; --d) {
    stride_out[perm[d]] = stride_out[perm[d+1]] * dim_size[perm[d+1]] ;
  }
  const int64_t s0 = stride_out[0] ;
  const int64_t s1 = stride_out[1] ;
  const int64_t s2 = stride_out[2] ;
  const int64_t s3 = stride_out[3] ;

  for (int64_t i0 = 0; i0 < d0; ++i0) {
    for (int64_t i1 = 0; i1 < d1; ++i1) {
      for (int64_t i2 = 0; i2 < d2; ++i2) {
	for (int64_t i3 = 0; i3 < d3; ++i3) {
	  po[i0*s0+i1*s1+i2*s2+i3*s3] = pi[((i0*d1+i1)*d2+i2)*d3+i3];
	}
      }
    }
  }

  return 0;
}
}

template<typename T>
int op_TransposeBase(const void* args, size_t len)
{
  struct Args {
    int dtype;
    uint64_t in;
    uint64_t out;
    int size;
    int32_t dim_size[4]; // in
    int32_t perm[4];
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  switch(p->size) {
  case 1 :
    ret = notranspose_copy<T>(p->out, p->in, p->dim_size[0]) ;
    break ;
  case 2 :
    if ( p->perm[0] == 1 && p->perm[1] == 0 ) {
      ret = transpose2_10<T>(p->out, p->in, p->dim_size) ;
    }
    else {
      ret = notranspose_copy<T>(p->out, p->in, p->dim_size[0]*p->dim_size[1]) ;
    }
    break ;
  case 3 :
    if (p->perm[0] == 1 && p->perm[1] == 0 && p->perm[2] == 2) {
      ret = transpose3_102<T>(p->out, p->in, p->dim_size) ;
    }
    else {
      ret = transpose3<T>(p->out, p->in, p->dim_size, p->perm) ;
    }
    break ;
  case 4 :
    if (p->perm[0] == 0 && p->perm[1] == 2
	&& p->perm[2] == 3 && p->perm[3] == 1) {
      ret = 0 ;
#pragma omp parallel reduction(|:ret)
      {
	int64_t nthreads = omp_get_num_threads() ;
	int64_t threadid = omp_get_thread_num() ;

	int64_t chunkSize = p->dim_size[0] / nthreads ;
	int64_t remain    = p->dim_size[0] % nthreads ;

	int64_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
	int64_t myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

	int64_t offset    = chunkBegin * sizeof(T) *  p->dim_size[1] * p->dim_size[2] * p->dim_size[3] ;

	if( myChunk > 0 ) {
	  int32_t dim_size[4] = { (int32_t)myChunk, p->dim_size[1], p->dim_size[2], p->dim_size[3] } ;
	  ret |= transpose4_0231<T>(p->out+offset, p->in+offset, dim_size) ;
	}
      }

    } else if (p->perm[0] == 0 && p->perm[1] == 3
	       && p->perm[2] == 1 && p->perm[3] == 2) {
      ret = 0 ;
#pragma omp parallel reduction(|:ret)
      {
	int64_t nthreads = omp_get_num_threads() ;
	int64_t threadid = omp_get_thread_num() ;

	int64_t chunkSize = p->dim_size[0] / nthreads ;
	int64_t remain    = p->dim_size[0] % nthreads ;

	int64_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
	int64_t myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

	int64_t offset    = chunkBegin * sizeof(T) *  p->dim_size[1] * p->dim_size[2] * p->dim_size[3] ;

	if( myChunk > 0 ) {
	  int32_t dim_size[4] = { (int32_t)myChunk, p->dim_size[1], p->dim_size[2], p->dim_size[3] } ;
	  ret |= transpose4_0312<T>(p->out+offset, p->in+offset, dim_size) ;
	}
      }
    } else if (p->perm[0] == 1 && p->perm[1] == 0
	       && p->perm[2] == 2 && p->perm[3] == 3) {
      ret = transpose4_1023<T>(p->out, p->in, p->dim_size) ;
    }
    else
      {
	ret = transpose4<T>(p->out, p->in, p->dim_size, p->perm) ;
      }
    break ;
  default :
    break ;
  }

  return ret;
}

int op_Transpose(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    uint64_t in;
    uint64_t out;
    int size;
    int32_t dim_size[4]; // in
    int32_t perm[4];
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype
    << " size=" << p->size
    << " perm=(" << p->perm[0]
    << " " << p->perm[1]
    << " " << p->perm[2]
    << " " << p->perm[3]
    << ") dim=(" << p->dim_size[0]
    << " " << p->dim_size[1]
    << " " << p->dim_size[2]
    << " " << p->dim_size[3]
    << ")";

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    ret = op_TransposeBase<float>(args, len);
  } else if (p->dtype == DT_DOUBLE) {
    ret = op_TransposeBase<double>(args, len);
  } else if (p->dtype == DT_INT32) {
    ret = op_TransposeBase<int32_t>(args, len);
  } else if (p->dtype == DT_UINT8) {
    ret = op_TransposeBase<uint8_t>(args, len);
  } else if (p->dtype == DT_INT16) {
    ret = op_TransposeBase<int16_t>(args, len);
  } else if (p->dtype == DT_INT8) {
    ret = op_TransposeBase<int8_t>(args, len);
  } else if (p->dtype == DT_COMPLEX64) {
    ret = op_TransposeBase<float _Complex>(args, len);
  } else if (p->dtype == DT_INT64) {
    ret = op_TransposeBase<int64_t>(args, len);
  } else if (p->dtype == DT_BOOL) {
    ret = op_TransposeBase<bool>(args, len);
  } else if (p->dtype == DT_UINT16) {
    ret = op_TransposeBase<uint16_t>(args, len);
  } else if (p->dtype == DT_COMPLEX128) {
    ret = op_TransposeBase<double _Complex>(args, len);
  } else if (p->dtype == DT_HALF) {
    ret = op_TransposeBase<uint16_t>(args, len);
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

//
// MatMul
//

namespace {
#define GEMM_ARGS(T) \
char* transa, char* transb, \
const int* N, const int* M, const int* K, \
const T* alpha, \
const T* A, const int* lda, \
const T* B, const int* ldb, \
const T* beta, \
T* C, const int* ldc

#define GEMM_REAL_ARGS \
transa, transb, N, M, K, alpha, A, lda, B, ldb, beta, C, ldc

template <typename T> void blas_gemm(GEMM_ARGS(T)) { assert(false && "blas_gemm: not implemented"); }
template<> void blas_gemm<float>(GEMM_ARGS(float)) { sgemm_(GEMM_REAL_ARGS); }

// C[M x N] = A[M x K] * B[K x N] ([rows x cols] in row-major)
//
// M[H x W] (rows x cols in row-major) = M[W x H] (rows x cols in col-major)
//
// C[M x N] (RxC in RM)
//   = C[N x M] (RxC in CM)
//   = B[N x K] (RxC in CM) * A[K x M] (RxC in CM)
//   = B[K x N] (RxC in RM) * A[M x K] (RxC in RM)
//                
template<typename T, char TransA, char TransB>
  int matmul(uint64_t c, uint64_t a, uint64_t b, int M, int N, int K)
  {
    LOG(LOG_DETAIL) << __FUNCTION__ << " begin: (" << M << "," << N << "," << K << ")";
    T* C = reinterpret_cast<T*>(c);
    const T* A = reinterpret_cast<const T*>(a);
    const T* B = reinterpret_cast<const T*>(b);

    T alpha = T(1);
    T beta = T(0);

    char transa = TransA;
    char transb = TransB;
    int lda = TransA == 'N' ? K : M;
    int ldb = TransB == 'N' ? N : K;

#pragma omp parallel
    {
      int nthreads = omp_get_num_threads() ;
      int threadid = omp_get_thread_num() ;

      int chunkSize = M / nthreads ;
      int remain    = M % nthreads ;

      int chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
      int myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

      int offset    = TransA == 'N' ? K : 1 ;

      if( myChunk > 0 ) {
        blas_gemm<T>(&transb, &transa, &N, &myChunk, &K, &alpha, B, &ldb, A+offset*chunkBegin, &lda, &beta, C+N*chunkBegin, &N);
      }
    }

    LOG(LOG_DETAIL) << __FUNCTION__ << " end";
    return 0;
  }
}

int op_MatMul(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    uint64_t a;
    uint64_t b;
    uint64_t out;
    int64_t dim_size_a[2];
    int64_t dim_size_b[2];
    int32_t transpose_a;
    int32_t transpose_b;
  } const *p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__
    << ": dtype=" << p->dtype
    << " a=(" << p->dim_size_a[0] << ", " << p->dim_size_a[1] << ")"
    << " b=(" << p->dim_size_b[0] << ", " << p->dim_size_b[1] << ")"
    << " transpose_a=" << p->transpose_a
    << " transpose_b=" << p->transpose_b;

  int ret = 0;
  if (p->dtype == DT_FLOAT) {
    if (!p->transpose_a && !p->transpose_b) {
      assert(p->dim_size_a[1] == p->dim_size_b[0]);
#if 0
      ret = matmul<float, 'N', 'N'>(
          p->out, p->a, p->b, p->dim_size_a[0], p->dim_size_b[1], p->dim_size_a[1]);
#else
      /* vednn version */
      const uint64_t inDim  = p->dim_size_a[1] ;
      const uint64_t outDim = p->dim_size_b[1] ;
      const uint64_t nBatch = p->dim_size_a[0] ;
      const void* pDataIn     = reinterpret_cast<const void*>(p->a);
      const void* pDataWeight = reinterpret_cast<const void*>(p->b);
      void*       pDataOut    = reinterpret_cast<void*>(p->out);
      ret = vednnLinearForward(inDim,outDim,nBatch, pDataIn, pDataWeight, pDataOut) ;
#endif
    } else if (!p->transpose_a && p->transpose_b) {
      assert(p->dim_size_a[1] == p->dim_size_b[1]);
#if 1
      ret = matmul<float, 'N', 'T'>(
          p->out, p->a, p->b, p->dim_size_a[0], p->dim_size_b[0], p->dim_size_a[1]);
#else
      /* vednn version */
      const uint64_t inDim  = p->dim_size_b[0] ;
      const uint64_t outDim = p->dim_size_a[1] ;
      const uint64_t nBatch = p->dim_size_a[0] ;
      const void* pDataGradOut = reinterpret_cast<const void*>(p->a);
      const void* pDataWeight  = reinterpret_cast<const void*>(p->b);
      void*       pDataGradIn  = reinterpret_cast<void*>(p->out);
      ret = vednnLinearBackwardData(inDim,outDim,nBatch, pDataGradOut, pDataWeight, pDataGradIn) ;
#endif
    } else if (p->transpose_a && !p->transpose_b) {
      assert(p->dim_size_a[0] == p->dim_size_b[0]);
      ret = matmul<float, 'T', 'N'>(
          p->out, p->a, p->b, p->dim_size_a[1], p->dim_size_b[1], p->dim_size_a[0]);
    }
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}


//
// Pack
//

template<typename T>
  int pack(uint64_t n, uint64_t l, uint64_t *in, uint64_t out)
  {
    const T** pi = reinterpret_cast<const T**>(in);
    T* po = reinterpret_cast<T*>(out);

    for(int64_t i=0; i<n; i++) {
      for(int64_t j=0; j<l; j++) {
        po[j] = pi[i][j] ;
      }
      po += l ;
    }

    return 0 ;
  }

int op_Pack(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  int ret=1;

  struct Args {
    int dtype;
    uint64_t n;
    uint64_t l;
    uint64_t out;
    uint64_t in[1] ;
  } const* p;

  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype;

  if (p->dtype == DT_FLOAT) {
    ret = pack<float>(p->n, p->l, (uint64_t*)&p->in[0], p->out) ;
  }
  else if (p->dtype == DT_INT32) {
    ret = pack<int32_t>(p->n, p->l, (uint64_t*)&p->in[0], p->out) ;
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;

  return ret;
}


//
// Slice
//
template <typename T>
int slice1(uint64_t input_ptr, uint64_t output_ptr,
           uint64_t input_size, uint64_t output_size, uint64_t index) 
{
  const T* pi = reinterpret_cast<const T*>(input_ptr);
  T* po = reinterpret_cast<T*>(output_ptr);

  const T* pi_ = pi + index ;
  for(int i=0; i<output_size; i++) {
    po[i] = pi_[i] ;
  }

  return 0 ;
} 

template <typename T>
int slice2(uint64_t input_ptr, uint64_t output_ptr,
           uint64_t *input_size, uint64_t *output_size, uint64_t *index) 
{
  const T* pi = reinterpret_cast<const T*>(input_ptr);
  T* po = reinterpret_cast<T*>(output_ptr);

  for(size_t i0=0; i0<output_size[0]; i0++) {
    const size_t ii0 = i0+index[0];
    const size_t io0 = i0 ;
    for(size_t i1=0; i1<output_size[1]; i1++) {
      const size_t ii1 = ii0 * input_size[1] + index[1] + i1 ;
      const size_t io1 = io0 * output_size[1] + i1 ;
      po[io1] = pi[ii1] ;
    }
  }

  return 0 ;
}

template <typename T>
int slice3(uint64_t input_ptr, uint64_t output_ptr,
           uint64_t *input_size, uint64_t *output_size, uint64_t *index)
{
  const T* pi = reinterpret_cast<const T*>(input_ptr);
  T* po = reinterpret_cast<T*>(output_ptr);

  for(size_t i0=0; i0<output_size[0]; i0++) {
    const size_t ii0 = i0+index[0];
    const size_t io0 = i0 ;
    for(size_t i1=0; i1<output_size[1]; i1++) {
      const size_t ii1 = ii0 * input_size[1] + index[1] + i1 ;
      const size_t io1 = io0 * output_size[1] + i1 ;
      for(size_t i2=0; i2<output_size[2]; i2++) {
        const size_t ii2 = ii1 * input_size[2] + index[2] + i2 ;
        const size_t io2 = io1 * output_size[2] + i2 ;
        po[io2] = pi[ii2] ;
      }
    }
  }
  return 0 ;
} 

template <typename T>
int slice4(uint64_t input_ptr, uint64_t output_ptr,
           uint64_t *input_size, uint64_t *output_size, uint64_t *index)
{
  const T* pi = reinterpret_cast<const T*>(input_ptr);
  T* po = reinterpret_cast<T*>(output_ptr);

  for(size_t i0=0; i0<output_size[0]; i0++) {
    const size_t ii0 = i0+index[0];
    const size_t io0 = i0 ;
    for(size_t i1=0; i1<output_size[1]; i1++) {
      const size_t ii1 = ii0 * input_size[1] + index[1] + i1 ;
      const size_t io1 = io0 * output_size[1] + i1 ;
      for(size_t i2=0; i2<output_size[2]; i2++) {
        const size_t ii2 = ii1 * input_size[2] + index[2] + i2 ;
        const size_t io2 = io1 * output_size[2] + i2 ;
        for(size_t i3=0; i3<output_size[3]; i3++) {
          const size_t ii3 = ii2 * input_size[3] + index[3] + i3 ;
          const size_t io3 = io2 * output_size[3] + i3 ;
          po[io3] = pi[ii3] ;
        }
      }
    }
  }
  return 0 ;
}

template <typename T>
int slice_handle(uint64_t input_dims, uint64_t input_ptr, uint64_t output_ptr, uint64_t *array) 
{
  int ret=1 ;
  switch( input_dims ) {
  case 1 :
    ret = slice1<T>(input_ptr, output_ptr, array[0], array[1], array[2]) ;
    break ;
  case 2 :
    ret = slice2<T>(input_ptr, output_ptr, array, array+2, array+4) ;
    break ;
  case 3 :
    ret = slice3<T>(input_ptr, output_ptr, array, array+3, array+6) ;
    break ;
  case 4 :
    ret = slice4<T>(input_ptr, output_ptr, array, array+4, array+8) ;
    break ;
#if 0 // todo : add larger dim
  case 5 :
    break ;
  case 6 :
    break ;
  case 7 :
    break ;
#endif
  default :
    break ;
  }
  return ret ;
}
//
int op_Slice(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  int ret=1;

  struct Args {
    int dtype;
    uint64_t input_dims;
    uint64_t input_ptr ;
    uint64_t output_ptr ;
    uint64_t array[1] ;
  } const* p;

  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype;

  if (p->dtype == DT_FLOAT) {
    ret = slice_handle<float>(p->input_dims, p->input_ptr, p->output_ptr, (uint64_t*) p->array) ;
  }
  else if (p->dtype == DT_INT32) {
    ret = slice_handle<int32_t>(p->input_dims, p->input_ptr, p->output_ptr, (uint64_t*) p->array) ;
  }
  else if (p->dtype == DT_DOUBLE) {
    ret = slice_handle<double>(p->input_dims, p->input_ptr, p->output_ptr, (uint64_t*) p->array) ;
  }
  

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret ; 
}
