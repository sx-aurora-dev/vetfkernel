#include <cstdint>
#include "ve_ops_common.h"
#include "types.h"
#include "vml.h"

namespace vml {
namespace {

// (data_format)_(ksize)_(stride)_(paddding)

template<typename T>
int avgpool_nchw_1133_1111_same(T* out, T const* in, int64_t const* dim_size)
{
  //LOG(LOG_DETAIL) << __FUNCTION__;
  size_t N = dim_size[0];
  size_t C = dim_size[1];
  size_t H = dim_size[2];
  size_t W = dim_size[3];
  size_t szCHW = C * H * W;
  size_t szHW = H * W;
  T div = T(1) / (3 * 3);

#pragma omp parallel for
  for (size_t b = 0; b < N; ++b) {
    for (size_t c = 0; c < C; ++c) {
      T* out0 = out + b * szCHW + c * szHW;
      T const* in0 = in + b * szCHW + c * szHW;

      // (0, 0)
      out0[0] = (in0[0] + in0[1] + in0[W] + in0[W + 1]) / 4;
      // (0, W - 1)
      out0[W - 1] = (in0[W - 2] + in0[W - 1] 
                     + in0[2 * W - 2] + in0[2 * W - 1]) / 4;
      // (H - 1, 0)
      out0[(H - 1) * W] = (in0[(H - 2) * W] + in0[(H - 2) * W + 1]
                           + in0[(H - 1) * W] + in0[(H - 1) * W + 1]) / 4;
      // (H - 1, W - 1)
      out0[H * W - 1] = (in0[H * W - W - 2] + in0[H * W - W - 1]
                         + in0[H * W - 2] + in0[H * W - 1]) / 4;

      for (size_t x = 1; x < W - 1; ++x) {
        T s;

        // (0, x)
        s = T(0);
        s += in0[0 * W + x - 1] + in0[0 * W + x] + in0[0 * W + x + 1];
        s += in0[1 * W + x - 1] + in0[1 * W + x] + in0[1 * W + x + 1];
        out0[x] = s / 6;

        // (H - 1, x)
        s = T(0);
        s += in0[(H-2)*W + x-1] + in0[(H-2)*W + x] + in0[(H-2)*W + x+1];
        s += in0[(H-1)*W + x-1] + in0[(H-1)*W + x] + in0[(H-1)*W + x+1];
        out0[(H - 1) * W + x] = s / 6;
      }

      for (size_t y = 1; y < H - 1; ++y) {
        T s;

        // (y, 0)
        s = T(0);
        s += in0[(y - 1) * W] + in0[(y - 1) * W + 1];
        s += in0[(y + 0) * W] + in0[(y + 0) * W + 1];
        s += in0[(y + 1) * W] + in0[(y + 1) * W + 1];
        out0[y * W] = s / 6;

        // (y, W - 1)
        s = T(0);
        s += in0[(y - 1) * W + W - 2] + in0[(y - 1) * W + W - 1];
        s += in0[(y + 0) * W + W - 2] + in0[(y + 0) * W + W - 1];
        s += in0[(y + 1) * W + W - 2] + in0[(y + 1) * W + W - 1];
        out0[y * W + W - 1] = s / 6;
      }

      for (size_t y = 1; y < H - 1; ++y) {
        for (size_t x = 1; x < W - 1; ++x) {
          T s = T(0);
          s += in0[(y - 1) * W + (x - 1)];
          s += in0[(y - 1) * W + (x + 0)];
          s += in0[(y - 1) * W + (x + 1)];
          s += in0[(y + 0) * W + (x - 1)];
          s += in0[(y + 0) * W + (x + 0)];
          s += in0[(y + 0) * W + (x + 1)];
          s += in0[(y + 1) * W + (x - 1)];
          s += in0[(y + 1) * W + (x + 0)];
          s += in0[(y + 1) * W + (x + 1)];
          out0[y * W + x] = s * div;
        }
      }
    }
  }

  return 0;
}

template<typename T>
int avgpool_nchw_11hw_1111_same(T* out, T const* in, int64_t const* dim_size,
                                PoolingParam const& param)
{
  size_t h = param.ksize[2];
  size_t w = param.ksize[3];
  size_t N = dim_size[0];
  size_t C = dim_size[1];
  size_t H = dim_size[2];
  size_t W = dim_size[3];

  for (size_t b = 0; b < N; ++b) {
    for (size_t c = 0; c < C; ++c) {
      T* out0 = out + b * C * H * W + c * H * W;
      T const* in0 = in + b * C * H * W + c * H * W;

      for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {

          T s = T(0);
          int32_t c = 0;
          for (size_t y0 = 0; y0 < h; ++y0) {
            int64_t y1 = y + y0 - param.pad_rows;
            if (y1 < 0 || y1 >= H) continue;

            for (size_t x0 = 0; x0 < w; ++x0) {
              int64_t x1 = x + x0 - param.pad_cols;
              if (x1 < 0 || x1 >= H) continue;
              s += in0[y1 * W + x1];
              ++c;
            }
          }
          out0[y * W + x] = s / c;
        }
      }
    }
  }

  return 0;
}

} // namespace

int avgpool(vml::Tensor const& out, vml::Tensor const& in, 
            vml::PoolingParam const& param)
{
  LOG(LOG_PARAM) << __FUNCTION__ << ": in=" << in << " out=" << out;
  LOG(LOG_PARAM) << __FUNCTION__ << ": param: ksize=[" << param.ksize[0] 
    << "," << param.ksize[1] 
    << "," << param.ksize[2]
    << "," << param.ksize[3]
    << "] stride=[" << param.stride[0]
    << "," << param.stride[1]
    << "," << param.stride[2]
    << "," << param.stride[3]
    << "] data_format=" << param.data_format
    << " padding=" << param.padding
    << "(rows=" << param.pad_rows 
    << ",cols=" << param.pad_cols 
    << ")";

  if (in.dtype == DT_FLOAT && out.dtype == DT_FLOAT
      && param.ksize[0] == 1 && param.ksize[1] == 1
      && param.stride[0] == 1 && param.stride[1] == 1
      && param.stride[2] == 1 && param.stride[3] == 1
      && param.data_format == FORMAT_NCHW
      && param.padding == SAME) {
    float* po = reinterpret_cast<float*>(out.addr);
    float const* pi = reinterpret_cast<float const*>(in.addr);
    if (param.ksize[2] == 3 && param.ksize[3] == 3) {
      return avgpool_nchw_1133_1111_same<float>(po, pi, in.dim_size);
    } else {
      return avgpool_nchw_11hw_1111_same<float>(po, pi, in.dim_size, param);
    }
  }

  return 1;
};

namespace {

template<typename T>
int avgpoolgrad_nchw_1133_1111_same(T* out, T const* in, int64_t const* dim_size)
{
  size_t N = dim_size[0];
  size_t C = dim_size[1];
  size_t H = dim_size[2];
  size_t W = dim_size[3];
  size_t szCHW = C * H * W;
  size_t szHW = H * W;
  T div = T(1) / T(3 * 3);

  memset(out, 0, sizeof(T) * N * C * H * W);

#pragma omp parallel for
  for (size_t b = 0; b < N; ++b) {
    for (size_t c = 0; c < C; ++c) {
      T* out0 = out + b * szCHW + c * szHW;
      T const* in0 = in + b * szCHW + c * szHW;

#if 1
      // (0, 0)
      out0[0] += in0[0] / 4;
      out0[1] += in0[0] / 4;
      out0[W + 0] += in0[0] / 4;
      out0[W + 1] += in0[0] / 4;
      // (0, W - 1)
      out0[W - 2] += in0[W - 1] / 4;
      out0[W - 1] += in0[W - 1] / 4;
      out0[2 * W - 2] += in0[W - 1] / 4;
      out0[2 * W - 1] += in0[W - 1] / 4;
      // (H - 1, 0)
      out0[(H - 2) * W + 0] += in0[(H - 1) * W]  / 4;
      out0[(H - 2) * W + 1] += in0[(H - 1) * W]  / 4;
      out0[(H - 1) * W + 0] += in0[(H - 1) * W]  / 4;
      out0[(H - 1) * W + 1] += in0[(H - 1) * W]  / 4;
      // (H - 1, W - 1)
      out0[(H - 2) * W + W - 2] += in0[H * W  - 1] / 4;
      out0[(H - 2) * W + W - 1] += in0[H * W  - 1] / 4;
      out0[(H - 1) * W + W - 2] += in0[H * W  - 1] / 4;
      out0[(H - 1) * W + W - 1] += in0[H * W  - 1] / 4;

      for (size_t x = 1; x < W - 1; ++x) {
        T v;

        // (0, x)
        v = in0[x] / 6;
        out0[x - 1] += v;
        out0[x + 0] += v;
        out0[x + 1] += v;
        out0[W + x - 1] += v;
        out0[W + x + 0] += v;
        out0[W + x + 1] += v;

        // (H - 1, x)
        v = in0[(H - 1) * W + x] / 6;
        out0[(H - 2) * W + x - 1] += v;
        out0[(H - 2) * W + x + 0] += v;
        out0[(H - 2) * W + x + 1] += v;
        out0[(H - 1) * W + x - 1] += v;
        out0[(H - 1) * W + x + 0] += v;
        out0[(H - 1) * W + x + 1] += v;
      }

      for (size_t y = 1; y < H - 1; ++y) {
        T v;

        // (y, 0)
        v = in0[y * W] / 6;
        out0[(y - 1) * W + 0] += v;
        out0[(y - 1) * W + 1] += v;
        out0[(y + 0) * W + 0] += v;
        out0[(y + 0) * W + 1] += v;
        out0[(y + 1) * W + 0] += v;
        out0[(y + 1) * W + 1] += v;

        // (y, W - 1)
        v = in0[y * W + W - 1] / 6;
        out0[(y - 1) * W + W - 2] += v;
        out0[(y - 1) * W + W - 1] += v;
        out0[(y + 0) * W + W - 2] += v;
        out0[(y + 0) * W + W - 1] += v;
        out0[(y + 1) * W + W - 2] += v;
        out0[(y + 1) * W + W - 1] += v;
      }
#endif

      for (size_t y = 1; y < H - 1; ++y) {
        for (size_t x = 1; x < W - 1; ++x) {
          T v = in0[y * W + x] * div;
          out0[(y - 1) * W + x - 1] += v;
          out0[(y - 1) * W + x + 0] += v;
          out0[(y - 1) * W + x + 1] += v;
          out0[(y + 0) * W + x - 1] += v;
          out0[(y + 0) * W + x + 0] += v;
          out0[(y + 0) * W + x + 1] += v;
          out0[(y + 1) * W + x - 1] += v;
          out0[(y + 1) * W + x + 0] += v;
          out0[(y + 1) * W + x + 1] += v;
        }
      }
    }
  }

  return 0;
}

} // namespace

int avgpoolgrad(vml::Tensor const& out, vml::Tensor const& in, PoolingParam const& param)
{
  LOG(LOG_PARAM) << __FUNCTION__ << ": in=" << in << " out=" << out;
  LOG(LOG_PARAM) << __FUNCTION__ << ": param: ksize=[" << param.ksize[0] 
    << "," << param.ksize[1] 
    << "," << param.ksize[2]
    << "," << param.ksize[3]
    << "] stride=[" << param.stride[0]
    << "," << param.stride[1]
    << "," << param.stride[2]
    << "," << param.stride[3]
    << "] data_format=" << param.data_format
    << " padding=" << param.padding;

  if (in.dtype == DT_FLOAT && out.dtype == DT_FLOAT
      && param.ksize[0] == 1 && param.ksize[1] == 1
      && param.ksize[2] == 3 && param.ksize[3] == 3
      && param.stride[0] == 1 && param.stride[1] == 1
      && param.stride[2] == 1 && param.stride[3] == 1
      && param.data_format == FORMAT_NCHW
      && param.padding == SAME) {
    float* po = reinterpret_cast<float*>(out.addr);
    float const* pi = reinterpret_cast<float const*>(in.addr);
    return avgpoolgrad_nchw_1133_1111_same<float>(po, pi, in.dim_size);
  }

  return 1;
}

}; // namespace vml

// End of VML

namespace {

int op_avgpool(const VEOpArgs& args)
{
  if (args.nArguments() != 3)
    return 1;

  // ndims == 4 (checked by tf)

  const vml::Tensor* out = args.arg<vml::Tensor>(0);
  const vml::Tensor* in = args.arg<vml::Tensor>(1);
  const vml::PoolingParam* param = args.arg<vml::PoolingParam>(2);

  return vml::avgpool(*out, *in, *param);
}

int op_avgpoolgrad(const VEOpArgs& args)
{
  if (args.nArguments() != 3)
    return 1;

  // ndims == 4 (checked by tf)

  const vml::Tensor* out = args.arg<vml::Tensor>(0);
  const vml::Tensor* in = args.arg<vml::Tensor>(1);
  const vml::PoolingParam* param = args.arg<vml::PoolingParam>(2);

  return vml::avgpoolgrad(*out, *in, *param);
}

} // namespace

DEFINE_KERNEL(AvgPool, op_avgpool);
DEFINE_KERNEL(AvgPoolGrad, op_avgpoolgrad);
