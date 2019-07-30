#include <cstdint>
#include "ve_ops_common.h"
#include "types.h"

namespace {
struct Param {
  int64_t ksize[4]; // NCHW
  int64_t stride[4];
  int32_t data_format;
  int32_t padding; // 1=VALID, 2=SAME
} __attribute__((__packed__));

// (data_format)_(ksize)_(stride)_(paddding)
template<typename T>
int avgpool_nchw_11hw_1111_same(T* out, T const* in, int64_t const* dim_size,
                                size_t h, size_t w)
{
  size_t h2 = h / 2;
  size_t w2 = w / 2;
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
            int64_t y1 = y + y0 - h2;
            if (y1 < 0 || y1 >= H) continue;

            for (size_t x0 = 0; x0 < w; ++x0) {
              int64_t x1 = x + x0 - w2;
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

int op_avgpool(const VEOpArgs& args)
{
  if (args.nVariables() != 3)
    return 1;

  // ndims == 4 (checked by tf)

  const Tensor* out = args.arg<Tensor>(0);
  const Tensor* in = args.arg<Tensor>(1);
  const Param* param = args.arg<Param>(2);

  LOG(3) << "in=" << in->to_s() << " out=" << out->to_s();
  LOG(3) << "param: ksize=[" << param->ksize[0] 
    << "," << param->ksize[1] 
    << "," << param->ksize[2]
    << "," << param->ksize[3]
    << "] stride=[" << param->stride[0]
    << "," << param->stride[1]
    << "," << param->stride[2]
    << "," << param->stride[3]
    << "] data_format=" << param->data_format
    << " padding=" << param->padding;

  if (in->dtype == DT_FLOAT && out->dtype == DT_FLOAT
      && param->ksize[0] == 1 && param->ksize[1] == 1
      && param->stride[0] == 1 && param->stride[1] == 1
      && param->stride[2] == 1 && param->stride[3] == 1
      && param->data_format == FORMAT_NCHW
      && param->padding == SAME) {
    float* po = reinterpret_cast<float*>(out->addr);
    float const* pi = reinterpret_cast<float const*>(in->addr);
    return avgpool_nchw_11hw_1111_same<float>(po, pi, in->dim_size,
                                              param->ksize[2], param->ksize[3]);
  }


  return 1;
}

} // namespace

DEFINE_KERNEL(AvgPool, op_avgpool);
