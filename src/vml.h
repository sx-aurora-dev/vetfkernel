// VML: VE Machile Learning Kernels

namespace vml
{

struct Tensor {
  int32_t dtype;
  uint64_t addr;
  int32_t dims;
  int64_t nelems;
  int64_t dim_size[8];

#if 0
  size_t size() const {
    return sizeof(Tensor) + sizeof(int64_t) * (dims - 1);
  }
#endif

  template <typename T> T ptr() const {
    return reinterpret_cast<T>(addr);
  }

  std::string to_s() const {
    std::stringstream s;

    s << "[dtype=" << dtype
      << ",dims=" << dims
      << ",nelems=" << nelems
      << ",dim_size=[";

    for (size_t i = 0; i < dims; ++i) {
      s << dim_size[i];
      if (i < dims - 1)
        s << ",";
    }
    s << "]]";
    return s.str();
  }
} __attribute__((__packed__));

struct PoolingParam {
  int64_t ksize[4];
  int64_t stride[4];
  int32_t data_format;
  int32_t padding; // 1=VALID, 2=SAME
  int64_t pad_rows;
  int64_t pad_cols;
} __attribute__((__packed__));

int avgpool(Tensor const& out, Tensor const& in, PoolingParam const& param);
int avgpoolgrad(Tensor const& out, Tensor const& in, PoolingParam const& param);

// unary op
int abs(vml::Tensor const& out, vml::Tensor const& in);
int exp(vml::Tensor const& out, vml::Tensor const& in);
int floor(vml::Tensor const& out, vml::Tensor const& in);
int log(vml::Tensor const& out, vml::Tensor const& in);
int neg(vml::Tensor const& out, vml::Tensor const& in);
int reciprocal(vml::Tensor const& out, vml::Tensor const& in);
int rsqrt(vml::Tensor const& out, vml::Tensor const& in);
int sigmoid(vml::Tensor const& out, vml::Tensor const& in);
int sqrt(vml::Tensor const& out, vml::Tensor const& in);
int square(vml::Tensor const& out, vml::Tensor const& in);
int tanh(vml::Tensor const& out, vml::Tensor const& in);

// binary op
int add(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int div(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int mul(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int sub(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int divnonan(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int pow(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int sqdiff(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int rsqrt_grad(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int minimum(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int maximum(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int equal(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int notEqual(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int less(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int lessEqual(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int greater(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int greaterEqual(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);

int randomUniform(vml::Tensor const& t);
int tile(vml::Tensor const& out, vml::Tensor const& in);
};
