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

};
