#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include "log.h"
#include "kernel.h"
#include "types.h"

#define REGISTER_UNARY_OP(NAME, FUNC) \
  REGISTER_KERNEL(#NAME, "__op_" # NAME); \
  extern "C" { \
    int __op_##NAME(const void* args, size_t len) { \
      return unary_op(args, len, FUNC, "op_" # NAME); \
    } \
  }


namespace {

struct _Tensor {
  int dtype;
  int data_format;
  uint64_t addr;
  int32_t dims;
  int64_t nelems;
  int64_t dim_size[8];

  template <typename T> T ptr() const {
    return reinterpret_cast<T>(addr);
  }
};

// valid: in.dtype, in.nelems, in.addr, out.addr
struct UnaryOpArgs
{
  _Tensor in;
  _Tensor out;

  bool isT(int ty) const { return in.dtype == ty; }
  size_t nelems() const { return in.nelems; }
};

int unary_op(const void* args, size_t len,
             int (*func)(UnaryOpArgs const& args),
             char const* name)
{
  LOG(2) << __FUNCTION__ << "::" << name << ": begin";
  int ret = 1;

  if (sizeof(UnaryOpArgs) == len) {
    const UnaryOpArgs* p = reinterpret_cast<const UnaryOpArgs*>(args);

    LOG(1) << __FUNCTION__ << "::" << name << ":"
        << " dtype=" << p->in.dtype << " nelems=" << p->in.nelems;

    if (func) {
      ret = func(*p);
    }
  } else {
    LOG(3) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(UnaryOpArgs) << " bytes expected";
  }

  LOG(2) << __FUNCTION__ << "::" << name << ": end. ret=" << ret;
  return ret;
}

bool isT(UnaryOpArgs const& args, int type)
{
  return args.in.dtype;
}

} // namespace

int op_Abs(UnaryOpArgs const& args)
{
  if (args.isT(DT_FLOAT)) {
    float* out = args.out.ptr<float*>();
    float const* in = args.in.ptr<float const*>();
    for (size_t i = 0; i < args.nelems(); ++i)
      out[i] = fabsf(in[i]);
    return 0;
  } else if (args.isT(DT_DOUBLE)) {
    double* out = args.out.ptr<double*>();
    double const* in = args.in.ptr<double const*>();
    for (size_t i = 0; i < args.nelems(); ++i)
      out[i] = fabs(in[i]);
    return 0;
  } else if (args.isT(DT_INT32)) {
    int32_t* out = args.out.ptr<int32_t*>();
    int32_t const* in = args.in.ptr<int32_t const*>();
    for (size_t i = 0; i < args.nelems(); ++i)
      out[i] = abs(in[i]);
    return 0;
  } else if (args.isT(DT_INT64)) {
    int64_t* out = args.out.ptr<int64_t*>();
    int64_t const* in = args.in.ptr<int64_t const*>();
    for (size_t i = 0; i < args.nelems(); ++i)
      out[i] = labs(in[i]);
    return 0;
  }

  return 1;
}

REGISTER_UNARY_OP(Abs, op_Abs);
