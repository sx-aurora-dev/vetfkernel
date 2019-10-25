#include <cstdint>
#include "kernel.h"
#include "types.h"
#include "log.h"
#include <sstream>
#include "vml.h"

REGISTER_KERNEL("SigmoidGrad", "op_SigmoidGrad");
REGISTER_KERNEL("SqrtGrad",    "op_SqrtGrad");
REGISTER_KERNEL("TanhGrad",    "op_TanhGrad");


extern "C" {
int op_SigmoidGrad(const void* arg, size_t len);
int op_SqrtGrad(const void* arg, size_t len);
int op_TanhGrad(const void* arg, size_t len);
}

namespace {

struct BinaryOpArgs {
  vml::Tensor in0;
  vml::Tensor in1;
  vml::Tensor out;
};

static inline
bool CheckTypes(const BinaryOpArgs& args, int dt0, int dt1, int dt2)
{
  return args.in0.dtype == dt0
    && args.in1.dtype == dt1
    && args.out.dtype == dt2;
}

static inline
bool CheckTypesAll(const BinaryOpArgs& args, int dtype) {
  return CheckTypes(args, dtype, dtype, dtype);
}


static
int op_CwiseGradients(const void* args, size_t len,
                      int (*func)(const BinaryOpArgs&),
		      const char* name)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << " begin";
  int ret = 1;

  if (sizeof(BinaryOpArgs) == len) {
    const BinaryOpArgs* p = reinterpret_cast<const BinaryOpArgs*>(args);

    LOG(LOG_PARAM) << name << ":"
      << " out="  << p->in0
      << " gout=" << p->in1
      << " gin="  << p->out;

    if (func)
      ret = func(*p);
  } else {
    LOG(LOG_ERROR) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(BinaryOpArgs) << " bytes expected";
  }

  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << " end. ret=" << ret;
  return ret;
}



// SigmoidGrad

template <typename T>
int sigmoid_grad_nn(uint64_t gin, uint64_t out, uint64_t gout, size_t n)
{
  T* gi = reinterpret_cast<T*>(gin);
  const T* oo = reinterpret_cast<const T*>(out);
  const T* go = reinterpret_cast<const T*>(gout);

  for (size_t i = 0; i < n; ++i) {
    gi[i] = go[i] * oo[i] * (T(1.) - oo[i]) ;
  }

  return 0;
}

int op_sigmoidGrad(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    // TODO : impl other patterns
    if (args.in0.nelems == args.in1.nelems) {
     r = sigmoid_grad_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                               args.in0.nelems);
    }

    return r;
  }
  return 1;
}


// SqrtGrad

template <typename T>
int sqrt_grad_nn(uint64_t gin, uint64_t out, uint64_t gout, size_t n)
{
  T* gi = reinterpret_cast<T*>(gin);
  const T* oo = reinterpret_cast<const T*>(out);
  const T* go = reinterpret_cast<const T*>(gout);

  for (size_t i = 0; i < n; ++i) {
    gi[i] = T(0.5) * go[i] / std::sqrt(oo[i]) ;
  }

  return 0;
}

int op_sqrtGrad(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    // TODO : impl other patterns
    if (args.in0.nelems == args.in1.nelems) {
     r = sqrt_grad_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                               args.in0.nelems);
    }

    return r;
  }
  return 1;
}


// TanhGrad

template <typename T>
int tanh_grad_nn(uint64_t gin, uint64_t out, uint64_t gout, size_t n)
{
  T* gi = reinterpret_cast<T*>(gin);
  const T* oo = reinterpret_cast<const T*>(out);
  const T* go = reinterpret_cast<const T*>(gout);

  for (size_t i = 0; i < n; ++i) {
    gi[i] = go[i] * (T(1.) - oo[i] * oo[i]) ;
  }

  return 0;
}

int op_tanhGrad(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    // TODO : impl other patterns
    if (args.in0.nelems == args.in1.nelems) {
     r = tanh_grad_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                               args.in0.nelems);
    }

    return r;
  }
  return 1;
}

} // namespace


int op_SigmoidGrad(const void* args, size_t len)
{
  return op_CwiseGradients(args, len, op_sigmoidGrad, "op_SigmoidGrad");
}
int op_SqrtGrad(const void* args, size_t len)
{
  return op_CwiseGradients(args, len, op_sqrtGrad, "op_SqrtGrad");
}
int op_TanhGrad(const void* args, size_t len)
{
  return op_CwiseGradients(args, len, op_tanhGrad, "op_TanhGrad");
}
