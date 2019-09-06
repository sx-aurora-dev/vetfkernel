#include <algorithm>
#include <cstdint>
#include <cassert>
#include "kernel.h"
#include "types.h"
#include "log.h"
#include <sstream>
#include <vector>

#include "vml.h"

#ifdef LIBVETF_INTRINSIC
#include "libvetfkernel.h"
#endif

//#define TIMER
#ifdef TIMER
#include "timer.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

//#define DEBUG

namespace {

struct BinaryOpArgs {
  vml::Tensor in0;
  vml::Tensor in1;
  vml::Tensor out;
} __attribute__((__packed__));

bool CheckTypes(const BinaryOpArgs& args, int dt0, int dt1, int dt2)
{
  return args.in0.dtype == dt0
    && args.in1.dtype == dt1
    && args.out.dtype == dt2;
}

bool CheckTypesAll(vml::Tensor const& X,
                   vml::Tensor const& Y,
                   vml::Tensor const& Z,
                   int dtype) {
  return X.dtype == dtype && Y.dtype == dtype && Z.dtype == dtype;
}


bool CheckTypesAll(const BinaryOpArgs& args, int dtype) {
  return CheckTypes(args, dtype, dtype, dtype);
}

bool CheckDimsAll(const vml::Tensor &out, const vml::Tensor &in0, const vml::Tensor &in1, size_t dims)
{
  bool ret = false;
  if (out.dims == dims) {
    if (in0.dims == dims) {
      if (in1.dims == dims) {
	ret = true;
      }
    }
  }
  return ret;
}

bool CheckDimsAll(const BinaryOpArgs& args, size_t dims)
{
  return CheckDimsAll(args.out, args.in0, args.in1, dims);
}

bool IsSameDims(vml::Tensor const& X,
                vml::Tensor const& Y,
                vml::Tensor const& Z)
{
  return X.dims == Y.dims && X.dims == Z.dims;
}

bool IsSameDims(const BinaryOpArgs& args)
{
    return args.in0.dims == args.in1.dims
        && args.in0.dims == args.out.dims;
}

bool check_dim(vml::Tensor const& s, std::vector<int64_t> const& dim)
{
  return s.dims == dim.size()
      && s.dim_size[0] == dim[0]
      && s.dim_size[1] == dim[1]
      && s.dim_size[2] == dim[2]
      && s.dim_size[3] == dim[3]
      && s.dim_size[4] == dim[4];
}



bool IsSameSize(const vml::Tensor &out, const vml::Tensor &in0, const vml::Tensor &in1)
{
  if (in0.nelems != in1.nelems || in0.nelems != out.nelems) {
    return false;
  }
  const int32_t dims_in0 = in0.dims;
  const int32_t dims_in1 = in1.dims;
  const int32_t dims_out = out.dims;
  int32_t dimsMin = std::min(std::min(dims_in0, dims_in1), dims_out);
  for (int32_t i=1; i<dimsMin; i++) {
    if (in0.dim_size[dims_in0 - i] != in1.dim_size[dims_in1 - i] ||
	in0.dim_size[dims_in1 - i] != out.dim_size[dims_out - i]) {
      return false;
    }
  }
  // 一番小さい配列でサイズが同じでnelemsが等しい場合、残りの次元は常に1なので true を返す
  return true;
}



bool IsSameSize(const BinaryOpArgs& args)
{
  return IsSameSize(args.out, args.in0, args.in1);
}



// X = Y op Z
int op_Binary(const void* args, size_t len, 
              int (*func)(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z),
              const char* name)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": begin";
  int ret = 1;

  if (sizeof(BinaryOpArgs) == len) {
    const BinaryOpArgs* p = reinterpret_cast<const BinaryOpArgs*>(args);

    LOG(LOG_PARAM) << __FUNCTION__ << "::" << name << ":"
      << " in0=" << p->in0.to_s()
      << " in1=" << p->in1.to_s()
      << " out=" << p->out.to_s();

    if (func) {
#ifdef TIMER
      double t0 = second();
#endif
      ret = func(p->out, p->in0, p->in1);
#ifdef TIMER
      double ms = (second() - t0) * 1e3;
      LOG(LOG_TIMER) << __FUNCTION__ << "::" << name << ": " << ms << " msec";
#endif
    }
  } else {
    LOG(LOG_ERROR) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(BinaryOpArgs) << " bytes expected";
  }

  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": end. ret=" << ret;
  return ret;
}



// obsolete
int op_Binary_(const void* args, size_t len, 
              int (*func)(const BinaryOpArgs&),
              const char* name)
{
  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": begin";
  int ret = 1;

  if (sizeof(BinaryOpArgs) == len) {
    const BinaryOpArgs* p = reinterpret_cast<const BinaryOpArgs*>(args);

    LOG(LOG_PARAM) << __FUNCTION__ << "::" << name << ":"
      << " in0=" << p->in0.to_s()
      << " in1=" << p->in1.to_s()
      << " out=" << p->out.to_s();

    if (func) {
#ifdef TIMER
      double t0 = second();
#endif
      ret = func(*p);
#ifdef TIMER
      double ms = (second() - t0) * 1e3;
      LOG(LOG_TIMER) << __FUNCTION__ << "::" << name << ": " << ms << " msec";
#endif
    }
  } else {
    LOG(LOG_ERROR) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(BinaryOpArgs) << " bytes expected";
  }

  LOG(LOG_TRACE) << __FUNCTION__ << "::" << name << ": end. ret=" << ret;
  return ret;
}



// Add
template <typename T>
int add_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] + i1;
  }

  return 0;
}

#ifdef LIBVETF_INTRINSIC
template <>
int add_n1<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  return add_n1_f32(out,in0,in1,n) ;
}
#endif



template <typename T>
int add_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] + pi1[i];
  }

  return 0;
}



#ifdef LIBVETF_INTRINSIC
template <>
inline int add_nn<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  return add_nn_f32(out,in0,in1,n) ;
}
#endif



template <typename T>
int add2_nn_1n(uint64_t out,
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] + pi1[j];
    }
  }
  return 0;
}



// X = Y op Z
template<typename T0, typename T1, typename F>
int binop_dim3(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z, F op)
{
    LOG(LOG_DETAIL) << __FUNCTION__;
    T0* px = reinterpret_cast<T0*>(X.addr);
    T1 const* py = reinterpret_cast<T1*>(Y.addr);
    T1 const* pz = reinterpret_cast<T1*>(Z.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;
    int64_t const* sz = Z.dim_size;

    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        for (size_t i1 = 0; i1 < sx[1]; ++i1) {
            for (size_t i2 = 0; i2 < sx[2]; ++i2) {
                px[i0 * sx[1] * sx[2] + i1 * sx[2] + i2]
                    = op(py[(i0 % sy[0]) * sy[1] * sy[2] + (i1 % sy[1]) * sy[2] + (i2 % sy[2])],
                         pz[(i0 % sz[0]) * sz[1] * sz[2] + (i1 % sz[1]) * sz[2] + (i2 % sz[2])]);
            }
        }
    }

    return 0;
}



// X = Y op Z
template<typename TO, typename TI, typename F>
int binop_dimN(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z, F op)
{
    TO* px = reinterpret_cast<TO*>(X.addr);
    TI const* py = reinterpret_cast<TI*>(Y.addr);
    TI const* pz = reinterpret_cast<TI*>(Z.addr);

    assert(X.dims == Y.dims && X.dims == Z.dims);

    if (X.dims == 3)
        return binop_dim3<TO,TI>(X, Y, Z, op);

    LOG(LOG_DETAIL) << __FUNCTION__
        << " [" << X.nelems << "] = [" << Y.nelems << "] op [" << Z.nelems << "]";

    size_t dims = X.dims;

    size_t stX[dims];
    stX[dims - 1] = 1;
#pragma _NEC novector
    for (int dim = dims - 2; dim >= 0; --dim) {
      stX[dim] = stX[dim + 1] * X.dim_size[dim + 1];
    }

#ifdef DEBUG
    for (int dim = 0; dim < dims; ++dim)
      LOG(LOG_DEBUG) << __FUNCTION__ << " stX[" << dim << "]=" << stX[dim];
#endif

    for (size_t ix = 0; ix < X.nelems; ++ix) {
      size_t tmp = ix;
      size_t iy = 0;
      size_t iz = 0;
#pragma _NEC novector
      for (size_t dim = 0; dim < dims; ++dim) {
        size_t tmp1 = tmp / stX[dim];
        iy = (iy * Y.dim_size[dim]) + tmp1 % Y.dim_size[dim];
        iz = (iz * Z.dim_size[dim]) + tmp1 % Z.dim_size[dim];
        tmp = tmp % stX[dim];
      }
      px[ix] = op(py[iy], pz[iz]);
#ifdef DEBUG
      LOG(LOG_DEBUG) << __FUNCTION__ << " ix=" << ix << " iy=" << iy << " iz=" << iz;
#endif
    }

    return 0;
}



// X = Y op Z
// X = [d0, d1, d2, d3, d4]
// Y = [d0, d1, d2, d3, d4]
// Z = [e0, e1, e2, 1, 1]
// di >= ei

bool check_binop_dim5_x(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  return X.dims == 5 && Y.dims == 5 && Z.dims == 5
      && X.dim_size[0] == Y.dim_size[0]
      && X.dim_size[1] == Y.dim_size[1]
      && X.dim_size[2] == Y.dim_size[2]
      && X.dim_size[3] == Y.dim_size[3]
      && X.dim_size[4] == Y.dim_size[4]
      && X.dim_size[0] >= Z.dim_size[0]
      && X.dim_size[1] >= Z.dim_size[1]
      && X.dim_size[2] >= Z.dim_size[2]
      && Z.dim_size[3] == 1
      && Z.dim_size[4] == 1;
}



template<typename T, typename F>
int binop_dim5_x(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z, F op)
{
  LOG(LOG_DETAIL) << __FUNCTION__
      << " [" << X.nelems << "] = [" << Y.nelems << "] op [" << Z.nelems << "]";

  size_t n = X.dim_size[3] * X.dim_size[4];
  size_t st0[5];
  size_t st1[5];

  st0[4] = 1;
  st1[4] = 1;
  for (int dim = 3; dim >= 0; --dim) {
    st0[dim] = st0[dim + 1] * X.dim_size[dim + 1];
    st1[dim] = st1[dim + 1] * Z.dim_size[dim + 1];
  }

#if 0
  fprintf(stderr, "st0=[");
  for (int dim = 0; dim < 5; ++dim) { fprintf(stderr, " %lu", st0[dim]); }
  fprintf(stderr, " ]\n");
  fprintf(stderr, "st1=[");
  for (int dim = 0; dim < 5; ++dim) { fprintf(stderr, " %lu", st1[dim]); }
  fprintf(stderr, " ]\n");

  fprintf(stderr, "n=%lu\n", n);
#endif

#if 0
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    for (size_t i1 = 0; i1 < X.dim_size[1]; ++i1) {
      for (size_t i2 = 0; i2 < X.dim_size[2]; ++i2) {
        uint64_t out = X.addr + (i0 * st0[0] + i1 * st0[1] + i2 * st0[2]) * sizeof(T);
        uint64_t in0 = Y.addr + (i0 * st0[0] + i1 * st0[1] + i2 * st0[2]) * sizeof(T);
        uint64_t in1 = Z.addr
            + ((i0 % Z.dim_size[0]) * st1[0]
                    + (i1 % Z.dim_size[1]) * st1[1]
                    + (i2 % Z.dim_size[2]) * st1[2]) * sizeof(T);
        op(out, in0, in1, n);
      }
    }
  }
#else	// use openmp
#pragma omp parallel for
  for (size_t i012 = 0; i012 < X.dim_size[0] * X.dim_size[1] * X.dim_size[2] ; ++i012) {
    size_t i0 = i012 / (X.dim_size[1] * X.dim_size[2]) ;
    size_t i1 = (i012 % (X.dim_size[1] * X.dim_size[2])) / X.dim_size[2] ;
    size_t i2 = (i012 % X.dim_size[2]) ;

    uint64_t out = X.addr + (i0 * st0[0] + i1 * st0[1] + i2 * st0[2]) * sizeof(T);
    uint64_t in0 = Y.addr + (i0 * st0[0] + i1 * st0[1] + i2 * st0[2]) * sizeof(T);
    uint64_t in1 = Z.addr
        + ((i0 % Z.dim_size[0]) * st1[0]
                + (i1 % Z.dim_size[1]) * st1[1]
                + (i2 % Z.dim_size[2]) * st1[2]) * sizeof(T);
    op(out, in0, in1, n);
  }
#endif

  LOG(LOG_DETAIL) << __FUNCTION__ << " done";

  return 0;
}



template <typename T>
int add_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  size_t n = 16 * 64 * 8 * 8;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX = pX0 + i0 * n;
    T const* pY = pY0 + i0 * n;
    T const* pZ = pZ0;
    for (size_t i = 0; i < n; ++i) {
      pX[i] = pY[i] + pZ[i / 64];
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}



template <typename T>
int add_8x16x64x8x8_8x16x64x8x8_1x1x64x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  size_t n = 16 * 64 * 8 * 8;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX = pX0 + i0 * n;
    T const* pY = pY0 + i0 * n;
    T const* pZ = pZ0;
    for (size_t i = 0; i < n; ++i) {
      pX[i] = pY[i] + pZ[(i % (64 * 8 * 8)) / 64];
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}

} // namespace



template <typename T>
int vml::add(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("Z.dims = %ld\n", Z.dims) ;
//  for(int i=0; i<Z.dims ; i++ ) printf(" [%d] = %ld\n", i, Z.dim_size[i]) ;

  if (Y.nelems == 1) {
    return add_n1<T>(X.addr, Z.addr, Y.addr, X.nelems);
  }
  if (Z.nelems == 1) {
    return add_n1<T>(X.addr, Y.addr, Z.addr, X.nelems);
  }
  if (IsSameSize(X, Y, Z)) {
    return add_nn<T>(X.addr, Y.addr, Z.addr, Y.nelems);
  }

  // 2次元以下で配列の次元サイズを最大次元に合わせない旧呼び出しパターン用
  if (Y.dims == 2 && Z.dims == 1 && Y.dim_size[1] == Z.dim_size[0] ) {
    return add2_nn_1n<T>(X.addr, Y.addr, Z.addr, Y.dim_size[0], Y.dim_size[1]) ;
  }

  if (CheckDimsAll(X, Y, Z, 2)) {
    if (Y.dim_size[1] == Z.dim_size[1]) {
      if (Z.dim_size[0] == 1) {
	return add2_nn_1n<T>(X.addr, Y.addr, Z.addr, Y.dim_size[0], Y.dim_size[1]) ;
      }
    }
    goto general_purpose_implementation;
  }

  if (CheckDimsAll(X, Y, Z, 3)) {
    if (Z.dim_size[2] == Y.dim_size[2]) {
      if (Z.dim_size[0] == 1  &&  Z.dim_size[1] == 1) {
	return add2_nn_1n<T>(X.addr, Y.addr, Z.addr, Y.dim_size[0]*Y.dim_size[1], Y.dim_size[2]) ;
      }
      if (Y.dim_size[0] == 1  &&  Y.dim_size[1] == 1) {
	return add2_nn_1n<T>(X.addr, Z.addr, Y.addr, Z.dim_size[0]*Z.dim_size[1], Z.dim_size[2]) ;
      }
    }
    goto general_purpose_implementation;
  }

  if (CheckDimsAll(X, Y, Z, 5)) {
    if (check_dim(X, {8, 16, 64, 8, 8}) &&
	check_dim(Y, {8, 16, 64, 8, 8}) &&
	check_dim(Z, {1, 16, 64, 1, 1})) {
      return add_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1<T>(X, Y, Z);
    }
    if (check_dim(X, {8, 16, 64, 8, 8}) &&
	check_dim(Y, {8, 16, 64, 8, 8}) &&
	check_dim(Z, {1,  1, 64, 1, 1})) {
      return add_8x16x64x8x8_8x16x64x8x8_1x1x64x1x1<T>(X, Y, Z);
    }
    if (check_binop_dim5_x(X, Y, Z)) {
      return binop_dim5_x<T>(X, Y, Z, add_n1<T>);
    }
    goto general_purpose_implementation;
  }

 general_purpose_implementation:
  if (IsSameDims(X, Y, Z)) {
    return binop_dimN<T, T>(X, Y, Z,
			    [](T y, T z) -> T { return y + z; });
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int vml::add(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  if (CheckTypesAll(X, Y, Z, DT_FLOAT)) {
    return vml::add<float>(X, Y, Z);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}


namespace {

// Sub

template <typename T>
int sub_1n(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  T i0 = *reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = i0 - pi1[i];
  }
  return 0;
}



template <typename T>
int sub_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] - i1;
  }
  return 0;
}



template <typename T>
int sub_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] - pi1[i];
  }
  return 0;
}



#ifdef LIBVETF_INTRINSIC
template <>
inline int sub_nn<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  return sub_nn_f32(out,in0,in1,nelems) ;
}
#endif



template <typename T>
int sub2_nn_n1(uint64_t out, 
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] - pi1[i];
    }
  }
  return 0;
}



template <typename T>
int sub2_nn_1n(uint64_t out,
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] - pi1[j];
    }
  }
  return 0;
}



template <typename T, int M, int N>
int sub_MxN_1xN_MxN(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  T* x = reinterpret_cast<T*>(X.addr);
  T const* y = reinterpret_cast<T const*>(Y.addr);
  T const* z = reinterpret_cast<T const*>(Z.addr);

  for (int i = 0; i < M*N; ++i) {
    x[i] = y[i % N] - z[i];
  }
  return 0;
}



template <typename T>
int sub_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  size_t n = 16 * 64 * 8 * 8;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX = pX0 + i0 * n;
    T const* pY = pY0 + i0 * n;
    T const* pZ = pZ0;
    for (size_t i = 0; i < n; ++i) {
      pX[i] = pY[i] - pZ[i / 64];
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}

} // namespace



template<typename T>
int vml::sub(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{
//  printf("in0.dims = %ld\n", in0.dims) ;
//  for(int i=0; i<in0.dims ; i++ ) printf(" [%d] = %ld\n", i, in0.dim_size[i]) ;
//  printf("in1.dims = %ld\n", in1.dims) ;
//  for(int i=0; i<in1.dims ; i++ ) printf(" [%d] = %ld\n", i, in1.dim_size[i]) ;

  if (in0.nelems == 1) {
    return sub_1n<T>(out.addr, in0.addr, in1.addr, out.nelems);
  }
  if (in1.nelems == 1) {
    return sub_n1<T>(out.addr, in0.addr, in1.addr, out.nelems);
  }
  if (IsSameSize(out, in0, in1)) {
    return sub_nn<T>(out.addr, in0.addr, in1.addr, in0.nelems);
  }

  if (CheckDimsAll(out, in0, in1, 2)) {
    if (in0.dim_size[0] == in1.dim_size[0]  &&  in1.dim_size[1] == 1) {
      return sub2_nn_n1<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]);
    }
    if (in0.dim_size[1] == in1.dim_size[1]  &&  in1.dim_size[0] == 1) {
      return sub2_nn_1n<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]);
    }
    goto general_purpose_implementation;
  }

  if (CheckDimsAll(out, in0, in1, 5)) {
    if (check_dim(out, {8, 16, 64, 8, 8}) &&
	check_dim(in0, {8, 16, 64, 8, 8}) &&
	check_dim(in1, {1, 16, 64, 1, 1})) {
      return sub_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1<T>(out, in0, in1);
    }
    if (check_binop_dim5_x(out, in0, in1)) {
      return  binop_dim5_x<T>(out, in0, in1, sub_n1<T>);
    }
    if (check_dim(out, {1, 16, 64, 1, 1}) &&
	check_dim(in0, {1,  1, 64, 1, 1}) &&
	check_dim(in1, {1, 16, 64, 1, 1})) {
      return sub_MxN_1xN_MxN<T, 16, 64>(out, in0, in1);
    }
    if (check_dim(out, {1, 16, 32, 1, 1}) &&
	check_dim(in0, {1,  1, 32, 1, 1}) &&
	check_dim(in1, {1, 16, 32, 1, 1})) {
      return sub_MxN_1xN_MxN<T, 16, 32>(out, in0, in1);
    }
    if (check_dim(out, {1, 16, 16, 1, 1}) &&
	check_dim(in0, {1,  1, 16, 1, 1}) &&
	check_dim(in1, {1, 16, 16, 1, 1})) {
      return sub_MxN_1xN_MxN<T, 16, 16>(out, in0, in1);
    }
    goto general_purpose_implementation;
  }
 general_purpose_implementation:
  if (IsSameDims(out, in0, in1)) {
    return binop_dimN<T, T>(out, in0, in1,
			    [](T y, T z) -> T { return y - z; });
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int vml::sub(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{
  if (CheckTypesAll(out, in0, in1, DT_FLOAT)) {
    return vml::sub<float>(out, in0, in1);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}


namespace {

// Mul
template <typename T>
int mul_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] * i1;
  }

  return 0;
}



#ifdef LIBVETF_INTRINSIC
template <>
inline int mul_n1<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  return mul_n1_f32(out, in0, in1, n) ;
}
#endif



template <typename T>
int mul_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] * pi1[i];
  }

  return 0;
}



#ifdef LIBVETF_INTRINSIC
template <>
inline int mul_nn<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  if (n > 1024 * 1024) { // Tekito
#pragma omp parallel
    {
      // TODO: align chunk for pack
      int t = omp_get_thread_num();
      int nt = omp_get_num_threads();
      int64_t chunk = (n + nt - 1) / nt;
      uint64_t d = chunk * t * sizeof(float);
      if (chunk * (t + 1) > n)
        chunk = n - chunk * t;
      if (chunk > 0)
        mul_nn_f32(out + d, in0 + d, in1 + d, chunk);
    }
    return 0;
  } else {
    return mul_nn_f32(out, in0, in1, n) ;
  }
}
#endif



// nelems_in0 > nelems_in1
template <typename T>
int mul2_nn_n1(uint64_t out, 
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] * pi1[i];
    }
  }
  return 0;
}



template <typename T>
int mul2_nn_1n(uint64_t out,
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] * pi1[j];
    }
  }
  return 0;
}



template <typename T, int M, int N>
int mul_MxN_1xN_MxN(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  T* x = reinterpret_cast<T*>(X.addr);
  T const* y = reinterpret_cast<T const*>(Y.addr);
  T const* z = reinterpret_cast<T const*>(Z.addr);

  for (int i = 0; i < M*N; ++i) {
    x[i] = y[i % N] * z[i];
  }
  return 0;
}



template <typename T>
int mul_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  size_t n = 16 * 64 * 8 * 8;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX = pX0 + i0 * n;
    T const* pY = pY0 + i0 * n;
    T const* pZ = pZ0;
    for (size_t i = 0; i < n; ++i) {
      pX[i] = pY[i] * pZ[i / 64];
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}



template <typename T>
int mul_8x16x64x8x8_8x16x64x8x8_1x1x64x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  size_t n = 16 * 64 * 8 * 8;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX = pX0 + i0 * n;
    T const* pY = pY0 + i0 * n;
    T const* pZ = pZ0;
    for (size_t i = 0; i < n; ++i) {
      pX[i] = pY[i] * pZ[(i % (64 * 8 * 8)) / 64];
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}



template <typename T>
int mul_8x16x32x16x16_8x16x32x16x16_1x16x32x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX1 = pX0 + i0 * 16 * 32 * 16 * 16;
    T const* pY1 = pY0 + i0 * 16 * 32 * 16 * 16;
    T const* pZ = pZ0;
#pragma _NEC novector
    for (size_t i = 0; i < 16 * 32; ++i) {
      T* pX = pX1 + i * 16 * 16;
      T const* pY = pY1 + i * 16 * 16;
      for (size_t j = 0; j < 16 * 16; ++j) {
        pX[j] = pY[j] * pZ[i];
      }
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}



template <typename T>
int mul_8x16x16x32x32_8x16x16x32x32_1x16x16x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX1 = pX0 + i0 * 16 * 16 * 32 * 32;
    T const* pY1 = pY0 + i0 * 16 * 16 * 32 * 32;
    T const* pZ = pZ0;
#pragma _NEC novector
    for (size_t i = 0; i < 16 * 16; ++i) {
      T* pX = pX1 + i * 32 * 32;
      T const* pY = pY1 + i * 32 * 32;
#if 1 // faster?
      for (size_t j = 0; j < 32 * 32; ++j) {
        pX[j] = pY[j] * pZ[i];
      }
#else
      mul_n1<T>(reinterpret_cast<uint64_t>(pX),
              reinterpret_cast<uint64_t>(pY),
              reinterpret_cast<uint64_t>(pZ + i),
              16 * 16);
#endif
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}



template <typename T>
int mul_8x16x32x16x16_8x16x32x16x16_1x1x32x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX1 = pX0 + i0 * 16 * 32 * 16 * 16;
    T const* pY1 = pY0 + i0 * 16 * 32 * 16 * 16;
    T const* pZ = pZ0;
#pragma _NEC novector
    for (size_t i = 0; i < 16 * 32; ++i) {
      T* pX = pX1 + i * 16 * 16;
      T const* pY = pY1 + i * 16 * 16;
      T z = pZ[i % 32];
      for (size_t j = 0; j < 16 * 16; ++j) {
        pX[j] = pY[j] * z;
      }
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}



template <typename T>
int mul_8x16x16x32x32_8x16x16x32x32_1x1x16x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX1 = pX0 + i0 * 16 * 16 * 32 * 32;
    T const* pY1 = pY0 + i0 * 16 * 16 * 32 * 32;
    T const* pZ = pZ0;
#pragma _NEC novector
    for (size_t i = 0; i < 16 * 16; ++i) {
      T* pX = pX1 + i * 32 * 32;
      T const* pY = pY1 + i * 32 * 32;
      T z = pZ[i % 16];
      for (size_t j = 0; j < 32 * 32; ++j) {
        pX[j] = pY[j] * z;
      }
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}

} // namespace



template<typename T>
int vml::mul(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{
//  printf("in0.dims = %ld\n", in0.dims) ;
//  for(int i=0; i<in0.dims ; i++ ) printf(" [%d] = %ld\n", i, in0.dim_size[i]) ;
//  printf("in1.dims = %ld\n", in1.dims) ;
//  for(int i=0; i<in1.dims ; i++ ) printf(" [%d] = %ld\n", i, in1.dim_size[i]) ;

  if (in0.nelems == 1) {
    return mul_n1<T>(out.addr, in1.addr, in0.addr, out.nelems);
  }
  if (in1.nelems == 1) {
    return mul_n1<T>(out.addr, in0.addr, in1.addr, out.nelems);
  }
  if (IsSameSize(out, in0, in1)) {
    return mul_nn<T>(out.addr, in0.addr, in1.addr, in0.nelems);
  }

  // 2次元以下で配列の次元サイズを最大次元に合わせない旧呼び出しパターン用
  if (in0.dims == 2 && in1.dims == 1  &&  in0.dim_size[1] == in1.dim_size[0] ) {
    return mul2_nn_1n<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]) ;
  }

  if (CheckDimsAll(out, in0, in1, 2)) {
    if (in0.dim_size[1] == in1.dim_size[1]) {
      if (in1.dim_size[0] == 1) {
	return mul2_nn_1n<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]) ;
      }
    }
    if (in0.dim_size[0] == in1.dim_size[0]) {
      if (in1.dim_size[1] == 1) {
	return mul2_nn_n1<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]);
      }
      if (in0.dim_size[1] == 1) {
	return mul2_nn_n1<T>(out.addr, in1.addr, in0.addr, in1.dim_size[0], in1.dim_size[1]);
      }
    }
    goto general_purpose_implementation;
  }

  if (CheckDimsAll(out, in0, in1, 3)) {
    if (in1.dim_size[0] == 1  &&  in1.dim_size[1] == 1  &&  in1.dim_size[2] == in0.dim_size[2]) {
      return mul2_nn_1n<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0]*in0.dim_size[1], in0.dim_size[2]) ;
    }
    if (in0.dim_size[0] == 1  &&  in0.dim_size[1] == 1  &&  in0.dim_size[2] == in1.dim_size[2]) {
      return mul2_nn_1n<T>(out.addr, in1.addr, in0.addr, in1.dim_size[0]*in1.dim_size[1], in1.dim_size[2]) ;
    }
    goto general_purpose_implementation;
  }

  if (CheckDimsAll(out, in0, in1, 5)) {
    if (check_dim(out, {8, 16, 64, 8, 8}) &&
	check_dim(in0, {8, 16, 64, 8, 8}) &&
	check_dim(in1, {1, 16, 64, 1, 1})) {
      return mul_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1<T>(out, in0, in1);
    }
    if (check_dim(out, {8, 16, 64, 8, 8}) &&
	check_dim(in0, {8, 16, 64, 8, 8}) &&
	check_dim(in1, {1,  1, 64, 1, 1})) {
      return mul_8x16x64x8x8_8x16x64x8x8_1x1x64x1x1<T>(out, in0, in1);
    }
    if (check_dim(out, {8, 16, 32, 16, 16}) &&
	check_dim(in0, {8, 16, 32, 16, 16}) &&
	check_dim(in1, {1, 16, 32,  1,  1})) {
      return mul_8x16x32x16x16_8x16x32x16x16_1x16x32x1x1<T>(out, in0, in1);
    }
    if (check_dim(out, {8, 16, 16, 32, 32}) &&
	check_dim(in0, {8, 16, 16, 32, 32}) &&
	check_dim(in1, {1, 16, 16,  1,  1})) {
      return mul_8x16x16x32x32_8x16x16x32x32_1x16x16x1x1<T>(out, in0, in1);
    }
    if (check_dim(out, {8, 16, 32, 16, 16}) &&
	check_dim(in0, {8, 16, 32, 16, 16}) &&
	check_dim(in1, {1,  1, 32,  1,  1})) {
      return mul_8x16x32x16x16_8x16x32x16x16_1x1x32x1x1<T>(out, in0, in1);
    }
    if (check_dim(out, {8, 16, 16, 32, 32}) &&
	check_dim(in0, {8, 16, 16, 32, 32}) &&
	check_dim(in1, {1,  1, 16,  1,  1})) {
      return mul_8x16x16x32x32_8x16x16x32x32_1x1x16x1x1<T>(out, in0, in1);
    }
#if 0
    if (check_dim(out, {1, 16, 64, 1, 1}) &&
	check_dim(in0, {1, 16, 64, 1, 1})
	check_dim(in1, {1,  1, 64, 1, 1})) {
      return mul_MxN_1xN_MxN<T, 16, 64>(out, in1, in0);
    }
    if (check_dim(out, {1, 16, 16, 1, 1}) &&
	check_dim(in0, {1, 16, 16, 1, 1}) &&
	check_dim(in1, {1,  1, 16, 1, 1})) {
      return mul_MxN_1xN_MxN<T, 16, 16>(out, in1, in0);
    }
    if (check_dim(out, {1, 16, 32, 1, 1}) &&
	check_dim(in0, {1, 16, 32, 1, 1}) &&
	check_dim(in1, {1,  1, 32, 1, 1})) {
      return mul_MxN_1xN_MxN<T, 16, 32>(out, in1, in0);
    }
#endif
    if (check_binop_dim5_x(out, in0, in1)) {
      return binop_dim5_x<T>(out, in0, in1, mul_n1<T>);
    }
    goto general_purpose_implementation;
  }

 general_purpose_implementation:
  if (IsSameDims(out, in0, in1)) {
    return binop_dimN<T, T>(out, in0, in1,
			    [](T a, T b) -> T { return a * b; });
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int vml::mul(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{
  if (CheckTypesAll(out, in0, in1, DT_FLOAT)) {
    return vml::mul<float>(out, in0, in1);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



namespace {

// Div

template <typename T>
int div_1n(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  T i0 = *reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = i0 / pi1[i];
  }
  return 0;
}

template <typename T>
int div_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] / i1;
  }
  return 0;
}

template <typename T>
int div_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] / pi1[i];
  }
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template <>
inline int div_n1<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  return div_n1_f32(out, in0, in1, nelems) ;
}
#endif

// nelems_in0 > nelems_in1
template <typename T>
int div2_nn_n1(uint64_t out, 
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] / pi1[i];
    }
  }
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template <>
inline int div2_nn_n1<float>(uint64_t out,
                             uint64_t in0,
                             uint64_t in1,
                             size_t n0,
                             size_t n1)
{
  return div2_nn_n1_f32(out, in0, in1, n0, n1) ;
}
#endif

} // namespace

template <typename T>
int vml::div(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{

//  printf("in0.dims = %ld\n", in0.dims) ;
//  for(int i=0; i<in0.dims ; i++ ) printf(" [%d] = %ld\n", i, in0.dim_size[i]) ;
//  printf("in1.dims = %ld\n", in1.dims) ;
//  for(int i=0; i<in1.dims ; i++ ) printf(" [%d] = %ld\n", i, in1.dim_size[i]) ;

  int r=1;

  if (in0.nelems == 1) {
    /* TODO : impl intrinsic */
    return div_1n<T>(out.addr, in0.addr, in1.addr, out.nelems);
  }
  if (in1.nelems == 1) {
    return div_n1<T>(out.addr, in0.addr, in1.addr, out.nelems);
  }
  if (IsSameSize(out, in0, in1)) {
    return div_nn<T>(out.addr, in0.addr, in1.addr, out.nelems);
  }

  if (CheckDimsAll(out, in0, in1, 2)) {
    if (in0.dim_size[0] == in1.dim_size[0]  &&  in1.dim_size[1] == 1) {
      return div2_nn_n1<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]);
    }
    goto general_purpose_implementation;
  }

 general_purpose_implementation:
  if (IsSameDims(out, in0, in1)) {
    return binop_dimN<T, T>(out, in0, in1,
			    [](T y, T z) -> T { return y / z; });
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int vml::div(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{
  if (CheckTypesAll(out, in0, in1, DT_FLOAT)) {
    return vml::div<float>(out, in0, in1);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



namespace {

// DivNoNan

template <typename T>
int divnonan_1n(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  T i0 = *reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    if( pi1[i] == T(0.) ) po[i] = T(0.) ;
    else                  po[i] = i0 / pi1[i];
  }
  return 0;
}



template <typename T>
int divnonan_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  if( i1 == T(0.) ) {
    for (size_t i = 0; i < nelems; ++i) {
      po[i] = T(0.) ;
    }
  }
  else {
    for (size_t i = 0; i < nelems; ++i) {
      po[i] = pi0[i] / i1;
    }
  }
  return 0;
}



template <typename T>
int divnonan_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi1[i] == T(0.) ? T(0.) : pi0[i]/pi1[i] ;
  }

  return 0;
}



template <typename T>
int divnonan2_nn_n1(uint64_t out, 
                    uint64_t in0,
                    uint64_t in1,
                    size_t n0,
                    size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    if( pi1[i] == T(0.) ) {
      for (size_t j = 0; j < n1; ++j) {
        po[i * n1 + j] = T(0.) ;
      }
    }
    else {
      for (size_t j = 0; j < n1; ++j) {
        po[i * n1 + j] = pi0[i * n1 + j] / pi1[i];
      }
    }
  }
  return 0;
}



template <typename T>
int op_divnonan(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (args.in1.nelems == 1) {
    return divnonan_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
  }
  if (args.in0.nelems == 1) {
    return divnonan_1n<T>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
  }
  if (IsSameSize(args)) {
    return divnonan_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  } 
  if (CheckDimsAll(args, 2)) {
    if (args.in0.dim_size[0] == args.in1.dim_size[0]  &&  args.in1.dim_size[1] == 1) {
      return divnonan2_nn_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.dim_size[0], args.in0.dim_size[1]);
    }
    goto general_purpose_implementation;
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_divnonan(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {
    return op_divnonan<float>(args);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// Pow
template <typename T>
int pow_1n(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  T i0 = *reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = std::pow(i0, pi1[i]);
  }

  return 0;
}



template <typename T>
int pow_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = std::pow(pi0[i], i1);
  }

  return 0;
}



template <typename T>
int pow_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = std::pow(pi0[i], pi1[i]) ;
  }

  return 0;
}



template<typename T>
int op_pow(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  // TODO : impl other patterns
  if (args.in0.nelems == 1) {
    return pow_1n<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (args.in1.nelems == 1) {
    return pow_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (IsSameSize(args)) {
    return pow_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";
  
  return 1;
}



int op_pow(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {
    return op_pow<float>(args);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
  
  return 1;
}



// SquaredDifference
template <typename T>
int sqdiff_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    T diff = pi0[i] - i1 ;
    po[i] = diff * diff;
  }

  return 0;
}

template <typename T>
int sqdiff_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    T diff = pi0[i] - pi1[i];
    po[i] = diff * diff ;
  }

  return 0;
}

// nelems_in0 > nelems_in1
template <typename T>
int sqdiff2_nn_n1(uint64_t out,
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      T diff = pi0[i * n1 + j] - pi1[i];
      po[i * n1 + j] = diff * diff ;
    }
  }
  return 0;
}

template <typename T>
int sqdiff2_nn_1n(uint64_t out,
               uint64_t in0,
               uint64_t in1,
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      T diff = pi0[i * n1 + j] - pi1[j];
      po[i * n1 + j] = diff * diff ;
    }
  }
  return 0;
}

template <typename T>
int sqdiff_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1(
        vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z)
{
  LOG(LOG_DETAIL) << __FUNCTION__;
  size_t n = 16 * 64 * 8 * 8;
  T* pX0 = reinterpret_cast<T*>(X.addr);
  T const* pY0 = reinterpret_cast<T const*>(Y.addr);
  T const* pZ0 = reinterpret_cast<T const*>(Z.addr);
#pragma omp parallel for
  for (size_t i0 = 0; i0 < X.dim_size[0]; ++i0) {
    T* pX = pX0 + i0 * n;
    T const* pY = pY0 + i0 * n;
    T const* pZ = pZ0;
    for (size_t i = 0; i < n; ++i) {
      T diff = pY[i] - pZ[i / 64];
      pX[i] = diff * diff;
    }
  }
  LOG(LOG_DETAIL) << __FUNCTION__ << ": done";
  return 0;
}

} // namespace



template <typename T>
int vml::sqdiff(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{
//  printf("in0.dims = %ld\n", in0.dims) ;
//  for(int i=0; i<in0.dims ; i++ ) printf(" [%d] = %ld\n", i, in0.dim_size[i]) ;
//  printf("in1.dims = %ld\n", in1.dims) ;
//  for(int i=0; i<in1.dims ; i++ ) printf(" [%d] = %ld\n", i, in1.dim_size[i]) ;

  if (in0.nelems == 1) {
    return sqdiff_n1<T>(out.addr, in1.addr, in0.addr, out.nelems);
  }
  if (in1.nelems == 1) {
    return sqdiff_n1<T>(out.addr, in0.addr, in1.addr, out.nelems);
  }
  if (IsSameSize(out, in0, in1)) {
    return sqdiff_nn<T>(out.addr, in0.addr, in1.addr, in0.nelems);
  }

  if (CheckDimsAll(out, in0, in1, 2)) {
    if (in0.dim_size[0] == in1.dim_size[0]) {
      if (in1.dim_size[1] == 1) {
	return sqdiff2_nn_n1<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]);
      }
      if (in0.dim_size[1] == 1) {
	return sqdiff2_nn_n1<T>(out.addr, in1.addr, in0.addr, in1.dim_size[0], in1.dim_size[1]);
      }
    }
    if (in0.dim_size[1] == in1.dim_size[1]  &&  in1.dim_size[0] == 1) {
      return sqdiff2_nn_1n<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0], in0.dim_size[1]) ;
    }
    goto general_purpose_implementation;
  }

  if (CheckDimsAll(out, in0, in1, 3)) {
    if (in1.dim_size[0] == 1  &&  in1.dim_size[1] == 1  &&  in1.dim_size[2] == in0.dim_size[2]) {
      return sqdiff2_nn_1n<T>(out.addr, in0.addr, in1.addr, in0.dim_size[0]*in0.dim_size[1], in0.dim_size[2]) ;
    }
    if (in0.dim_size[0] == 1  &&  in0.dim_size[1] == 1  &&  in0.dim_size[2] == in1.dim_size[2]) {
      return sqdiff2_nn_1n<T>(out.addr, in1.addr, in0.addr, in1.dim_size[0]*in1.dim_size[1], in1.dim_size[2]) ;
    }
    goto general_purpose_implementation;
  }

  if (CheckDimsAll(out, in0, in1, 5)) {
    if (check_dim(out, {8, 16, 64, 8, 8}) &&
	check_dim(in0, {8, 16, 64, 8, 8}) &&
	check_dim(in1, {1, 16, 64, 1, 1})) {
      return sqdiff_8x16x64x8x8_8x16x64x8x8_1x16x64x1x1<T>(out, in0, in1);
    }
    if (check_binop_dim5_x(out, in0, in1)) {
      return binop_dim5_x<T>(out, in0, in1, sqdiff_n1<T>);
    }
    goto general_purpose_implementation;
  }

 general_purpose_implementation:
  if (IsSameDims(out, in0, in1)) {
    return binop_dimN<T, T>(out, in0, in1,
			    [](T y, T z) -> T { return (y-z)*(y-z); });
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int vml::sqdiff(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1)
{
  if (CheckTypesAll(out, in0, in1, DT_FLOAT)) {
    return vml::sqdiff<float>(out, in0, in1);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



namespace {


// RsqrtGrad
template <typename T>
int rsqrt_grad_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0  = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    T out     = pi0[i] ;
    T gradout = pi1[i] ;
    po[i] = T(-0.5) * gradout * out * out * out ;
  }

  return 0;
}



template <typename T>
int op_rsqrt_grad(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  // TODO : impl other patterns
  if (IsSameSize(args)) {
    return rsqrt_grad_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_rsqrt_grad(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {
    return op_rsqrt_grad<float>(args);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// Minimum

template <typename T>
int minimum_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] < i1 ? pi0[i] : i1;
  }
  return 0;
}



template <typename T>
int minimum_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] < pi1[i] ? pi0[i] : pi1[i];
  }
  return 0;
}



template <typename T>
int op_minimum(const BinaryOpArgs& args)
{
  if (args.in1.nelems == 1) {
    return minimum_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
  }
  if (args.in0.nelems == 1) {
    return minimum_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.out.nelems);
  }
  if (IsSameSize(args)) {
    return minimum_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_minimum(const BinaryOpArgs& args)
{
  if (CheckTypesAll(args, DT_FLOAT)) {
    return op_minimum<float>(args);
  }
  if (CheckTypesAll(args, DT_DOUBLE)) {
    return op_minimum<double>(args);
  }
  if (CheckTypesAll(args, DT_INT64)) {
    return op_minimum<int64_t>(args);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// Maximum

template <typename T>
int maximum_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] > i1 ? pi0[i] : i1;
  }
  return 0;
}



template <typename T>
int maximum_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] > pi1[i] ? pi0[i] : pi1[i];
  }
  return 0;
}



template <typename T>
int op_maximum(const BinaryOpArgs& args)
{
  if (args.in1.nelems == 1) {
    return maximum_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
  }
  if (args.in0.nelems == 1) {
    return maximum_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.out.nelems);
  }
  if (IsSameSize(args)) {
    return maximum_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_maximum(const BinaryOpArgs& args)
{
  if (CheckTypesAll(args, DT_FLOAT)) {
    return op_maximum<float>(args);
  }
  if (CheckTypesAll(args, DT_DOUBLE)) {
    return op_maximum<double>(args);
  }
  if (CheckTypesAll(args, DT_INT64)) {
    return op_maximum<int64_t>(args);
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// Equal

template <typename T>
int equal_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] == i1);
  }
  return 0;
}



template <typename T>
int equal_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] == pi1[i]);
  }
  return 0;
}



template <typename T>
int op_equal(const BinaryOpArgs& args) {
  if (args.in1.nelems == 1) {
    return equal_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (args.in0.nelems == 1) {
    return equal_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.in1.nelems);
  } 
  if (IsSameSize(args)) {
    return equal_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (IsSameDims(args)) {
    return binop_dimN<bool, T>(args.out, args.in0, args.in1,
			       [](T y, T z) -> bool { return y == z; });
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_equal(const BinaryOpArgs& args) {
  if (args.out.dtype == DT_BOOL) {
    if (args.in0.dtype == DT_FLOAT && args.in1.dtype == DT_FLOAT) {
      return op_equal<float>(args);
    }
    if (args.in0.dtype == DT_DOUBLE && args.in1.dtype == DT_DOUBLE) {
      return op_equal<double>(args);
    }
    if (args.in0.dtype == DT_INT64 && args.in1.dtype == DT_INT64) {
      return op_equal<int64_t>(args);
    }
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// NotEqual

template <typename T>
int notEqual_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] != i1);
  }
  return 0;
}



template <typename T>
int notEqual_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] != pi1[i]);
  }
  return 0;
}



template <typename T>
int op_notEqual(const BinaryOpArgs& args) {
  if (args.in1.nelems == 1) {
    return notEqual_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (args.in0.nelems == 1) {
    return notEqual_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.in1.nelems);
  }
  if (IsSameSize(args)) {
    return notEqual_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_notEqual(const BinaryOpArgs& args) {
  if (args.out.dtype == DT_BOOL) {
    if (args.in0.dtype == DT_FLOAT && args.in1.dtype == DT_FLOAT) {
      return op_notEqual<float>(args);
    }
    if (args.in0.dtype == DT_DOUBLE && args.in1.dtype == DT_DOUBLE) {
      return op_notEqual<double>(args);
    }
    if (args.in0.dtype == DT_INT64 && args.in1.dtype == DT_INT64) {
      return op_notEqual<int64_t>(args);
    }
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// Compaire operation

template <typename T>
int less_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] < i1;
  }
  return 0;
}



template <typename T>
int less_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] < pi1[i];
  }
  return 0;
}



template <typename T>
int lessEqual_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] <= i1;
  }
  return 0;
}



template <typename T>
int lessEqual_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] <= pi1[i];
  }
  return 0;
}



template <typename T>
int greater_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

#if 0 // original ( partialy vectorized )
  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] > i1;
  }
#else
  const size_t vloop_begin =  out & 0x3 ;
  const size_t vloop_end   =  n   & 0xFFFFFFFFFFFFFFFC ;

#pragma novector
  for(size_t i=0; i < vloop_begin ; i++) {
    po[i] = pi0[i] > i1;
  }

  int*  po_i = reinterpret_cast<int*>(&po[vloop_begin]);
  for(size_t j=0; j < (vloop_end - vloop_begin)>>2 ; j++) {
    const int32_t b0 = pi0[vloop_begin+4*j+0] > i1 ? 1 : 0 ;
    const int32_t b1 = pi0[vloop_begin+4*j+1] > i1 ? 1 : 0 ;
    const int32_t b2 = pi0[vloop_begin+4*j+2] > i1 ? 1 : 0 ;
    const int32_t b3 = pi0[vloop_begin+4*j+3] > i1 ? 1 : 0 ;

    const int32_t b  = (b3 << 24) | (b2 << 16) | (b1 <<8) | b0 ;
    po_i[j] = b ;
  }

#pragma novector
  for(size_t i=vloop_end; i < n ; i++) {
    po[i] = pi0[i] > i1;
  }
#endif
  return 0;
}



template <typename T>
int greater_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] > pi1[i];
  }
  return 0;
}



template <typename T>
int greaterEqual_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

#if 0 // original ( partialy vectorized )
  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] >= i1;
  }
#else
  const size_t vloop_begin =  out & 0x3 ;
  const size_t vloop_end   =  n   & 0xFFFFFFFFFFFFFFFC ;

#pragma novector
  for(size_t i=0; i < vloop_begin ; i++) {
    po[i] = pi0[i] >= i1;
  }

  int*  po_i = reinterpret_cast<int*>(&po[vloop_begin]);
  for(size_t j=0; j < (vloop_end - vloop_begin)>>2 ; j++) {
    const int32_t b0 = pi0[vloop_begin+4*j+0] >= i1 ? 1 : 0 ;
    const int32_t b1 = pi0[vloop_begin+4*j+1] >= i1 ? 1 : 0 ;
    const int32_t b2 = pi0[vloop_begin+4*j+2] >= i1 ? 1 : 0 ;
    const int32_t b3 = pi0[vloop_begin+4*j+3] >= i1 ? 1 : 0 ;

    const int32_t b  = (b3 << 24) | (b2 << 16) | (b1 <<8) | b0 ;
    po_i[j] = b ;
  }

#pragma novector
  for(size_t i=vloop_end; i < n ; i++) {
    po[i] = pi0[i] >= i1;
  }
#endif
  return 0;
}



template <typename T>
int greaterEqual_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] >= pi1[i];
  }
  return 0;
}



// Less
template <typename T>
int op_less(const BinaryOpArgs& args) {
  if (args.in1.nelems == 1) {
    return less_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (args.in0.nelems == 1) {
    return greaterEqual_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.in1.nelems);
  }
  if (IsSameSize(args)) {
    return less_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_less(const BinaryOpArgs& args) {
  if (args.out.dtype == DT_BOOL) {
    if (args.in0.dtype == DT_FLOAT && args.in1.dtype == DT_FLOAT) {
      return op_less<float>(args);
    }
    if (args.in0.dtype == DT_DOUBLE && args.in1.dtype == DT_DOUBLE) {
      return op_less<double>(args);
    }
    if (args.in0.dtype == DT_INT64 && args.in1.dtype == DT_INT64) {
      return op_less<int64_t>(args);
    }
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// LessEqual
template <typename T>
int op_lessEqual(const BinaryOpArgs& args) {
  if (args.in1.nelems == 1) {
    return lessEqual_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (args.in0.nelems == 1) {
    return greater_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.in1.nelems);
  }
  if (IsSameSize(args)) {
    return lessEqual_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_lessEqual(const BinaryOpArgs& args) {
  if (args.out.dtype == DT_BOOL) {
    if (args.in0.dtype == DT_FLOAT && args.in1.dtype == DT_FLOAT) {
      return op_lessEqual<float>(args);
    }
    if (args.in0.dtype == DT_DOUBLE && args.in1.dtype == DT_DOUBLE) {
      return op_lessEqual<double>(args);
    }
    if (args.in0.dtype == DT_INT64 && args.in1.dtype == DT_INT64) {
      return op_lessEqual<int64_t>(args);
    }
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// Greater
template <typename T>
int op_greater(const BinaryOpArgs& args) {
  if (args.in1.nelems == 1) {
    return greater_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (args.in0.nelems == 1) {
    return lessEqual_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.in1.nelems);
  }
  if (IsSameSize(args)) {
    return greater_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_greater(const BinaryOpArgs& args) {
  if (args.out.dtype == DT_BOOL) {
    if (args.in0.dtype == DT_FLOAT && args.in1.dtype == DT_FLOAT) {
      return op_greater<float>(args);
    }
    if (args.in0.dtype == DT_DOUBLE && args.in1.dtype == DT_DOUBLE) {
      return op_greater<double>(args);
    }
    if (args.in0.dtype == DT_INT64 && args.in1.dtype == DT_INT64) {
      return op_greater<int64_t>(args);
    }
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



// GreaterEqual
template <typename T>
int op_greaterEqual(const BinaryOpArgs& args) {
  if (args.in1.nelems == 1) {
    return greaterEqual_n1<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }
  if (args.in0.nelems == 1) {
    return less_n1<T>(args.out.addr, args.in1.addr, args.in0.addr, args.in1.nelems);
  }
  if (IsSameSize(args)) {
    return greaterEqual_nn<T>(args.out.addr, args.in0.addr, args.in1.addr, args.in0.nelems);
  }

 general_purpose_implementation:
  LOG(LOG_ERROR) << __FUNCTION__ << " parameter combination not supported on VE.";

  return 1;
}



int op_greaterEqual(const BinaryOpArgs& args) {
  if (args.out.dtype == DT_BOOL) {
    if (args.in0.dtype == DT_FLOAT && args.in1.dtype == DT_FLOAT) {
      return op_greaterEqual<float>(args);
    }
    if (args.in0.dtype == DT_DOUBLE && args.in1.dtype == DT_DOUBLE) {
      return op_greaterEqual<double>(args);
    }
    if (args.in0.dtype == DT_INT64 && args.in1.dtype == DT_INT64) {
      return op_greaterEqual<int64_t>(args);
    }
  }
  LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";

  return 1;
}



} // namespace


// Wrappers for TensorFlow kernel

#define DEFINE_BINARY_OP(name, FUNC) \
  extern "C" { \
  int _op_##name(const void* args, size_t len) \
{ \
  return op_Binary(args, len, FUNC, #name); \
} \
} \
REGISTER_KERNEL(#name, "_op_"#name);

// deprecated
#define DEFINE_BINARY_OP_DEP(name, FUNC) \
  extern "C" { \
  int _op_##name(const void* args, size_t len) \
{ \
  return op_Binary_(args, len, FUNC, #name); \
} \
} \
REGISTER_KERNEL(#name, "_op_"#name);

DEFINE_BINARY_OP(Add, vml::add);
DEFINE_BINARY_OP(Div, vml::div);
DEFINE_BINARY_OP(Mul, vml::mul);
DEFINE_BINARY_OP(SquaredDifference, vml::sqdiff);
DEFINE_BINARY_OP(Sub, vml::sub);

// FIXME: add vml interface as above
DEFINE_BINARY_OP_DEP(DivNoNan, op_divnonan);
DEFINE_BINARY_OP_DEP(Pow, op_pow);
DEFINE_BINARY_OP_DEP(RsqrtGrad, op_rsqrt_grad);
DEFINE_BINARY_OP_DEP(Minimum, op_minimum);
DEFINE_BINARY_OP_DEP(Maximum, op_maximum);
DEFINE_BINARY_OP_DEP(Equal, op_equal);
DEFINE_BINARY_OP_DEP(NotEqual, op_notEqual);
DEFINE_BINARY_OP_DEP(Less, op_less);
DEFINE_BINARY_OP_DEP(LessEqual, op_lessEqual);
DEFINE_BINARY_OP_DEP(Greater, op_greater);
DEFINE_BINARY_OP_DEP(GreaterEqual, op_greaterEqual);
