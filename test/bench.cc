#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

enum {
  FORMAT_NCHW = 1,
};

extern "C" {
  int op_Add(const void* args, size_t len);
  int op_BiasAdd(const void* args, size_t len);
  int op_Div(const void* args, size_t len);
  int op_Mean(const void* args, size_t len);
  int op_Mul(const void* args, size_t len);
  int op_Neg(const void* args, size_t len);
  int op_Rsqrt(const void* args, size_t len);
  int op_Sqrt(const void* args, size_t len);
  int op_Square(const void* args, size_t len);
  int op_Sub(const void* args, size_t len);
  int op_Sum(const void* args, size_t len);
  int op_Tile(const void* args, size_t len);
}

static double second()
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

template <typename T> struct to_dtype {};
template<> struct to_dtype<float> { static const int val = 1; };

template <typename T>
int check(T const* a, T const* b, size_t n, int verbose = 0)
{
  int flag = 1;
  for (int j = 0; j < n; ++j) {
    if (a[j] != b[j]) {
      flag = 0;
      if (verbose > 0) {
        fprintf(stderr, "a %18.12e b %18.12e diff %18.12e\n", 
                a[j], b[j], std::abs(a[j] - b[j]));
      }
    }
  }
  return flag;
}

struct Bench
{
  Bench(std::string name) 
    : name_(name), data_size_(0), flop_count_(0) {}

  std::string name() const { return name_; }
  virtual int validate(int verbose) = 0;
  virtual int run() = 0;
  
  std::string name_;

  size_t data_size_; // bytes
  size_t flop_count_;
};

void run_bench(Bench& bench, int ntimes = 1)
{
  double t0 = second();
  for (int i = 0; i < ntimes; ++i) {
    int ret = bench.run();
    if (ret != 0)
      fprintf(stderr, "ret=%d\n", ret);
  }
  double t1 = second();
  double sec = (t1 - t0) / ntimes;
  double flops = bench.flop_count_ / sec;
  double bw = bench.data_size_ / sec;

  fprintf(stderr, "%-20s %8.3lf ms %8.3lf GFlops %8.3lf GB/s\n",
          bench.name_.c_str(), sec*1e3, flops/1e9, bw/1e9);
}

namespace ref
{

// UnaryOp

template <typename T>
int neg(T* x, T const* y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = - y[i];
}

template <typename T>
int sqrt(T* x, T const* y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = std::sqrt(y[i]);
}

template <typename T>
int rsqrt(T* x, T const* y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = T(1.0) / std::sqrt(y[i]);
}

template <typename T>
int square(T* x, T const* y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = y[i] * y[i];
}

//
// BinaryOp
//

template <typename T>
int add(T* x, T const* y, T const* z, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = y[i] + z[i];
  return 0;
}

template <typename T>
int sub(T* x, T const* y, T const* z, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = y[i] - z[i];
  return 0;
}

template <typename T>
int mul(T* x, T const* y, T const* z, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = y[i] * z[i];
  return 0;
}

template <typename T>
int div(T* x, T const* y, T const* z, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = y[i] / z[i];
  return 0;
}

// Reduction

template <typename T>
int mean_d2a0(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t j = 0; j < dim1; ++j) {
        T s = T(0);
        for (size_t i = 0; i < dim0; ++i) {
            s += pi[i * dim1 + j];
        }
        po[j] = s / dim0 ;
    }

    return 0;
}

template <typename T>
int sum_d2a0(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t j = 0; j < dim1; ++j) {
        T s = T(0);
        for (size_t i = 0; i < dim0; ++i) {
            s += pi[i * dim1 + j];
        }
        po[j] = s;
    }

    return 0;
}

#if 0
template <typename T>
int sum_d2a0(T* x, T const* y, size_t m, size_t n)
{
#pragma _NEC novector
  for (size_t i = 0; i < m; ++i) {
    T s = T(0);
#pragma _NEC novector
    for (size_t j = 0; j < n; ++j)
      s += y[i * n + j];
    x[i] = s;
  }
  return 0;
}
#endif

template<typename T>
int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias,
                 int batch, int width, int height, int channel)
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

  return 0;
}

// Tile

struct Tensor {
  int32_t dtype;
  uint64_t addr;
  int32_t dims;
  int64_t nelems;
  int64_t dim_size[8];

  size_t size() const {
    return sizeof(Tensor) + sizeof(int64_t) * (dims - 1);
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


template<typename T>
int tile_dim5_11(Tensor const& X, Tensor const& Y)
{
    //LOG(3) << __FUNCTION__;
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;

    //printf("tile5_11: x %d %d %d %d %d\n",sx[0],sx[1],sx[2],sx[3],sx[4],sx[5]);
    //printf("tile5_11: y %d %d %d %d %d\n",sy[0],sy[1],sy[2],sy[3],sy[4],sy[5]);

#pragma _NEC novector
    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        const size_t ix0 = i0 ;
        const size_t iy0 = i0 % sy[0] ;
#pragma _NEC novector
        for (size_t i1 = 0; i1 < sx[1]; ++i1) {
            const size_t ix1 = ix0 * sx[1] + i1 ;
            const size_t iy1 = iy0 * sy[1] + (i1 % sy[1]) ;
#pragma _NEC novector
            for (size_t i2 = 0; i2 < sx[2]; ++i2) {
                const size_t ix2 = ix1 * sx[2] + i2 ;
                const size_t iy2 = iy1 * sy[2] + (i2 % sy[2]) ;
                for (size_t i34 = 0; i34 < sx[3] * sx[4] ; ++i34) {
                    const size_t ix34 = ix2 * sx[3] * sx[4] + i34 ;
                    px[ix34] = py[iy2] ;
                }
            }
        }
    }

    return 0;
}

} // namespace ref

template <typename T>
class UnaryOpBench : public Bench
{
  public:
    UnaryOpBench(std::string name, 
                 int (*op)(const void* args, size_t len),
                 int (*ref_op)(T*, T const*, size_t),
                 T const* y, size_t n) 
      : Bench(name), op_(op), ref_op_(ref_op), y_(y), n_(n) { 
        this->data_size_ = sizeof(T) * n * 2;

        x0_ = new T[n];
        x1_ = new T[n];

        args_.in.dtype = to_dtype<T>::val;
        args_.in.data_format = -1; // do not use
        args_.in.addr = reinterpret_cast<uint64_t>(x0_);
        args_.in.dims = 1;
        args_.in.nelems = n;
        args_.in.dim_size[0] = n;

        args_.out.dtype = to_dtype<T>::val;
        args_.out.data_format = -1; // do not use
        args_.out.addr = reinterpret_cast<uint64_t>(y);
        args_.out.dims = 1;
        args_.out.nelems = n;
        args_.out.dim_size[0] = n;
      }

    int validate(int verbose) {
      memset(x0_, 0, sizeof(T) * n_);
      memset(x1_, 0, sizeof(T) * n_);
      run();
      ref_op_(x1_, y_, n_);
      return check(x0_, x1_, n_, verbose);
    }

    int run() {
      return op_(reinterpret_cast<const void*>(&args_), sizeof(Args));
    }

  private:
    struct _Tensor {
      int dtype;
      int data_format;
      uint64_t addr;
      int32_t dims;
      int64_t nelems;
      int64_t dim_size[8];
    };

    struct Args {
      _Tensor in;
      _Tensor out;
    } args_;

    T* x0_;
    T* x1_;
    T const* y_;
    size_t n_;

    int (*op_)(const void* args, size_t len);
    int (*ref_op_)(T* x, T const* y, size_t n);
};

template <typename T>
class BinaryOpBench : public Bench 
{
  public:
    BinaryOpBench(std::string name, 
                  int (*op)(const void* args, size_t len),
                  int (*ref_op)(T*, T const*, T const*, size_t),
                  T const* y, T const* z, size_t n) 
      : Bench(name), op_(op), ref_op_(ref_op), y_(y), z_(z), n_(n) { 
        x0_ = new T[n];
        x1_ = new T[n];

        args_.in0 = mktensor(y, n);
        args_.in1 = mktensor(z, n);
        args_.out = mktensor(x0_, n);

        data_size_ = n * 3 * sizeof(T);
        flop_count_ = n;
      }

    int validate(int verbose) override {
      memset(x0_, 0, sizeof(T) * n_);
      memset(x1_, 0, sizeof(T) * n_);
      int ret = run();
      if (ret != 0)
        fprintf(stderr, "ret=%d\n", ret);
      ref_op_(x1_, y_, z_, n_);
      return check(x0_, x1_, n_, verbose);
    }

    int run() override {
      return op_(reinterpret_cast<void const*>(&args_), sizeof(BinaryOpArgs));
    }

  private:
    T* x0_;
    T* x1_;
    T const* y_;
    T const* z_;
    size_t n_;
    int (*op_)(const void* args, size_t len);
    int (*ref_op_)(T*, T const*, T const*, size_t);

    // copy from src/binary_ops.cc
    struct _Tensor {
      int dtype;
      uint64_t addr;
      int32_t dims;
      int64_t nelems;
      int64_t dim_size[8];

      std::string to_s() const {
        std::stringstream s;

        s << "[dtype=" << dtype
          << ",dims=" << dims
          << "[";
        for (int i = 0; i < dims; ++i)
          s << " " << dim_size[i];
        s  << " ],nelems=" << nelems
          << "]";
        return s.str();
      }
    };

    struct BinaryOpArgs {
      _Tensor in0;
      _Tensor in1;
      _Tensor out;
    };

    _Tensor mktensor(float const* x, int n)
    {
      _Tensor t;
      t.dtype = to_dtype<T>::val;
      t.addr = reinterpret_cast<uint64_t>(x);
      t.dims = 1;
      t.nelems = n;
      t.dim_size[0] = n;
      return t;
    }

    BinaryOpArgs args_;
};

template <typename T>
class ReductionOpBench : public Bench
{
  public:
    ReductionOpBench(std::string name, 
                     int (*op)(const void* args, size_t len),
                     int (*ref_op)(uint64_t, uint64_t, size_t, size_t),
                     T const* y, size_t n) 
      : Bench(name), op_(op), ref_op_(ref_op), y_(y), n_(n) { 
        data_size_ = n * 2 * sizeof(T);
        flop_count_ = n;

        x0_ = new T[n];
        x1_ = new T[n];

        args_.dtype = to_dtype<T>::val;
        args_.ndims = 2;
        args_.in = reinterpret_cast<uint64_t>(y);
        args_.out = reinterpret_cast<uint64_t>(x0_);
        args_.dim_size[0] = 1;
        args_.dim_size[1] = n;
        args_.axis = 0;
      }

    int run() override {
      return op_(reinterpret_cast<void const*>(&args_), sizeof(Args));
    }

    int validate(int verbose) override {
      x0_[0] = 0;
      x1_[0] = 0;
      run();
      ref_op_(reinterpret_cast<uint64_t>(x1_), 
              reinterpret_cast<uint64_t>(y_), 1, n_);
      return check(x0_, x1_, 1, verbose);
    }

  private:
    struct Args {
      int dtype;
      int ndims;
      uint64_t in;
      uint64_t out;
      int64_t dim_size[3];
      int axis;
    } args_;

    int (*op_)(const void* args, size_t len);
    int (*ref_op_)(uint64_t, uint64_t, size_t, size_t);

    T* x0_;
    T* x1_;
    T const* y_;
    size_t n_;
};

template <typename T>
class BiasAddOpBench : public Bench
{
  public:
    BiasAddOpBench(size_t nchw[4]) : Bench("BiasAdd") {
      memcpy(nchw_, nchw, sizeof(size_t) * 4);
      size_t szio = nchw[0] * nchw[1] * nchw[2] * nchw[3];
      size_t szb = nchw[1];
      this->data_size_ =  (szio * 2 + szb) * sizeof(T);
      this->flop_count_ = szio;

      in_ = new T[szio];
      out0_ = new T[szio];
      out1_ = new T[szio];
      bias_ = new T[szb];

      for (size_t i = 0; i < szio; ++i)
        in_[i] = T(drand48());

      for (size_t i = 0; i < szb; ++i)
        bias_[i] = T(drand48());

      szio_ = szio;

      args_.dtype = to_dtype<float>::val;
      args_.data_format = FORMAT_NCHW;
      args_.in = reinterpret_cast<uint64_t>(in_);
      args_.bias = reinterpret_cast<uint64_t>(bias_);
      args_.out = reinterpret_cast<uint64_t>(out0_);
      args_.batch = nchw[0];
      args_.width = nchw[1];
      args_.height = nchw[2];
      args_.channel = nchw[3];
    }

    int validate(int verbose) override {
      memset(out0_, 0, sizeof(T) * szio_);
      memset(out1_, 0, sizeof(T) * szio_);
      run();
      ref::BiasAdd_NCHW<float>(reinterpret_cast<uint64_t>(out1_),
                               reinterpret_cast<uint64_t>(in_),
                               reinterpret_cast<uint64_t>(bias_),
                               nchw_[0], nchw_[1], nchw_[2], nchw_[3]);
      return check(out0_, out1_, szio_, verbose);
    }

    int run() override {
      return op_BiasAdd(reinterpret_cast<void const*>(&args_), sizeof(Args));
    }

  private:
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
    } args_;

    size_t nchw_[4];
    size_t szio_;

    T* in_;
    T* bias_;
    T* out0_;
    T* out1_;
};

template <typename T>
class TileOpBench : public Bench
{
  public:
    TileOpBench() : Bench("Tile") 
    {
      size_t ndims = 5;
      size_t dimsIn[5] = {8, 16, 16, 1, 1};
      size_t dimsOut[5] = {8, 16, 16, 32, 32};
      size_t szIn = 1;
      size_t szOut = 1;
      for (size_t i = 0; i < ndims; ++i) {
        szIn *= dimsIn[i];
        szOut *= dimsOut[i];
      }

      x0_ = new T[szOut];
      x1_ = new T[szOut];
      y_ = new T[szIn];

      szOut_ = szOut;

      in_.dtype = to_dtype<T>::val;
      in_.addr = reinterpret_cast<uint64_t>(y_);
      in_.dims = ndims;
      in_.nelems = szIn;
      for (size_t i = 0; i < ndims; ++i)
        in_.dim_size[i] = dimsIn[i];

      out0_.dtype = to_dtype<T>::val;
      out0_.addr = reinterpret_cast<uint64_t>(x0_);
      out0_.dims = ndims;
      out0_.nelems = szOut;
      for (size_t i = 0; i < ndims; ++i)
        out0_.dim_size[i] = dimsOut[i];

      out1_.dtype = to_dtype<T>::val;
      out1_.addr = reinterpret_cast<uint64_t>(x1_);
      out1_.dims = ndims;
      out1_.nelems = szOut;
      for (size_t i = 0; i < ndims; ++i)
        out1_.dim_size[i] = dimsOut[i];

      len_ = (sizeof(ref::Tensor) + sizeof(size_t)) * 2 + sizeof(int64_t);
      buf_ = new char[len_];
      uintptr_t addr = reinterpret_cast<uintptr_t>(buf_);
      *reinterpret_cast<int64_t*>(addr) = 2;
      addr += sizeof(int64_t);
      *reinterpret_cast<size_t*>(addr) = sizeof(ref::Tensor);
      addr += sizeof(size_t);
      *reinterpret_cast<ref::Tensor*>(addr) = in_;
      addr += sizeof(ref::Tensor);
      *reinterpret_cast<size_t*>(addr) = sizeof(ref::Tensor);
      addr += sizeof(size_t);
      *reinterpret_cast<ref::Tensor*>(addr) = out0_;
    }

    int validate(int verbose) override {
      memset(x0_, 0, sizeof(T) * szOut_);
      memset(x1_, 0, sizeof(T) * szOut_);
      run();
      ref::tile_dim5_11<float>(out1_, in_);
      return check(x0_, x1_, szOut_, verbose);
    }

    int run() override {
      op_Tile(buf_, len_);
    }

  private:
    void* buf_;
    size_t len_;
#if 0
    struct Tensor {
      int32_t dtype;
      uint64_t addr;
      int32_t dims;
      int64_t nelems;
      int64_t dim_size[1];

      size_t size() const {
        return sizeof(Tensor) + sizeof(int64_t) * (dims - 1);
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
#endif

    T* x0_;
    T* x1_;
    T* y_;

    size_t szOut_;

    ref::Tensor in_;
    ref::Tensor out0_;
    ref::Tensor out1_;
};

template <typename T>
void add_bench(std::vector<Bench*>& v, size_t n)
{
  T* y = new T[n];
  T* z = new T[n];

  for (int i = 0; i < n; ++i) {
    y[i] = drand48();
    z[i] = drand48();
  }

  v.push_back(new BinaryOpBench<float>("Add", op_Add, ref::add<float>, y, z, n));
  v.push_back(new BinaryOpBench<float>("Sub", op_Sub, ref::sub<float>, y, z, n));
  v.push_back(new BinaryOpBench<float>("Mul", op_Mul, ref::mul<float>, y, z, n));
  v.push_back(new BinaryOpBench<float>("Div", op_Div, ref::div<float>, y, z, n));

  v.push_back(new ReductionOpBench<float>("Mean", op_Mean, ref::mean_d2a0<float>, y, n));
  v.push_back(new ReductionOpBench<float>("Sum", op_Sum, ref::sum_d2a0<float>, y, n));

  v.push_back(new UnaryOpBench<float>("Sqrt", op_Sqrt, ref::sqrt, y, n));
  v.push_back(new UnaryOpBench<float>("Rsqrt", op_Rsqrt, ref::rsqrt, y, n));
  v.push_back(new UnaryOpBench<float>("Square", op_Square, ref::square, y, n));
  v.push_back(new UnaryOpBench<float>("Neg", op_Neg, ref::neg, y, n));

#if 0
  delete[] y;
  delete[] z;
#endif
}

int main(int argc, char* argv[])
{
  size_t n = 20000000;
  size_t nchw[4] = {256, 16, 64, 64};
  int repeat = 10;
  int verbose = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-n") == 0) {
      n = strtoul(argv[++i], NULL, 0);
    } else if (strcmp(argv[i], "--nchw") == 0) {
      const char* tmp0 = argv[++i];
      char* tmp1;
      nchw[0] = strtoul(tmp0, &tmp1, 0);
      tmp0 = ++tmp1;
      nchw[1] = strtoul(tmp0, &tmp1, 0);
      tmp0 = ++tmp1;
      nchw[2] = strtoul(tmp0, &tmp1, 0);
      tmp0 = ++tmp1;
      nchw[3] = strtoul(tmp0, &tmp1, 0);
      fprintf(stderr, "nchw=%lu,%lu,%lu,%lu\n", nchw[0], nchw[1], nchw[2], nchw[3]);
    } else if (strcmp(argv[i], "-r") == 0) {
      repeat = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-v") == 0) {
      ++verbose;
    }
  }

  fprintf(stderr, "n=%lu\n", n);
  fprintf(stderr, "nchw=%lu,%lu,%lu,%lu(%lu)\n", 
          nchw[0], nchw[1], nchw[2], nchw[3], nchw[0] * nchw[1] * nchw[2] * nchw[3]);

  std::vector<Bench*> v;

  add_bench<float>(v, n);
  v.push_back(new BiasAddOpBench<float>(nchw));
  v.push_back(new TileOpBench<float>());

  int flag = 1;
  for (Bench* b : v) {
    int tmp = b->validate(verbose);
    flag &= tmp;
    fprintf(stderr, "Validation: %-20s %s\n", b->name(), tmp ? "OK" : "NG");
  }

  fprintf(stderr, "flag=%d\n", flag);

  if (!flag)
    return 1;

  for (Bench* b : v) {
    run_bench(*b, repeat);
  }

  return 0;
}
