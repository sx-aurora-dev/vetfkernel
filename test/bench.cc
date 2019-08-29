#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

#include <vml.h>

enum {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
};

extern "C" {
  int _op_Add(const void* args, size_t len);
  int op_ApplyAdam(const void* args, size_t len);
  int op_BiasAdd(const void* args, size_t len);
  int op_BiasAddGrad(const void* args, size_t len);
  int op_Div(const void* args, size_t len);
  int op_Mean(const void* args, size_t len);
  int _op_Mul(const void* args, size_t len);
  //int op_Neg(const void* args, size_t len);
  int op_Rsqrt(const void* args, size_t len);
  //int op_Sqrt(const void* args, size_t len);
  int op_Square(const void* args, size_t len);
  int _op_Sub(const void* args, size_t len);
  int op_Sum(const void* args, size_t len);
  int op_Tile(const void* args, size_t len);
  int op_Transpose(const void* args, size_t len);
}

static double second()
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

template <typename T> struct to_dtype {};
template<> struct to_dtype<float> { static const int val = 1; };

struct BenchOpts
{
  int verbose;
  double threshold;
};

template <typename T>
int check(T const* a, T const* b, size_t n,
          BenchOpts const& opts, double threshold)
{
  int flag = 1;
  for (int i = 0; i < n; ++i) {
    double diff = a[i] - b[i];
    if (diff != 0.0) {
      double err;
      if (a[i] == 0.0 && b[i] == 0.0) {
        err = diff;
      } else {
        err = std::sqrt(diff * diff / (a[i] * a[i] + b[i] * b[i]));
      }

      if (err > threshold) {
        flag = 0;
        if (opts.verbose > 1) {
          fprintf(stderr, "a[%d] %18.12e b[%d] %18.12e diff %18.12e err %18.12e\n", 
                  i, a[i], i, b[i], diff, err);
        }
      }
    }
  }
  return flag;
}

template <typename T>
int check_exact(T const* a, T const* b, size_t n, BenchOpts const& opts)
{
  return check(a, b, n, opts, 0.0);
}

template <typename T>
int check(T const* a, T const* b, size_t n, BenchOpts const& opts)
{
  return check(a, b, n, opts, opts.threshold);
}

#if 0
template <typename T>
int check_exact(T const* a, T const* b, size_t n, BenchOpts const& opts)
{
  int flag = 1;
  for (int i = 0; i < n; ++i) {
    double diff = a[i] - b[i];
    if (diff != 0.0) {
      double err;
      if (a[i] == 0.0 && b[i] == 0.0) {
        err = diff;
      } else {
        err = std::sqrt(diff * diff / (a[i] * a[i] + b[i] * b[i]));
      }

      if (err > opts.threshold) {
        flag = 0;
        if (opts.verbose > 1) {
          fprintf(stderr, "a[%d] %18.12e b[%d] %18.12e diff %18.12e err %18.12e\n", 
                  i, a[i], i, b[i], diff, err);
        }
      }
    }
  }
  return flag;
}
#endif

struct Bench
{
  Bench(std::string name) 
    : name_(name), data_size_(0), flop_count_(0) {}

  std::string name() const { return name_; }
  virtual int validate(BenchOpts const&) = 0;
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

//
// BiasAdd
//

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

  return 0;
}

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

template<typename T>
int BiasAddGrad_NHWC(uint64_t output, uint64_t output_backprop,
                     int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(output);
  const T* pin = reinterpret_cast<const T*>(output_backprop);

  memset(pout, 0, sizeof(T) * channel);

#pragma _NEC novector
  for (int b = 0; b < batch; ++b) {
#pragma _NEC novector
    for (int y = 0; y < height; ++y) {
#pragma _NEC novector
      for (int x = 0; x < width; ++x) {
#pragma _NEC novector
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

  return 0;
}

template<typename T>
int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop,
                     int batch, int width, int height, int channel)
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

//
// Transpopse
//

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

template <typename T>
int apply_adam(bool use_nesterov, int64_t num_elements,
               uint64_t var_ptr, uint64_t m_ptr, uint64_t v_ptr,
               uint64_t beta1_power_ptr, uint64_t beta2_power_ptr,
               uint64_t lr_ptr,
               uint64_t beta1_ptr, uint64_t beta2_ptr, uint64_t epsilon_ptr,
               uint64_t grd_ptr )
{
  T* var = reinterpret_cast<T*>(var_ptr);
  T* m   = reinterpret_cast<T*>(m_ptr);
  T* v   = reinterpret_cast<T*>(v_ptr);

  const T* grd = reinterpret_cast<const T*>(grd_ptr);

  const T beta1_power = reinterpret_cast<const T*>(beta1_power_ptr)[0];
  const T beta2_power = reinterpret_cast<const T*>(beta2_power_ptr)[0];
  const T lr = reinterpret_cast<const T*>(lr_ptr)[0];
  const T beta1 = reinterpret_cast<const T*>(beta1_ptr)[0];
  const T beta2 = reinterpret_cast<const T*>(beta2_ptr)[0];
  const T epsilon = reinterpret_cast<const T*>(epsilon_ptr)[0];

  const T one = T(1.) ; 

#if 0 // optimized
 
  const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;

#pragma omp parallel
  { 
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t eachNElement = num_elements / nthreads ;
    int64_t remain       = num_elements % nthreads ;

    int64_t elementBegin = eachNElement * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myElement    = eachNElement + ( threadid < remain ? 1 : 0 ) ;

    if( use_nesterov ) {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
        var[i] -= k * ( m[i] * beta1 + (one-beta1) * grd[i] ) / ( epsilon + std::sqrt(v[i])) ;
      }
    }
    else {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
        var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
      }
    }
  }
#else // original
  for(int64_t i=0; i<num_elements; i++) {
    m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
  }
  for(int64_t i=0; i<num_elements; i++) {
    v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
  }
  
  const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;
  if( use_nesterov ) {
    for(int64_t i=0; i<num_elements; i++) {
      var[i] -= k * ( m[i] * beta1 + (one-beta1) * grd[i] ) / ( epsilon + std::sqrt(v[i])) ;
    }
  }
  else {
    for(int64_t i=0; i<num_elements; i++) {
      var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
    }
  }
#endif

  return 0 ;
}


} // namespace ref

template <typename T>
class UnaryOpBench : public Bench
{
  public:
    UnaryOpBench(std::string name, 
                 int (*op)(vml::Tensor const& out, vml::Tensor const& in),
                 int (*ref_op)(T*, T const*, size_t),
                 T const* y, size_t n) 
      : Bench(name), op_(op), ref_op_(ref_op), y_(y), n_(n) { 
        this->data_size_ = sizeof(T) * n * 2;

        x0_ = new T[n];
        x1_ = new T[n];

        in_.dtype = to_dtype<T>::val;
        in_.addr = reinterpret_cast<uint64_t>(x0_);
        in_.dims = 1;
        in_.nelems = n;
        in_.dim_size[0] = n;

        out_.dtype = to_dtype<T>::val;
        out_.addr = reinterpret_cast<uint64_t>(y);
        out_.dims = 1;
        out_.nelems = n;
        out_.dim_size[0] = n;
      }

    int validate(BenchOpts const& opts) override {
      memset(x0_, 0, sizeof(T) * n_);
      memset(x1_, 0, sizeof(T) * n_);
      run();
      ref_op_(x1_, y_, n_);
      return check_exact(x0_, x1_, n_, opts);
    }

    int run() {
      return op_(out_, in_);
    }

  private:
    vml::Tensor in_;
    vml::Tensor out_;

    T* x0_;
    T* x1_;
    T const* y_;
    size_t n_;

    int (*op_)(vml::Tensor const& out, vml::Tensor const& in);
    //int (*op_)(const void* args, size_t len);
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

    int validate(BenchOpts const& opts) override {
      memset(x0_, 0, sizeof(T) * n_);
      memset(x1_, 0, sizeof(T) * n_);
      int ret = run();
      if (ret != 0)
        fprintf(stderr, "ret=%d\n", ret);
      ref_op_(x1_, y_, z_, n_);
      return check_exact(x0_, x1_, n_, opts);
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

    struct BinaryOpArgs {
      vml::Tensor in0;
      vml::Tensor in1;
      vml::Tensor out;
    };

    vml::Tensor mktensor(float const* x, int n)
    {
      vml::Tensor t;
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

    int validate(BenchOpts const& opts) override {
      x0_[0] = 0;
      x1_[0] = 0;
      run();
      ref_op_(reinterpret_cast<uint64_t>(x1_), 
              reinterpret_cast<uint64_t>(y_), 1, n_);
      return check_exact(x0_, x1_, 1, opts);
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
    BiasAddOpBench(std::string name, int data_format, 
                   int (*ref_op)(uint64_t out, uint64_t in, 
                                 uint64_t bias, int batch, int width, int height, int channel),
                   T const* in, size_t nchw[4])
      : Bench(name), ref_op_(ref_op), in_(in) {
      memcpy(nchw_, nchw, sizeof(size_t) * 4);
      size_t szio = nchw[0] * nchw[1] * nchw[2] * nchw[3];
      size_t szb = nchw[1];
      this->data_size_ =  (szio * 2 + szb) * sizeof(T);
      this->flop_count_ = szio;

      out0_ = new T[szio];
      out1_ = new T[szio];
      bias_ = new T[szb];

      for (size_t i = 0; i < szb; ++i)
        bias_[i] = T(drand48());

      szio_ = szio;

      args_.dtype = to_dtype<float>::val;
      args_.data_format = data_format;
      args_.in = reinterpret_cast<uint64_t>(in_);
      args_.bias = reinterpret_cast<uint64_t>(bias_);
      args_.out = reinterpret_cast<uint64_t>(out0_);
      args_.batch = nchw[0];
      args_.width = nchw[3];
      args_.height = nchw[2];
      args_.channel = nchw[1];
    }

    int validate(BenchOpts const& opts) override {
      memset(out0_, 0, sizeof(T) * szio_);
      memset(out1_, 0, sizeof(T) * szio_);
      run();
      ref_op_(reinterpret_cast<uint64_t>(out1_),
              reinterpret_cast<uint64_t>(in_),
              reinterpret_cast<uint64_t>(bias_),
              nchw_[0], nchw_[3], nchw_[2], nchw_[1]);
      return check_exact(out0_, out1_, szio_, opts);
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

    T const* in_;
    T* bias_;
    T* out0_;
    T* out1_;

    int (*ref_op_)(uint64_t out, uint64_t in, 
                   uint64_t bias, int batch, int width, int height, int channel);
};

template <typename T>
class BiasAddGradOpBench : public Bench
{
  public:
    BiasAddGradOpBench(std::string name, int data_format,
                       int (*ref_op)(uint64_t output, uint64_t output_backprop,
                                     int batch, int width, int height, int channel),
                       T const* in, size_t nchw[4]) 
      : Bench(name), ref_op_(ref_op), in_(in) {
      memcpy(nchw_, nchw, sizeof(size_t) * 4);
      size_t szb = nchw[1];

      output0_ = new T[szb];
      output1_ = new T[szb];

      szb_ = szb;

      args_.dtype = to_dtype<T>::val;
      args_.data_format = data_format;
      args_.output_backprop = reinterpret_cast<uint64_t>(in_);
      args_.output = reinterpret_cast<uint64_t>(output0_);
      args_.batch = nchw[0];
      args_.width = nchw[3];
      args_.height = nchw[2];
      args_.channel = nchw[1];
    }

    int validate(BenchOpts const& opts) override {
      memset(output0_, 0, sizeof(T) * szb_);
      memset(output1_, 0, sizeof(T) * szb_);
      run();
      ref_op_(reinterpret_cast<uint64_t>(output1_),
              reinterpret_cast<uint64_t>(in_),
              nchw_[0], nchw_[3], nchw_[2], nchw_[1]);
      //fprintf(stderr, "%f %f\n", output0_[0], output1_[0]);
      return check(output0_, output1_, szb_, opts);
    }

    int run() override {
      return op_BiasAddGrad(reinterpret_cast<void const*>(&args_), sizeof(Args));
    }

  private:
    struct Args{
      int dtype;
      int data_format;
      uint64_t output_backprop;
      uint64_t output;
      int batch;
      int width;
      int height;
      int channel;
    } args_;

    size_t nchw_[4];
    size_t szb_;

    T const* in_;
    T* output0_;
    T* output1_;

    int (*ref_op_)(uint64_t output, uint64_t output_backprop,
                   int batch, int width, int height, int channel);
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

    int validate(BenchOpts const& opts) override {
      memset(x0_, 0, sizeof(T) * szOut_);
      memset(x1_, 0, sizeof(T) * szOut_);
      run();
      ref::tile_dim5_11<float>(out1_, in_);
      return check_exact(x0_, x1_, szOut_, opts);
    }

    int run() override {
      op_Tile(buf_, len_);
    }

  private:
    void* buf_;
    size_t len_;

    T* x0_;
    T* x1_;
    T* y_;

    size_t szOut_;

    ref::Tensor in_;
    ref::Tensor out0_;
    ref::Tensor out1_;
};

template <typename T>
class TransposeOpBench : public Bench
{
  public:
    TransposeOpBench(std::string name, 
                     int (*ref_op)(uint64_t, uint64_t, const int32_t*),
                     T const* y,
                     size_t dims[4], std::vector<int> perm) 
      : Bench(name), ref_op_(ref_op), y_(y) {
        int ndims = 4;
        nelems_ = 1;
        for (size_t i = 0; i < ndims; ++i) {
          args_.dim_size[i] = dims[i];
          args_.perm[i] = perm[i];
          nelems_ *= dims[i];
        }

        x0_ = new T[nelems_];
        x1_ = new T[nelems_];

        args_.dtype = to_dtype<T>::val;
        args_.in = reinterpret_cast<uint64_t>(y_);
        args_.out = reinterpret_cast<uint64_t>(x0_);
        args_.size = ndims;
      }

    int validate(BenchOpts const& opts) override {
      memset(x0_, 0, sizeof(T) * nelems_);
      memset(x1_, 0, sizeof(T) * nelems_);
      run();
      ref_op_(reinterpret_cast<uint64_t>(x1_),
              reinterpret_cast<uint64_t>(y_), args_.dim_size);
      return check_exact(x0_, x1_, nelems_, opts);
    }

    int run() override {
      return op_Transpose(reinterpret_cast<const void*>(&args_), sizeof(Args));
    }

  private:
    struct Args {
      int dtype;
      uint64_t in;
      uint64_t out;
      int size;
      int32_t dim_size[4]; // in
      int32_t perm[4];
    } args_;

    int (*ref_op_)(uint64_t, uint64_t, const int32_t*);

    T* x0_;
    T* x1_;
    T const* y_;
    size_t nelems_;
};

template <typename T>
class ApplyAdamOpBench : public Bench
{
  public:
    ApplyAdamOpBench(std::string name, size_t n) : Bench(name), n_(n) {
      // output
      var0_ = new T[n];
      var1_ = new T[n];

      // input & output
      m0_ = new T[n];
      m1_ = new T[n];
      v0_ = new T[n];
      v1_ = new T[n];

      // input
      beta1_power = new T[1];
      beta2_power = new T[1];
      lr = new T[1];
      beta1 = new T[1];
      beta2 = new T[1];
      epsilon = new T[1];
      grad = new T[n];

      for (size_t i = 0; i < n; ++i) {
        m0_[i] = T(drand48());
        v0_[i] = T(drand48());
        m1_[i] = T(drand48());
        v1_[i] = T(drand48());
        grad[i] = T(drand48());
      }

      beta1_power[0] = T(drand48());
      beta2_power[0] = T(drand48());
      lr[0] = T(drand48());
      beta1[0] = T(drand48());
      beta1[0] = T(drand48());
      epsilon[0] = T(drand48());
      grad[0] = T(drand48());

#define C(p) reinterpret_cast<uint64_t>(p);
      args_.dtype = 1; // DT_FLOAT
      args_.use_nesterov_ = false;
      args_.num_elements = n;
      args_.var_ptr = C(var0_);
      args_.m_ptr = C(m0_);
      args_.v_ptr = C(v0_);
      args_.beta1_power_ptr = C(beta1_power);
      args_.beta2_power_ptr = C(beta2_power);
      args_.lr = C(lr);
      args_.beta1_ptr = C(beta1);
      args_.beta2_ptr = C(beta2);
      args_.epsilon_ptr = C(epsilon);
      args_.grad_ptr = C(grad);
#undef C
    }

    int run() override {
      return op_ApplyAdam(reinterpret_cast<void const*>(&args_), sizeof(Args));
    }

    int validate(BenchOpts const& opts) override {
      memset(var0_, 0, sizeof(T) * n_);
      memset(var1_, 0, sizeof(T) * n_);
      memset(m0_, 0, sizeof(T) * n_);
      memset(v0_, 0, sizeof(T) * n_);
      memset(m1_, 0, sizeof(T) * n_);
      memset(v1_, 0, sizeof(T) * n_);
      run();
      ref::apply_adam<float>(false, n_,
                             reinterpret_cast<uint64_t>(var1_),
                             reinterpret_cast<uint64_t>(m1_),
                             reinterpret_cast<uint64_t>(v1_),
                             args_.beta1_power_ptr, args_.beta2_power_ptr,
                             args_.lr,
                             args_.beta1_ptr, args_.beta2_ptr, args_.epsilon_ptr,
                             args_.grad_ptr);

      int flag = 1;
      flag &= check(var0_, var1_, n_, opts);
      flag &= check(m0_, m1_, n_, opts);
      flag &= check(v0_, v1_, n_, opts);
      return flag;
    }

  private:
    struct Args {
      int dtype;
      bool use_nesterov_ ;
      int64_t num_elements ;
      uint64_t var_ptr, m_ptr, v_ptr ;
      uint64_t beta1_power_ptr, beta2_power_ptr ;
      uint64_t lr ;
      uint64_t beta1_ptr, beta2_ptr, epsilon_ptr ;
      uint64_t grad_ptr;
    } args_;

    size_t n_;
    T* var0_;
    T* var1_;
    T* m0_;
    T* m1_;
    T* v0_;
    T* v1_;
    T* beta1_power;
    T* beta2_power;
    T* lr;
    T* beta1;
    T* beta2;
    T* epsilon;
    T* grad;
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

  v.push_back(new BinaryOpBench<float>("Add", _op_Add, ref::add<float>, y, z, n));
  v.push_back(new BinaryOpBench<float>("Sub", _op_Sub, ref::sub<float>, y, z, n));
  v.push_back(new BinaryOpBench<float>("Mul", _op_Mul, ref::mul<float>, y, z, n));
  v.push_back(new BinaryOpBench<float>("Div", op_Div, ref::div<float>, y, z, n));

  v.push_back(new ReductionOpBench<float>("Mean", op_Mean, ref::mean_d2a0<float>, y, n));
  v.push_back(new ReductionOpBench<float>("Sum", op_Sum, ref::sum_d2a0<float>, y, n));

  v.push_back(new UnaryOpBench<float>("Neg", vml::neg, ref::neg, y, n));
  v.push_back(new UnaryOpBench<float>("Rsqrt", vml::rsqrt, ref::rsqrt, y, n));
  v.push_back(new UnaryOpBench<float>("Sqrt", vml::sqrt, ref::sqrt, y, n));
  v.push_back(new UnaryOpBench<float>("Square", vml::square, ref::square, y, n));

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

  BenchOpts opts;
  opts.verbose = 0;
  opts.threshold = 1e-4;
  bool opt_force_bench = false;
  char const* filter = nullptr;
  bool opt_validation_only = false;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-n") == 0) {
      n = strtoul(argv[++i], NULL, 0);
    } else if (strcmp(argv[i], "--nchw") == 0) {
      const char* tmp0 = argv[++i];
      char* tmp1;
      for (int j = 0; j < 4; ++j) {
        nchw[j] = strtoul(tmp0, &tmp1, 0);
        tmp0 = ++tmp1;
      }
      fprintf(stderr, "nchw=%lu,%lu,%lu,%lu\n", nchw[0], nchw[1], nchw[2], nchw[3]);
    } else if (strcmp(argv[i], "--filter") == 0) {
      filter = argv[++i];
    } else if (strcmp(argv[i], "-f") == 0) {
      opt_force_bench = true;
    } else if (strcmp(argv[i], "-r") == 0) {
      repeat = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-v") == 0) {
      ++opts.verbose;
    } else if (strcmp(argv[i], "--validation-only") == 0) {
      opt_validation_only = true;
    } else if (strcmp(argv[i], "--threshold") == 0) {
      opts.threshold = strtod(argv[++i], NULL);
    } else {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  if (opts.verbose > 0) {
    fprintf(stderr, "threshold=%e\n", opts.threshold);
    fprintf(stderr, "n=%lu\n", n);
    fprintf(stderr, "nchw=%lu,%lu,%lu,%lu(%lu)\n",
            nchw[0], nchw[1], nchw[2], nchw[3], nchw[0] * nchw[1] * nchw[2] * nchw[3]);
  }

  std::vector<Bench*> v;

  size_t nchw_elems = 1;
  for (int i = 0; i < 4; ++i)
    nchw_elems *= nchw[i];

  float* y = new float[nchw_elems];

  for (size_t i = 0; i < nchw_elems; ++i)
    y[i] = (float)drand48();

  add_bench<float>(v, n);

  v.push_back(new BiasAddOpBench<float>("BiasAdd(NHWC)", FORMAT_NHWC, 
                                        ref::BiasAdd_NHWC<float>, y, nchw));
  v.push_back(new BiasAddOpBench<float>("BiasAdd(NCHW)", FORMAT_NCHW, 
                                        ref::BiasAdd_NCHW<float>, y, nchw));
  v.push_back(new BiasAddGradOpBench<float>("BiasAddGrad(NHWC)", FORMAT_NHWC, 
                                            ref::BiasAddGrad_NHWC<float>, y, nchw));
  v.push_back(new BiasAddGradOpBench<float>("BiasAddGrad(NCHW)", FORMAT_NCHW, 
                                            ref::BiasAddGrad_NCHW<float>, y, nchw));

  v.push_back(new TileOpBench<float>());

  v.push_back(new TransposeOpBench<float>("Transpose(0231)",
                                          ref::transpose4_0231<float>, y, nchw,
                                          {0, 2, 3, 1}));
  v.push_back(new TransposeOpBench<float>("Transpose(0312)",
                                          ref::transpose4_0312<float>, y, nchw,
                                          {0, 3, 1, 2}));
  v.push_back(new ApplyAdamOpBench<float>("ApplyAdam", n));

  int flag = 1;
  for (Bench* b : v) {
    int tmp = b->validate(opts);
    flag &= tmp;
    if (opts.verbose > 0 || !tmp)
      fprintf(stderr, "Validation: %-20s %s\n", b->name().c_str(), tmp ? "OK" : "NG");
  }

  if (opt_validation_only) {
    fprintf(stderr, "Validation: %s\n", flag ? "OK" : "NG");
    return 1;
  }

  if (!flag && !opt_force_bench)
    return 1;

  for (Bench* b : v) {
    if (!filter || b->name() == filter)
      run_bench(*b, repeat);
  }

  return 0;
}
