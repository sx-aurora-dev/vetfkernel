#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

#include <vml.h>
#include <vml/types.h>
#include "test.h"

extern "C" {
  int op_BiasAdd(const void* args, size_t len);
  int op_BiasAddGrad(const void* args, size_t len);
  int op_Mean(const void* args, size_t len);
  int op_Sum(const void* args, size_t len);
  int op_Transpose(const void* args, size_t len);
  int op_Conv2d(const void* arg, size_t len);
  int conv2d_backprop_input(const void* arg, size_t len);
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
vml::Tensor* makeTensor1D(T const* x, size_t n)
{
  return test::allocTensorDesc<T>(1, {n}, x);
}

#ifdef __ve__
#define USE_VML_RANDOM
#endif

template <typename T>
void randomInit(T* p, size_t n)
{
#ifdef USE_VML_RANDOM
  vml::Tensor* t = makeTensor1D(p, n);
  vml::randomUniform(*t);
#else
  for (size_t i = 0; i < n; ++i)
    p[i] = T(drand48());
#endif
}

template<typename T>
vml::Tensor* createRandomTensor(std::vector<size_t> const& shape)
{
  vml::Tensor* t = test::makeTensor<T>(shape.size(), shape);
  randomInit(t->ptr<T*>(), t->nelems);
  return t;
}

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
  for (size_t i = 0; i < n; ++i) {
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
          fprintf(stderr, "a[%lu] %18.12e b[%lu] %18.12e diff %18.12e err %18.12e\n",
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

struct Bench
{
  Bench(std::string name, int ntimes = -1) 
    : name_(name), data_size_(0), flop_count_(0), ntimes_(ntimes) {}

  std::string name() const { return name_; }
  virtual int validate(BenchOpts const&) = 0;
  virtual int run() = 0;
  
  std::string name_;

  size_t data_size_; // bytes
  size_t flop_count_;
  int ntimes_;
};

void run_bench(Bench& bench, int ntimes = 1, bool detail = false)
{
  int ntimes0 = bench.ntimes_;
  if (ntimes0 < 0)
    ntimes0 = ntimes;

  //fprintf(stderr, "ntimes=%d (%d)\n", ntimes0, bench.ntimes_);

  // warmup
  int ret = bench.run();
  if (ret != 0)
      fprintf(stderr, "ret=%d\n", ret);

  double t0 = second();
  for (int i = 0; i < ntimes0; ++i) {
    int ret = bench.run();
    if (ret != 0)
      fprintf(stderr, "ret=%d\n", ret);
  }
  double t1 = second();
  double sec = (t1 - t0) / ntimes0;
  double flops = bench.flop_count_ / sec;
  double bw = bench.data_size_ / sec;

  if (detail) {
    fprintf(stdout, "%-80s %8.3lf ms %8.3lf GFlops %8.3lf GB/s %8.3lf sec\n",
            bench.name_.c_str(), sec*1e3, flops/1e9, bw/1e9, (t1 - t0));
  } else {
    fprintf(stdout, "%-80s %8.3lf ms\n", bench.name_.c_str(), sec*1e3);
  }
}

namespace ref
{

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
// BiasAddGrad
//

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

template<typename T>
int tile_dim5_11(vml::Tensor const& X, vml::Tensor const& Y)
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


template <typename T, int N>
vml::TensorDesc<N> createStaticTensor(T const* p, std::vector<int64_t> const& dims)
{
  vml::TensorDesc<N> t;
  t.dtype = test::dtype_s<T>::type;
  t.addr = reinterpret_cast<uint64_t>(p);
  t.dims = dims.size();
  t.nelems = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    t.dim_size[i] = dims[i];
    t.nelems *= dims[i];
  }
  return t;
}

template <typename T>
class UnaryOpBench : public Bench
{
  public:
    UnaryOpBench(std::string name, 
                 int (*op)(vml::Tensor const& out, vml::Tensor const& in),
                 T const* y, int64_t n, int ntimes = -1) 
      : Bench(name, ntimes), op_(op) {
        this->data_size_ = sizeof(T) * n * 2;

        T* x = new T[n];
        in_ = createStaticTensor<T, 1>(y, {n});
        out_ = createStaticTensor<T, 1>(x, {n});
      }

    int validate(BenchOpts const& opts) override {
      return 1;
    }

    int run() {
      return op_(out_, in_);
    }

  private:
    vml::TensorDesc<1> in_;
    vml::TensorDesc<1> out_;

    int (*op_)(vml::Tensor const& out, vml::Tensor const& in);
};

template <typename T>
class BinaryOpBench : public Bench 
{
  public:
    BinaryOpBench(std::string name, 
                  int (*op)(vml::Tensor const& out,
                            vml::Tensor const& in0,
                            vml::Tensor const& in1),
                  T const* y, T const* z, int64_t n, int ntimes = -1) 
      : Bench(name, ntimes), op_(op) {
        T* x = new T[n];
        X_ = createStaticTensor<T, 1>(x, {n});
        Y_ = createStaticTensor<T, 1>(y, {n});
        Z_ = createStaticTensor<T, 1>(z, {n});

        data_size_ = n * 3 * sizeof(T);
        flop_count_ = n;
      }

    int validate(BenchOpts const& opts) override {
      return 1;
    }

    int run() override {
      return op_(X_, Y_, Z_);
    }

  private:
    int (*op_)(vml::Tensor const&, vml::Tensor const&, vml::Tensor const&);
    vml::TensorDesc<1> X_;
    vml::TensorDesc<1> Y_;
    vml::TensorDesc<1> Z_;
};

template <typename T>
class ReductionOpBench : public Bench
{
  public:
    ReductionOpBench(std::string name, 
                     int (*op)(const void* args, size_t len),
                     int (*ref_op)(uint64_t, uint64_t, size_t, size_t),
                     T const* y, size_t n, int ntimes) 
      : Bench(name, ntimes), op_(op), ref_op_(ref_op), y_(y), n_(n) { 
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
    BiasAddOpBench(std::string name,
                   int data_format,
                   T const* in,
                   std::vector<size_t> shape,
                   int ntimes)
      : Bench(name, ntimes), data_format_(data_format) {

        out_ = test::makeTensor<float>(4, shape);
        in_ = test::allocTensorDesc(4, shape, in);
        int channel = shape[3];
        if (data_format == FORMAT_NCHW)
          channel = shape[1];
        bias_ = createRandomTensor<float>({channel}); // channel

#if 0
        std::cerr << *in_ << std::endl;
        std::cerr << *bias_ << std::endl;
#endif
    }

    int validate(BenchOpts const& opts) override {
      return 1;
    }

    int run() override {
      return vml::biasAdd(*out_, *in_, *bias_, data_format_);
    }

  private:
    vml::Tensor* out_;
    vml::Tensor* in_;
    vml::Tensor* bias_;
    int data_format_;
};

template <typename T>
class BiasAddGradOpBench : public Bench
{
  public:
    BiasAddGradOpBench(std::string name, int data_format,
                       int (*ref_op)(uint64_t output, uint64_t output_backprop,
                                     int batch, int width, int height, int channel),
                       T const* in,
                       std::vector<size_t> const& nchw,
                       int ntimes = -1)
      : Bench(name, ntimes), ref_op_(ref_op), in_(in), nchw_(nchw) {
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

    std::vector<size_t> nchw_;
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
    TileOpBench(int ntimes) : Bench("Tile", ntimes) 
    {
      std::vector<size_t> dimsIn({8, 16, 16, 1, 1});
      std::vector<size_t> dimsOut({8, 16, 16, 32, 32});

      in_ = createRandomTensor<float>(dimsIn);
      out0_ = test::makeTensor<float>(5, dimsOut);
      out1_ = test::makeTensor<float>(5, dimsOut);
    }

    int validate(BenchOpts const& opts) override {
      memset(out0_->ptr<float*>(), 0, sizeof(T) * out0_->nelems);
      memset(out1_->ptr<float*>(), 0, sizeof(T) * out1_->nelems);
      run();
      ref::tile_dim5_11<float>(*out1_, *in_);
      return check_exact(out0_->ptr<float*>(), out1_->ptr<float*>(),
                         out1_->nelems, opts);
    }

    int run() override {
      return vml::tile(*out0_, *in_);
    }

  private:
    vml::Tensor* in_;
    vml::Tensor* out0_;
    vml::Tensor* out1_;
};

template <typename T>
class TransposeOpBench : public Bench
{
  public:
    TransposeOpBench(std::string name,
                     int (*ref_op)(uint64_t, uint64_t, const int32_t*),
                     T const* y,
                     std::vector<size_t> const& dims,
                     std::vector<int> perm,
                     int ntimes = -1)
      : Bench(name, ntimes), ref_op_(ref_op), y_(y) {
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
        args_.conjugate = 0;
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
      int conjugate;
      int32_t dim_size[8]; // in
      int32_t perm[8];
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
    ApplyAdamOpBench(std::string name, size_t n, int ntimes) : Bench(name, ntimes), n_(n) {
      // output
      var0_ = new T[n];
      var1_ = new T[n];

      // input & output
      m0_ = new T[n];
      m1_ = new T[n];
      v0_ = new T[n];
      v1_ = new T[n];

      // input
      beta1_power_ = new T[1];
      beta2_power_ = new T[1];
      lr_ = new T[1];
      beta1_ = new T[1];
      beta2_ = new T[1];
      epsilon_ = new T[1];
      grad_ = new T[n];

      randomInit(m0_, n);
      randomInit(v0_, n);
      randomInit(m1_, n);
      randomInit(v1_, n);
      randomInit(grad_, n);

      beta1_power_[0] = T(drand48());
      beta2_power_[0] = T(drand48());
      lr_[0] = T(drand48());
      beta1_[0] = T(drand48());
      beta1_[0] = T(drand48());
      epsilon_[0] = T(drand48());
      grad_[0] = T(drand48());
    }

    int run() override {
      vml::Tensor* var  = makeTensor1D<T>(var0_, n_) ;
      vml::Tensor* m    = makeTensor1D<T>(m0_, n_) ;
      vml::Tensor* v    = makeTensor1D<T>(v0_, n_) ;
      vml::Tensor* grad = makeTensor1D<T>(grad_, n_) ;

      vml::Tensor* beta1_power = makeTensor1D<T>(beta1_power_, 1) ;
      vml::Tensor* beta2_power = makeTensor1D<T>(beta2_power_, 1) ;
      vml::Tensor* lr          = makeTensor1D<T>(lr_, 1) ;
      vml::Tensor* beta1       = makeTensor1D<T>(beta1_, 1) ;
      vml::Tensor* beta2       = makeTensor1D<T>(beta2_, 1) ;
      vml::Tensor* epsilon     = makeTensor1D<T>(epsilon_, 1) ;

      return vml::applyAdam(
  	*var, *m, *v,
  	*beta1_power, *beta2_power,
  	*lr, *beta1, *beta2, *epsilon,
	*grad,
  	use_nesterov_ ) ;
    }

    int validate(BenchOpts const& opts) override {
      memset(var0_, 0, sizeof(T) * n_);
      memset(var1_, 0, sizeof(T) * n_);
      memset(m0_, 0, sizeof(T) * n_);
      memset(v0_, 0, sizeof(T) * n_);
      memset(m1_, 0, sizeof(T) * n_);
      memset(v1_, 0, sizeof(T) * n_);
      run();
      ref::apply_adam<float>(use_nesterov_, n_,
                             reinterpret_cast<uint64_t>(var1_),
                             reinterpret_cast<uint64_t>(m1_),
                             reinterpret_cast<uint64_t>(v1_),
			     reinterpret_cast<uint64_t>(beta1_power_),
			     reinterpret_cast<uint64_t>(beta2_power_),
			     reinterpret_cast<uint64_t>(lr_),
			     reinterpret_cast<uint64_t>(beta1_),
			     reinterpret_cast<uint64_t>(beta2_),
			     reinterpret_cast<uint64_t>(epsilon_),
			     reinterpret_cast<uint64_t>(grad_)) ;

      int flag = 1;
      flag &= check(var0_, var1_, n_, opts);
      flag &= check(m0_, m1_, n_, opts);
      flag &= check(v0_, v1_, n_, opts);
      return flag;
    }

  private:
    size_t n_;
    T* var0_;
    T* var1_;
    T* m0_;
    T* m1_;
    T* v0_;
    T* v1_;
    T* beta1_power_;
    T* beta2_power_;
    T* lr_;
    T* beta1_;
    T* beta2_;
    T* epsilon_;
    T* grad_;
    const bool use_nesterov_ = false ;
};

#ifdef USE_VEDNN
class Conv2DBench : public Bench
{
  public:
    Conv2DBench(//F func,
                std::string name,
                vml::Tensor const& out_bp,
                vml::Tensor const& filter,
                vml::Tensor& in_bp,
                std::vector<int> param, // stride[2], dilation[2], padding[2]
                int data_format,
                int data_type) : Bench(name) {

      _in     = in_bp;
      _filter = filter;
      _out    = out_bp;
  
      _params.push_back(param[0]);
      _params.push_back(param[1]);
      _params.push_back(param[2]);
      _params.push_back(param[3]);
      _params.push_back(param[4]);
      _params.push_back(param[5]);
      _params.push_back(data_format);

    }
  
    int validate(BenchOpts const&) override { return 1; }

    int run() override {
      vml::conv2d(_in, _filter, _out, _params);
      return 0;
    }

  private:
    struct TensorParam {
      int w,h,c,n ;
    } ;

    TensorParam nchw(vml::Tensor const& t) {
      int64_t const* d = t.dim_size;
      TensorParam p = {(int)d[2], (int)d[3], (int)d[1], (int)d[0]};
      return p;
    }

  vml::Tensor _in;
  vml::Tensor _filter;
  vml::Tensor _out;
  std::vector<int> _params;
  
  //F func_;
};

template<typename F>
class Conv2DBackPropBench : public Bench
{
  public:
    Conv2DBackPropBench(F func,
                std::string name,
                vml::Tensor const& out_bp,
                vml::Tensor const& filter,
                vml::Tensor& in_bp,
                std::vector<int> param, // stride[2], dilation[2], padding[2]
                int data_format,
                int data_type) : Bench(name), func_(func) {
      assert(data_format == FORMAT_NCHW);
      args_.out_bp = out_bp.addr;
      args_.filter = filter.addr;
      args_.in_bp = in_bp.addr;
      args_.out_bp_param = nchw(out_bp);
      args_.filter_param = nchw(filter);
      args_.in_bp_param = nchw(in_bp);
      args_.row_stride = param[0];
      args_.col_stride = param[1];
      args_.row_dilation = param[2];
      args_.col_dilation = param[3];
      args_.row_padding = param[4];
      args_.col_padding = param[5];
      args_.data_format = data_format;
      args_.data_type = data_type;
    }

    int validate(BenchOpts const&) override { return 1; }

    int run() override {
      func_(&args_, sizeof(args_));
      return 0;
    }

  private:
    struct TensorParam {
      int w,h,c,n ;
    } ;

    struct ConvParam {
      uint64_t out_bp;
      uint64_t filter;
      uint64_t in_bp;
      TensorParam out_bp_param;
      TensorParam filter_param;
      TensorParam in_bp_param;

      int row_stride;
      int col_stride;
      int row_dilation;
      int col_dilation;
      int row_padding;
      int col_padding;

      int data_format;
      int data_type;
    };

    TensorParam nchw(vml::Tensor const& t) {
      int64_t const* d = t.dim_size;
      TensorParam p = {(int)d[2], (int)d[3], (int)d[1], (int)d[0]};
      return p;
    }

    ConvParam args_;
    F func_;
};

Bench* make_conv2d_bench(
                         std::string name,
                         std::vector<size_t> const& in_shape,
                         std::vector<size_t> const& filter_shape,
                         std::vector<size_t> const& out_shape,
                         std::vector<int> param,
                         int data_format = FORMAT_NCHW,
                         int data_type = DT_FLOAT)
{
  vml::Tensor* in = createRandomTensor<float>(in_shape);
  vml::Tensor* filter = createRandomTensor<float>(filter_shape);
  vml::Tensor* out = createRandomTensor<float>(out_shape);

  std::stringstream buf;

#define S(t) (t).dim_size[0] << "x" << (t).dim_size[1] << "x" << (t).dim_size[2] << "x" << (t).dim_size[3]
  buf << name << "-" << S(*in) << "-" << S(*filter) << "-" << S(*out)
    << "-" << param[0] << "x" << param[1]
    << "-" << param[2] << "x" << param[3]
    << "-" << param[4] << "x" << param[5]
    << "-" << data_format << "-" << data_type;

  return new Conv2DBench(buf.str(), *in, *filter, *out, param, data_format, data_type);
}

template <typename F>
Bench* make_conv2d_backprop_bench(F func,
                         std::string name,
                         std::vector<size_t> const& in_shape,
                         std::vector<size_t> const& filter_shape,
                         std::vector<size_t> const& out_shape,
                         std::vector<int> param,
                         int data_format = FORMAT_NCHW,
                         int data_type = DT_FLOAT)
{
  vml::Tensor* in = createRandomTensor<float>(in_shape);
  vml::Tensor* filter = createRandomTensor<float>(filter_shape);
  vml::Tensor* out = createRandomTensor<float>(out_shape);

  std::stringstream buf;

#define S(t) (t).dim_size[0] << "x" << (t).dim_size[1] << "x" << (t).dim_size[2] << "x" << (t).dim_size[3]
  buf << name << "-" << S(*in) << "-" << S(*filter) << "-" << S(*out)
    << "-" << param[0] << "x" << param[1]
    << "-" << param[2] << "x" << param[3]
    << "-" << param[4] << "x" << param[5]
    << "-" << data_format << "-" << data_type;

  return new Conv2DBackPropBench<F>(func, buf.str(), *in, *filter, *out, param, data_format, data_type);
}

void add_conv2d_bench(std::vector<Bench*>& v)
{

  // in, filter, out, {stride[2], dilation[2], stride[2]}
  // conv2d
#define F(...) v.push_back(make_conv2d_bench("Conv2D", __VA_ARGS__))
  F({32, 256, 34, 34}, {512, 256, 4, 4}, {32, 512, 31, 31}, {1, 1, 1, 1, 0, 0});
  F({32, 1024, 14, 14}, {2048, 1024, 1, 1}, {32, 2048, 7, 7}, {2, 2, 1, 1, 0, 0}); // ResNet50
#undef F
}

void add_conv2d_backprop_bench(std::vector<Bench*>& v)
{
  // in, filter, out, {stride[2], dilation[2], stride[2]}
  
  // conv2d_backprop_input
#define F(...) v.push_back(make_conv2d_backprop_bench(conv2d_backprop_input, "Conv2DBackpropInput", __VA_ARGS__))
  F({32, 1024, 16, 16}, {1024, 256, 4, 4}, {32, 256, 32, 32}, {2, 2, 1, 1, 1, 1});
#undef F
}

#endif // USE_VEDNN

template <typename T>
void add_bench(std::vector<Bench*>& v, size_t n)
{
  T* y = new T[n];
  T* z = new T[n];

  randomInit(y, n);
  randomInit(z, n);

  v.push_back(new BinaryOpBench<float>("Add", vml::add, y, z, n, 100));
  v.push_back(new BinaryOpBench<float>("Sub", vml::sub, y, z, n, 100));
  v.push_back(new BinaryOpBench<float>("Mul", vml::mul, y, z, n, 100));
  v.push_back(new BinaryOpBench<float>("Div", vml::div, y, z, n, 100));

  v.push_back(new ReductionOpBench<float>("Mean", op_Mean, ref::mean_d2a0<float>, y, n, 100));
  v.push_back(new ReductionOpBench<float>("Sum", op_Sum, ref::sum_d2a0<float>, y, n, 100));

  v.push_back(new UnaryOpBench<float>("Neg", vml::neg, y, n, 500));
  v.push_back(new UnaryOpBench<float>("Rsqrt", vml::rsqrt, y, n, 500));
  v.push_back(new UnaryOpBench<float>("Sqrt", vml::sqrt, y, n, 200));
  v.push_back(new UnaryOpBench<float>("Square", vml::square, y, n, 500));

#if 0
  delete[] y;
  delete[] z;
#endif
}

int main(int argc, char* argv[])
{
  size_t n = 20000000;
  std::vector<size_t> nchw{256, 16, 64, 64};
  int repeat = 10;

  BenchOpts opts;
  opts.verbose = 0;
  opts.threshold = 1e-4;
  bool opt_force_bench = false;
  char const* filter = nullptr;
  bool opt_validation_only = false;
  bool opt_detail = false;

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
    } else if (strcmp(argv[i], "-d") == 0) {
      opt_detail = true;
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

  //vml::initialize();

  std::vector<Bench*> v;

  size_t nelems = 1;
  for (int i = 0; i < 4; ++i)
    nelems *= nchw[i];

  float* y = new float[nelems];
  randomInit(y, nelems);

  add_bench<float>(v, n);

#define PUSH(B) v.push_back(new B);
  PUSH(BiasAddOpBench<float>("BiasAdd(NHWC)", FORMAT_NHWC, y,
                             {nchw[0], nchw[2], nchw[3], nchw[1]}, 200));
  PUSH(BiasAddOpBench<float>("BiasAdd(NCHW)", FORMAT_NCHW, y, nchw, 500));

  PUSH(BiasAddGradOpBench<float>("BiasAddGrad(NHWC)", FORMAT_NHWC, 
                                 ref::BiasAddGrad_NHWC<float>, y, nchw, 30));
  PUSH(BiasAddGradOpBench<float>("BiasAddGrad(NCHW)", FORMAT_NCHW, 
                                 ref::BiasAddGrad_NCHW<float>, y, nchw, 300));

  PUSH(TileOpBench<float>(200));

  PUSH(TransposeOpBench<float>("Transpose(0231)",
                               ref::transpose4_0231<float>, y, nchw,
                               {0, 2, 3, 1}, 500));
  PUSH(TransposeOpBench<float>("Transpose(0312)",
                               ref::transpose4_0312<float>, y, nchw,
                               {0, 3, 1, 2}, 200));
  PUSH(ApplyAdamOpBench<float>("ApplyAdam", n, 200));

#ifdef USE_VEDNN
  if (!opt_validation_only)
    add_conv2d_bench(v);
#endif

  if (opts.verbose > 0)
    fprintf(stderr, "Initialization done\n");

  std::vector<Bench*> v2;

  if (filter) {
    for (Bench* b : v) {
      if (b->name() == filter)
        v2.push_back(b);
    }
  } else
    v2 = v;

  int flag = 1;
  for (Bench* b : v2) {
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

  for (Bench* b : v2)
    run_bench(*b, repeat, opt_detail);

  return 0;
}
