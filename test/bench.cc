#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

extern "C" {
  int op_Mul(const void* args, size_t len);
  int op_Add(const void* args, size_t len);
  int op_Sum(const void* args, size_t len);
}

static double second()
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

template <typename T>
int check(T const* a, T const* b, size_t n, int verbose = 0)
{
  int flag = 0;
  for (int j = 0; j < n; ++j) {
    if (a[j] != b[j]) {
      flag = 1;
      if (verbose > 0) {
        fprintf(stderr, "a %18.12e b %18.12e diff %18.12e\n", 
                a[j], b[j], std::abs(a[j] - b[j]));
      }
    }
  }
}

struct Bench
{
  Bench(std::string name) 
    : name_(name), data_size_(0), flop_count_(0) {}

  std::string name() const { return name_; }
  virtual int validate() = 0;
  virtual int run() = 0;
  
  std::string name_;

  size_t data_size_; // bytes
  size_t flop_count_;
};

void run_bench(Bench& bench, int ntimes = 1)
{
  double t0 = second();
  for (int i = 0; i < ntimes; ++i) {
    //int ret = bench.op_(args, len);
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

namespace binary 
{

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
  t.dtype = 1; // FIXME
  t.addr = reinterpret_cast<uint64_t>(x);
  t.dims = 1;
  t.nelems = n;
  t.dim_size[0] = n;
  return t;
}

template <typename T>
class BinaryOpBench : public Bench 
{
  public:
    BinaryOpBench(std::string name, 
                  int (*op)(const void* args, size_t len),
                  void (*ref_op)(T*, T const*, T const*, size_t),
                  T* x0, T* x1, T const* y, T const* z, size_t n) 
      : Bench(name), op_(op), ref_op_(ref_op), x0_(x0), x1_(x1), y_(y), z_(z), n_(n) { 
        args_.in0 = mktensor(x0, n);
        args_.in1 = mktensor(y, n);
        args_.out = mktensor(z, n);

        data_size_ = n * 3 * sizeof(T);
        flop_count_ = n;
      }

    int validate() {
      memset(x0_, 0, sizeof(T) * n_);
      memset(x1_, 0, sizeof(T) * n_);
      int ret = run();
      if (ret != 0)
        fprintf(stderr, "ret=%d\n", ret);
      ref_op_(x1_, y_, z_, n_);
      return check(x0_, x1_, n_);
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
    BinaryOpArgs args_;
    int (*op_)(const void* args, size_t len);
    void (*ref_op_)(T*, T const*, T const*, size_t);
};

template <typename T>
void mul(T* x, T const* y, T const* z, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    x[i] = y[i] * z[i];
}

} // namespace binary

template <typename T>
void sum_2a0(T* x, T const* y, size_t m, size_t n)
{
#pragma _NEC novector
  for (size_t i = 0; i < m; ++i) {
    T s = T(0);
#pragma _NEC novector
    for (size_t j = 0; j < n; ++j)
      s += y[i * n + j];
    x[i] = s;
  }
}

template <typename T>
class ReductionOpBench : public Bench
{
  public:
    ReductionOpBench(std::string name, 
                     int (*op)(const void* args, size_t len),
                     void (*ref_op)(T*, T const*, size_t, size_t),
                     T* x0, T* x1, T const* y, size_t n) 
      : Bench(name), op_(op), ref_op_(ref_op), x0_(x0), x1_(x1), y_(y), n_(n) { 
        data_size_ = n * 2 * sizeof(T);
        flop_count_ = n;

        args_.dtype = 1; // FIXME
        args_.ndims = 2;
        args_.in = reinterpret_cast<uint64_t>(y);
        args_.out = reinterpret_cast<uint64_t>(x0);
        args_.dim_size[0] = 1;
        args_.dim_size[1] = n;
        args_.axis = 0;
      }

    int run() override {
      return op_(reinterpret_cast<void const*>(&args_), sizeof(Args));
    }

    int validate() override {
      x0_[0] = 0;
      x1_[0] = 0;
      run();
      ref_op_(x1_, y_, n_, 1);
      return check(x0_, x1_, 1);
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
    void (*ref_op_)(T*, T const*, size_t, size_t);

    T* x0_;
    T* x1_;
    T const* y_;
    size_t n_;

};

template <typename T>
void do_bench(std::vector<Bench*>& v, size_t n)
{
  T* x0 = new T[n];
  T* x1 = new T[n];
  T* y = new T[n];
  T* z = new T[n];

  for (int i = 0; i < n; ++i) {
    y[i] = drand48();
    z[i] = drand48();
  }

  v.push_back(new binary::BinaryOpBench<float>("Mul", op_Mul, binary::mul<float>, x0, x1, y, z, n));
  v.push_back(new ReductionOpBench<float>("Sum", op_Sum, sum_2a0<float>, x0, x1, y, n));

#if 0
  delete[] x0;
  delete[] y;
  delete[] z;
#endif
}

int main(int argc, char* argv[])
{
  size_t n = 20000000;
  int repeat = 10;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-n") == 0) {
      n = strtoul(argv[++i], NULL, 0);
    } else if (strcmp(argv[i], "-r") == 0) {
      repeat = atoi(argv[++i]);
    }
  }

  fprintf(stderr, "n=%lu\n", n);

  std::vector<Bench*> v;

  do_bench<float>(v, n);

  for (Bench* b : v) {
    int flag = b->validate();
    fprintf(stderr, "Validation: %-20s %s\n", b->name(), flag ? "OK" : "NG");
  }

  for (Bench* b : v) {
    run_bench(*b, repeat);
  }

  return 0;
}
