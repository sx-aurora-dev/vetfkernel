#include <cstdio>
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
  t.dtype = 1;
  t.addr = reinterpret_cast<uint64_t>(x);
  t.dims = 1;
  t.nelems = n;
  t.dim_size[0] = n;
  return t;
}

BinaryOpArgs init(float* x, float const* y, float const* z, int n)
{
  BinaryOpArgs args;
  args.in0 = mktensor(x, n);
  args.in1 = mktensor(y, n);
  args.out = mktensor(z, n);
  return args;
}

template <typename T>
void mul(T* x, T const* y, T const* z, int n)
{
  for (int i = 0; i < n; ++i)
    x[i] = y[i] * z[i];
}

template <typename T>
void check(T const* a, T const* b, size_t n)
{
  for (int j = 0; j < n; ++j) {
    if (a[j] != b[j])
      fprintf(stderr, "a %18.12e b %18.12e diff %18.12e\n", 
              a[j], b[j], std::abs(a[j] - b[j]));
  }
}

void run_bench(std::string name, int (*op)(const void* args, size_t len),
        const void* args, size_t len, int repeat, size_t n)
{
  double t0 = second();
  for (int i = 0; i < repeat; ++i) {
    int ret = op(args, len);
    if (ret != 0)
      fprintf(stderr, "ret=%d\n", ret);
  }
  double t1 = second();
  double sec = (t1 - t0) / repeat;
  double bw = n * sizeof(float) * 3 / sec;

  fprintf(stderr, "%-20s %8.3lf ms %8.3lf GB/s\n",
          name.c_str(), sec*1e3, bw/1e9);
}

template <typename T>
void bench_mul(T* x0, T* x1, T const* y, T const* z, size_t n, int repeat)
{
  BinaryOpArgs args = init(x0, y, z, n);

  memset(x0, 0, sizeof(float) * n);
  memset(x1, 0, sizeof(float) * n);

  int ret = op_Mul(&args, sizeof(args));
  if (ret != 0)
    fprintf(stderr, "ret=%d\n", ret);

  mul(x1, y, z, n);

  check(x0, x1, n);


  run_bench("Mul", op_Mul, &args, sizeof(args), repeat, n);
}

template <typename T>
void sum_2a0(T* x, T const* y, int n, int m)
{
#pragma _NEC novector
  for (int i = 0; i < m; ++i) {
    T s = T(0);
#pragma _NEC novector
    for (int j = 0; j < n; ++j)
      s += y[i * n + j];
    x[i] = s;
  }
}

template <typename T>
void bench_sum(T* x0, T* x1, T const* y, size_t n, int repeat)
{
  struct Args {
    int dtype;
    int ndims;
    uint64_t in;
    uint64_t out;
    int64_t dim_size[3];
    int axis;
  } args;

  args.dtype = 1; // FIXME
  args.ndims = 2;
  args.in = reinterpret_cast<uint64_t>(y);
  args.out = reinterpret_cast<uint64_t>(x0);
  args.dim_size[0] = 1;
  args.dim_size[1] = n;
  args.axis = 0;

  x0[0] = 0;
  x1[0] = 0;

  op_Sum(&args, sizeof(args));

  sum_2a0(x1, y, 1, n);
  check(x0, x1, 1);

  run_bench("Sum", op_Sum, &args, sizeof(args), repeat, n);
}

template <typename T>
void do_bench(int n, int repeat)
{
  T* x0 = new T[n];
  T* x1 = new T[n];
  T* y = new T[n];
  T* z = new T[n];

  for (int i = 0; i < n; ++i) {
    y[i] = drand48();
    z[i] = drand48();
  }

  bench_mul(x0, x1, y, z, n, repeat);
  bench_sum(x0, x1, y, n, repeat);

  delete[] x0;
  delete[] y;
  delete[] z;
}

int main(int argc, char* argv[])
{
  size_t n = 2*1024*1024;
  int repeat = 1;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-n") == 0) {
      n = strtoul(argv[++i], NULL, 0);
    } else if (strcmp(argv[i], "-r") == 0) {
      repeat = atoi(argv[++i]);
    }
  }

  fprintf(stderr, "n=%lu\n", n);

  do_bench<float>(n, repeat);

  return 0;
}
