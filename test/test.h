#ifndef TEST_H
#define TEST_H

#include <string>
#include <vector>
#include <vml.h>

namespace test {


struct TestParam
{
    int verbose;
};

struct Test
{
    std::string name;
    bool (*func)(TestParam const&);
};

void register_test(std::string const& name,
                   bool (*func)(TestParam const&));

class RegisterTest
{
  public:
    RegisterTest(std::string name, bool (*func)(TestParam const&)) {
      register_test(name, func);
    }
};

#define REGISTER_TEST_HELPER_UNIQ(cnt, name, func) \
  static RegisterTest __RegisterTest_##cnt(name, func)


#define REGISTER_TEST_HELPER(cnt, name, func) \
  REGISTER_TEST_HELPER_UNIQ(cnt, name, func)

#define REGISTER_TEST(name, func) \
  REGISTER_TEST_HELPER(__COUNTER__, name, func)

template<typename T> struct dypte_s {};
template<> struct dypte_s<float> { static const int type = 1; };
template<> struct dypte_s<bool>  { static const int type = 10; };

template <typename T>
vml::Tensor makeTensor(size_t dims, std::vector<size_t> const& dim_size)
{
    vml::Tensor t;

    t.dtype = dypte_s<T>::type;
    t.dims = dims;
    t.nelems = 1;
    for (int i = 0; i < dims; ++i) {
        t.dim_size[i] = dim_size[i];
        t.nelems *= dim_size[i];
    }

    t.addr = reinterpret_cast<uint64_t>(new T[t.nelems]);

    return t;
}

template<typename T>
class Tensor {
    public:
        Tensor(std::vector<size_t> const& shape) {
          shape_ = shape;
          t = makeTensor<T>(shape.size(), shape);
          stride_.resize(shape.size());
          size_t dim = t.dims;
          stride_[dim - 1] = 1;
          for (int i = dim - 2; i >= 0; --i) {
            stride_[i] = stride_[i + 1] * t.dim_size[i + 1];
          }
        }
        ~Tensor() { delete[] reinterpret_cast<T*>(t.addr); }
        std::vector<size_t> const& shape() const { return shape_; }
        T* data() { return reinterpret_cast<T*>(t.addr); }
        T const* data() const { return reinterpret_cast<T const*>(t.addr); }
        size_t nelems() const { return t.nelems; }
        size_t dims() const { return t.dims; }
        size_t dim_size(size_t i) const { return t.dim_size[i]; }
        size_t stride(size_t i) const { return stride_[i]; }

        vml::Tensor tensor() const { return t; }

        operator vml::Tensor() const { return t; }

    private:
        vml::Tensor t;
        std::vector<size_t> stride_;
        std::vector<size_t> shape_;
};

template<typename T>
bool checkTensor(Tensor<T> const& a, Tensor<T> const& b)
{
    if (a.nelems() != b.nelems())
        return false;

    for (size_t i = 0; i < a.nelems(); ++i) {
#if 0
        if (a.data()[i] != b.data()[i])
            return false;
#else
        T ai = a.data()[i];
        T bi = b.data()[i];
        double err = ai - bi;
        if (err * err / (ai * bi) > 1e-8)
          return false;
#endif
    }
    return true;
}

template<typename T>
void printTensor(Tensor<T> const& t)
{
    std::string fmt = " %8.3f";
    if (typeid(T) == typeid(bool)) {
      fmt = std::string(" %d");
    }

    std::vector<size_t> s(t.dims() + 1);
    s[t.dims()] = 1;
    for (int i = t.dims() - 1; i >= 0; --i)
        s[i] = s[i + 1] * t.dim_size(i);

#if 0
    fprintf(stderr, "%d %d %d\n", t.dim_size(0), t.dim_size(1), t.dim_size(2));
    fprintf(stderr, "%d %d %d\n", s[0], s[1], s[2]);
#endif

    T const* p = t.data();
    size_t n = t.dim_size(t.dims() - 1); // innermost

    for (size_t i = 0; i < t.nelems(); ++i) {
        if (i % n == 0) {
            for (int j = 0; j < t.dims(); ++j) {
                fprintf(stderr, "%c", i % s[j] == 0 ? '[' : ' ');
            }
        }
        fprintf(stderr, fmt.c_str(), p[i]);
        if ((i + 1) % n == 0) {
            fprintf(stderr, " ");
            for (int j = 0; j < t.dims(); ++j) {
                if ((i + 1) % s[j] == 0) 
                    fprintf(stderr, "]");
            }
            fprintf(stderr, "\n");
        }
    }
}

} // namespace test


#endif // TEST_H
