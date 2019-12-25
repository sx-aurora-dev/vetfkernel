#include <vml.h>
#include <vml/log.h>
#include <vml/types.h>

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/intrinsic.h"
#endif

//
// Mean
//

namespace {
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

#ifdef LIBVETF_INTRINSIC
template <>
int mean_d2a0<float>(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    return mean_d2a0_f32(out, in, dim0, dim1);
}
#endif

template <typename T>
int mean_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t i = 0; i < dim0; ++i) {
        T s = T(0);
        for (size_t j = 0; j < dim1; ++j) {
            s += pi[i * dim1 + j];
        }
        po[i] = s / dim1;
    }

    return 0;
}

template <typename T>
int mean_d3a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t i = 0; i < dim0; ++i) {
        for (size_t k = 0; k < dim2; ++k) {
            T s = T(0);
            for (size_t j = 0; j < dim1; ++j) {
                s += pi[i * dim12 + j * dim2 + k];
            }
            po[i * dim2 + k] = s / dim1 ;
        }
    }

    return 0;
}




template <typename T>
int mean_d3a02(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t j = 0; j < dim1; ++j) {
        T s = T(0);
        for (size_t i = 0; i < dim0; ++i) {
            for (size_t k = 0; k < dim2; ++k) {
                s += pi[i * dim12 + j * dim2 + k];
            }
        }
        po[j] = s / (dim0*dim2);
    }

    return 0;
}

#ifdef LIBVETF_INTRINSIC
template <>
int mean_d3a02<float>(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    return mean_d3a02_f32(out, in, dim0, dim1, dim2);
}
#endif

} // namespace

namespace vml {
int mean(vml::Tensor const& out, vml::Tensor const& in, std::vector<int> const& axis)
{
    int ret = 0;
    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << in.dtype << " ndims=" << in.dims << " axis.size()=" << axis.size();

    if (in.dtype == DT_FLOAT) {
        if (in.dims == 2 && axis[0] == 1) {
            ret = mean_d2a1<float>(out.addr, in.addr, in.dim_size[0], in.dim_size[1]);
        }
        if (in.dims == 2 && axis[0] == 0) {
            ret = mean_d2a0<float>(out.addr, in.addr, in.dim_size[0], in.dim_size[1]);
        }
        if (in.dims == 3 && axis[0] == 1) {
            ret = mean_d3a1<float>(out.addr, in.addr, in.dim_size[0], in.dim_size[1], in.dim_size[2]);
        }
        if (in.dims == 3 && axis[0] == 0) {
            ret = mean_d3a02<float>(out.addr, in.addr, in.dim_size[0], in.dim_size[1], in.dim_size[2]);
        }
    }

    LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
    return ret;
}
} // namespace vml

