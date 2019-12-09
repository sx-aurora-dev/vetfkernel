#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "vml/types.h"
#include "vml/log.h"

#include <omp.h>

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/intrinsic.h"
#endif

REGISTER_KERNEL("Max", "op_Max");
REGISTER_KERNEL("Sum", "op_Sum");
REGISTER_KERNEL("Prod", "op_Prod");
REGISTER_KERNEL("Mean", "op_Mean");

#define CHECK_ARG_LEN(l0, l1) \
    if ((l0) != (l1)) { \
    LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << (l1) << " expected but " << (l0); \
    return 1; \
    }

extern "C" {
int op_Max(const void* arg, size_t len);
int op_Sum(const void* arg, size_t len);
int op_Prod(const void* arg, size_t len);
int op_Mean(const void* arg, size_t len);
}


//
// Max
//

template <typename T>
int max_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t i = 0; i < dim0; ++i) {
        T s = pi[i * dim1];
        for (size_t j = 0; j < dim1; ++j) {
            T tmp = pi[i * dim1 + j];
            if(tmp > s) s = tmp;
        }
        po[i] = s;
    }

    return 0;
}

template <typename T>
int max_d2a0(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t j = 0; j < dim1; ++j) {
        T s = pi[j];
        for (size_t i = 0; i < dim0; ++i) {
            T tmp = pi[i * dim1 + j];
            if(tmp > s) s = tmp;
        }
        po[j] = s;
    }

    return 0;
}


template <typename T>
int max_d3a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t i = 0; i < dim0; ++i) {
        for (size_t k = 0; k < dim2; ++k) {
            T s = pi[i * dim12 + k] ;
            for (size_t j = 0; j < dim1; ++j) {
                T tmp = pi[i * dim12 + j * dim2 + k];
                if(tmp > s) s = tmp;
            }
            po[i * dim2 + k] = s;
        }
    }

    return 0;
}

template <typename T>
int max_d3a02(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t j = 0; j < dim1; ++j) {
        T s = pi[j] ;
        for (size_t i = 0; i < dim0; ++i) {
            for (size_t k = 0; k < dim2; ++k) {
                T tmp = pi[i * dim12 + j * dim2 + k];
                if(tmp > s) s = tmp;
            }
        }
        po[j] = s;
    }

    return 0;
}


int op_Max(const void* args, size_t len)
{
    LOG(LOG_TRACE) << __FUNCTION__ << " begin";

    struct Args {
        int dtype;
        int ndims;
        uint64_t in;
        uint64_t out;
        int64_t dim_size[3];
        int axis;
    } const* p;

    CHECK_ARG_LEN(len, sizeof(Args));
    p = reinterpret_cast<const Args*>(args);

    int ret = 1;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " ndims=" << p->ndims << " axis=" << p->axis;

    if (p->dtype == DT_FLOAT) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = max_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = max_d2a0<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = max_d3a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = max_d3a02<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }
    else if (p->dtype == DT_DOUBLE) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = max_d2a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = max_d2a0<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = max_d3a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = max_d3a02<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }

    LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
    return ret;
}



//
// Sum
//

namespace {
template <typename T>
int sum_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t i = 0; i < dim0; ++i) {
        T s = T(0);
        for (size_t j = 0; j < dim1; ++j) {
            s += pi[i * dim1 + j];
        }
        po[i] = s;
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
#ifdef LIBVETF_INTRINSIC
template <>
int sum_d2a0<float>(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    return sum_d2a0_f32(out, in, dim0, dim1);
}
#endif

template <typename T>
int sum_d3a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
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
            po[i * dim2 + k] = s;
        }
    }

    return 0;
}

template <typename T>
int sum_d3a02(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
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
        po[j] = s;
    }

    return 0;
}

#ifdef LIBVETF_INTRINSIC
template <>
int sum_d3a02<float>(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    return sum_d3a02_f32(out, in, dim0, dim1, dim2);
}
#endif
}

int op_Sum(const void* args, size_t len)
{
    LOG(LOG_TRACE) << __FUNCTION__ << " begin";

    struct Args {
        int dtype;
        int ndims;
        uint64_t in;
        uint64_t out;
        int64_t dim_size[3];
        int axis;
    } const* p;

    CHECK_ARG_LEN(len, sizeof(Args));
    p = reinterpret_cast<const Args*>(args);

    int ret = 1;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " ndims=" << p->ndims << " axis=" << p->axis;

    if (p->dtype == DT_FLOAT) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = sum_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = sum_d2a0<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = sum_d3a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = sum_d3a02<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }
    else if (p->dtype == DT_DOUBLE) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = sum_d2a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = sum_d2a0<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = sum_d3a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = sum_d3a02<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }

    LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
    return ret;
}

//
// Prod
//

namespace {
template <typename T>
int prod_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t i = 0; i < dim0; ++i) {
        T s = T(1);
        for (size_t j = 0; j < dim1; ++j) {
            s *= pi[i * dim1 + j];
        }
        po[i] = s;
    }

    return 0;
}

template <typename T>
int prod_d2a0(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t j = 0; j < dim1; ++j) {
        T s = T(1);
        for (size_t i = 0; i < dim0; ++i) {
            s *= pi[i * dim1 + j];
        }
        po[j] = s;
    }

    return 0;
}

template <typename T>
int prod_d3a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t i = 0; i < dim0; ++i) {
        for (size_t k = 0; k < dim2; ++k) {
            T s = T(1);
            for (size_t j = 0; j < dim1; ++j) {
                s *= pi[i * dim12 + j * dim2 + k];
            }
            po[i * dim2 + k] = s;
        }
    }

    return 0;
}

template <typename T>
int prod_d3a02(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t j = 0; j < dim1; ++j) {
        T s = T(1);
        for (size_t i = 0; i < dim0; ++i) {
            for (size_t k = 0; k < dim2; ++k) {
                s *= pi[i * dim12 + j * dim2 + k];
            }
        }
        po[j] = s;
    }

    return 0;
}
}

int op_Prod(const void* args, size_t len)
{
    LOG(LOG_TRACE) << __FUNCTION__ << " begin";

    struct Args {
        int dtype;
        int ndims;
        uint64_t in;
        uint64_t out;
        int64_t dim_size[3];
        int axis;
    } const* p;

    CHECK_ARG_LEN(len, sizeof(Args));
    p = reinterpret_cast<const Args*>(args);

    int ret = 1;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " ndims=" << p->ndims << " axis=" << p->axis;

    if (p->dtype == DT_FLOAT) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = prod_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = prod_d2a0<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = prod_d3a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = prod_d3a02<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }
    else if (p->dtype == DT_DOUBLE) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = prod_d2a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = prod_d2a0<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = prod_d3a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = prod_d3a02<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }
    else if (p->dtype == DT_INT32) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = prod_d2a1<int32_t>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = prod_d2a0<int32_t>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = prod_d3a1<int32_t>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = prod_d3a02<int32_t>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }


    LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
    return ret;
}

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


int op_Mean(const void* args, size_t len)
{
    LOG(LOG_TRACE) << __FUNCTION__ << " begin";

    struct Args {
        int dtype;
        int ndims;
        uint64_t in;
        uint64_t out;
        int64_t dim_size[3];
        int axis;
    } const* p;

    CHECK_ARG_LEN(len, sizeof(Args));
    p = reinterpret_cast<const Args*>(args);

    int ret = 0;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " ndims=" << p->ndims << " axis=" << p->axis;

    if (p->dtype == DT_FLOAT) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = mean_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = mean_d2a0<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = mean_d3a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = mean_d3a02<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }

    LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
    return ret;
}


