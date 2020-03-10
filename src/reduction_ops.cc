#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "vml.h"
#include "vml/types.h"
#include "vml/log.h"

#include <omp.h>

#ifdef LIBVETF_INTRINSIC
#include "intrinsic/intrinsic.h"
#endif

REGISTER_KERNEL("Max", "op_Max");
REGISTER_KERNEL("Min", "op_Min");
REGISTER_KERNEL("Sum", "op_Sum");
REGISTER_KERNEL("Prod", "op_Prod");
REGISTER_KERNEL("Mean", "op_Mean");
REGISTER_KERNEL("All", "op_All");

#define CHECK_ARG_LEN(l0, l1) \
    if ((l0) != (l1)) { \
    LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << (l1) << " expected but " << (l0); \
    return 1; \
    }

extern "C" {
int op_Max(const void* arg, size_t len);
int op_Min(const void* arg, size_t len);
int op_Sum(const void* arg, size_t len);
int op_Prod(const void* arg, size_t len);
int op_Mean(const void* arg, size_t len);
int op_All(const void* arg, size_t len);
}


//
// Max
//

template <typename T>
void max_d2a1_v0(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t ost0)
{
  T* po = reinterpret_cast<T*>(out)  ;
  const T* pi = reinterpret_cast<const T*>(in);

  T s[256] ;
#pragma _NEC vreg(s)

  for (size_t i0 = 0; i0 < dim0; i0+=256) {
    const size_t ilen = dim0-i0 < 256 ? dim0-i0 : 256 ;

    for(size_t i1=0; i1<ilen; ++i1) s[i1] = pi[(ost0+i0+i1) * dim1 ] ;

    for (size_t j = 0; j < dim1; ++j) {
      for(size_t i1=0; i1<ilen; ++i1) {
	const T tmp = pi[(ost0+i0+i1) * dim1 + j] ;
	if( s[i1] < tmp )  s[i1] = tmp ;
      }
    }
    for(size_t i1=0; i1<ilen; ++i1) po[ost0+i0+i1] = s[i1] ;
  }
}

template <typename T>
void max_d2a1_v1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t ost0)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi = reinterpret_cast<const T*>(in);

  for (size_t i = 0; i < dim0; ++i) {
      T s = pi[(ost0+i) * dim1];
      for (size_t j = 0; j < dim1; ++j) {
	  const T tmp = pi[(ost0+i) * dim1 + j] ;
	  if( s < tmp ) s = tmp ;
      }
      po[ost0+i] = s;
  }
}

template <typename T>
int max_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

#pragma omp parallel
    {
      size_t nthreads = omp_get_num_threads() ;
      size_t threadid = omp_get_thread_num() ;

      size_t chunkSize = dim0 / nthreads ;
      size_t remain    = dim0 % nthreads ;

      size_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
      size_t nChunk     = chunkSize + ( threadid < remain ? 1 : 0 ) ;

      if( nChunk > 0 ) {
	if ( dim1 > 256 || dim1 > nChunk ) {
	  max_d2a1_v1<T>(out, in, nChunk, dim1, chunkBegin) ;
	}
	else {
	  max_d2a1_v0<T>(out, in, nChunk, dim1, chunkBegin) ;
	}
      }
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
// Min
//

template <typename T>
void min_d2a1_v0(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t ost0)
{
  T* po = reinterpret_cast<T*>(out)  ;
  const T* pi = reinterpret_cast<const T*>(in);

  T s[256] ;
#pragma _NEC vreg(s)

  for (size_t i0 = 0; i0 < dim0; i0+=256) {
    const size_t ilen = dim0-i0 < 256 ? dim0-i0 : 256 ;

    for(size_t i1=0; i1<ilen; ++i1) s[i1] = pi[(ost0+i0+i1) * dim1 ] ;

    for (size_t j = 0; j < dim1; ++j) {
      for(size_t i1=0; i1<ilen; ++i1) {
	const T tmp = pi[(ost0+i0+i1) * dim1 + j] ;
	if( s[i1] > tmp )  s[i1] = tmp ;
      }
    }
    for(size_t i1=0; i1<ilen; ++i1) po[ost0+i0+i1] = s[i1] ;
  }
}

template <typename T>
void min_d2a1_v1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t ost0)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi = reinterpret_cast<const T*>(in);

  for (size_t i = 0; i < dim0; ++i) {
      T s = pi[(ost0+i) * dim1];
      for (size_t j = 0; j < dim1; ++j) {
	  const T tmp = pi[(ost0+i) * dim1 + j] ;
	  if( s > tmp ) s = tmp ;
      }
      po[ost0+i] = s;
  }
}

template <typename T>
int min_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

#pragma omp parallel
    {
      size_t nthreads = omp_get_num_threads() ;
      size_t threadid = omp_get_thread_num() ;

      size_t chunkSize = dim0 / nthreads ;
      size_t remain    = dim0 % nthreads ;

      size_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
      size_t nChunk     = chunkSize + ( threadid < remain ? 1 : 0 ) ;

      if( nChunk > 0 ) {
	if ( dim1 > 256 || dim1 > nChunk ) {
	  min_d2a1_v1<T>(out, in, nChunk, dim1, chunkBegin) ;
	}
	else {
	  min_d2a1_v0<T>(out, in, nChunk, dim1, chunkBegin) ;
	}
      }
    }

    return 0;
}

template <typename T>
int min_d2a0(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    for (size_t j = 0; j < dim1; ++j) {
        T s = pi[j];
        for (size_t i = 0; i < dim0; ++i) {
            T tmp = pi[i * dim1 + j];
            if(tmp < s) s = tmp;
        }
        po[j] = s;
    }

    return 0;
}


template <typename T>
int min_d3a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t i = 0; i < dim0; ++i) {
        for (size_t k = 0; k < dim2; ++k) {
            T s = pi[i * dim12 + k] ;
            for (size_t j = 0; j < dim1; ++j) {
                T tmp = pi[i * dim12 + j * dim2 + k];
                if(tmp < s) s = tmp;
            }
            po[i * dim2 + k] = s;
        }
    }

    return 0;
}

template <typename T>
int min_d3a02(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t j = 0; j < dim1; ++j) {
        T s = pi[j] ;
        for (size_t i = 0; i < dim0; ++i) {
            for (size_t k = 0; k < dim2; ++k) {
                T tmp = pi[i * dim12 + j * dim2 + k];
                if(tmp < s) s = tmp;
            }
        }
        po[j] = s;
    }

    return 0;
}


int op_Min(const void* args, size_t len)
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
            ret = min_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = min_d2a0<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = min_d3a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = min_d3a02<float>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }
    else if (p->dtype == DT_DOUBLE) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = min_d2a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = min_d2a0<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = min_d3a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = min_d3a02<double>(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
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
void sum_d2a1_v0(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t ost0)
{
  T* po = reinterpret_cast<T*>(out)  ;
  const T* pi = reinterpret_cast<const T*>(in);

  T s[256] ;
#pragma _NEC vreg(s)

  for (size_t i0 = 0; i0 < dim0; i0+=256) {
    const size_t ilen = dim0-i0 < 256 ? dim0-i0 : 256 ;

    for(size_t i1=0; i1<ilen; ++i1) s[i1] = T(0) ;

    for (size_t j = 0; j < dim1; ++j) {
      for(size_t i1=0; i1<ilen; ++i1) {
	s[i1] += pi[(ost0+i0+i1) * dim1 + j];
      }
    }
    for(size_t i1=0; i1<ilen; ++i1) po[ost0+i0+i1] = s[i1] ;
  }
}

template <typename T>
void sum_d2a1_v1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t ost0)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi = reinterpret_cast<const T*>(in);

  for (size_t i = 0; i < dim0; ++i) {
      T s = T(0);
      for (size_t j = 0; j < dim1; ++j) {
	  s += pi[(ost0+i) * dim1 + j];
      }
      po[ost0+i] = s;
  }
}

template <typename T>
int sum_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    T* po = reinterpret_cast<T*>(out);
    const T* pi = reinterpret_cast<const T*>(in);

#pragma omp parallel
    {
      size_t nthreads = omp_get_num_threads() ;
      size_t threadid = omp_get_thread_num() ;

      size_t chunkSize = dim0 / nthreads ;
      size_t remain    = dim0 % nthreads ;

      size_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
      size_t nChunk     = chunkSize + ( threadid < remain ? 1 : 0 ) ;

      if( nChunk > 0 ) {
	if ( dim1 > 256 || dim1 > nChunk ) {
	  sum_d2a1_v1<T>(out, in, nChunk, dim1, chunkBegin) ;
	}
	else {
	  sum_d2a1_v0<T>(out, in, nChunk, dim1, chunkBegin) ;
	}
      }
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

    LOG(LOG_PARAM) << __FUNCTION__ << " ndims=" << p->ndims;


    vml::TensorDesc<3> in;
    vml::TensorDesc<3> out;
    std::vector<int> axis;

    in.dtype = p->dtype;
    in.addr = p->in;
    in.dims = p->ndims;
    in.nelems = 1;

#pragma _NEC novector
    for (int i = 0; i < p->ndims; ++i) {
      int64_t d = p->dim_size[i];
      in.dim_size[i] = d;
      in.nelems *= d;
    }

    out.dtype = p->dtype;
    out.addr = p->out;

    if (p->ndims == 3 && p->axis == 0) { // axis = {0, 2}
      out.dims = 1;
      out.dim_size[0] = p->dim_size[1];
      out.nelems = p->dim_size[1];
      axis.push_back(0);
      axis.push_back(2);
    } else { // axis = {p->axis}
      out.dims = p->ndims - 1;
      out.nelems = 1;
      int j = 0;
#pragma _NEC novector
      for (int i = 0; i < p->ndims; ++i) {
        if (i != p->axis) {
          out.dim_size[j++] = p->dim_size[i];
          out.nelems *= p->dim_size[i];
        }
      }
      axis.push_back(p->axis);
    }

    return vml::mean(out, in, axis);
}


//
// All
//

namespace {

int all_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    bool* po = reinterpret_cast<bool*>(out);
    const bool* pi = reinterpret_cast<const bool*>(in);

    for (size_t i = 0; i < dim0; ++i) {
        bool s = true ;
        for (size_t j = 0; j < dim1; ++j) {
            s *= pi[i * dim1 + j];
        }
        po[i] = s;
    }

    return 0;
}


int all_d2a0(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    bool* po = reinterpret_cast<bool*>(out);
    const bool* pi = reinterpret_cast<const bool*>(in);

    for (size_t j = 0; j < dim1; ++j) {
        bool s = true ;
        for (size_t i = 0; i < dim0; ++i) {
            s = s && pi[i * dim1 + j];
        }
        po[j] = s;
    }

    return 0;
}


int all_d3a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    bool* po = reinterpret_cast<bool*>(out);
    const bool* pi = reinterpret_cast<const bool*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t i = 0; i < dim0; ++i) {
        for (size_t k = 0; k < dim2; ++k) {
            bool s = true ;
            for (size_t j = 0; j < dim1; ++j) {
                s = s && pi[i * dim12 + j * dim2 + k];
            }
            po[i * dim2 + k] = s;
        }
    }

    return 0;
}

int all_d3a02(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    bool* po = reinterpret_cast<bool*>(out);
    const bool* pi = reinterpret_cast<const bool*>(in);

    size_t dim12 = dim1 * dim2;

    for (size_t j = 0; j < dim1; ++j) {
        bool s = true ;
        for (size_t i = 0; i < dim0; ++i) {
            for (size_t k = 0; k < dim2; ++k) {
                s = s && pi[i * dim12 + j * dim2 + k];
            }
        }
        po[j] = s;
    }

    return 0;
}
}

int op_All(const void* args, size_t len)
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

    if (p->dtype == DT_BOOL) {
        if (p->ndims == 2 && p->axis == 1) {
            ret = all_d2a1(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 2 && p->axis == 0) {
            ret = all_d2a0(p->out, p->in, p->dim_size[0], p->dim_size[1]);
        }
        if (p->ndims == 3 && p->axis == 1) {
            ret = all_d3a1(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
        if (p->ndims == 3 && p->axis == 0) {
            ret = all_d3a02(p->out, p->in, p->dim_size[0], p->dim_size[1], p->dim_size[2]);
        }
    }

    LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
    return ret;
}


