#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "vml/types.h"
#include "vml/log.h"

#include <omp.h>

REGISTER_KERNEL("Gather",      "op_Gather");
REGISTER_KERNEL("BatchGather", "op_BatchGather");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      LOG(LOG_ERROR) << __FUNCTION__ << ": illegal argument length: " << (l1) << " expected but " << (l0); \
      return 1; \
  }

extern "C" {
  int op_Gather(const void* arg, size_t len);
  int op_BatchGather(const void* arg, size_t len);
}


//
// Gather
//

namespace {

template <typename T, typename Index>
int gather(int64_t outer_size,
           int64_t gather_dim_size,
	   int64_t inner_size,
           int64_t nindex,
           uint64_t src_ptr,
	   uint64_t idx_ptr,
	   uint64_t dst_ptr)
{
  const T* src = reinterpret_cast<const T*>(src_ptr);
  const Index* idx = reinterpret_cast<const Index*>(idx_ptr);
  T* dst = reinterpret_cast<T*>(dst_ptr);

  for(int64_t o=0; o<outer_size; o++) {
    for(int64_t g=0; g<nindex; g++) {
      const int64_t j = idx[g] ;
      for(int64_t i=0; i<inner_size; i++) {
	dst[(o*nindex+g)*inner_size+i] = src[(o*gather_dim_size+j)*inner_size+i] ;
      }
    }
  }

  return 0 ;
}
}

int op_Gather(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int dtype, idxtype;
    int64_t outer_size ;
    int64_t gather_dim_size;
    int64_t inner_size ;
    int64_t nindex ;
    uint64_t src_ptr, idx_ptr, dst_ptr ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " idxtype=" << p->idxtype;

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    if ( p->idxtype == DT_INT32 ) {
      ret = gather<float, int32_t> (p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                    p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = gather<float, int64_t> (p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                    p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if ( p->idxtype == DT_INT32 ) {
      ret = gather<double, int32_t> (p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                     p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = gather<double, int64_t> (p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                     p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}


//
// BatchGather
//

namespace {

template <typename T, typename Index>
int batch_gather(int64_t batch_size,
                  int64_t outer_size,
                  int64_t gather_dim_size,
	          int64_t inner_size,
                  int64_t nindex,
                  uint64_t src_ptr,
	          uint64_t idx_ptr,
	          uint64_t dst_ptr)
{
  const T* src = reinterpret_cast<const T*>(src_ptr);
  const Index* idx = reinterpret_cast<const Index*>(idx_ptr);
  T* dst = reinterpret_cast<T*>(dst_ptr);

  for(int64_t b=0; b<batch_size; b++) {
    for(int64_t o=0; o<outer_size; o++) {
      for(int64_t g=0; g<nindex/batch_size; g++) {
	const int64_t j = idx[b*nindex+g] ;
	for(int64_t i=0; i<inner_size; i++) {
	  dst[((b*outer_size+o)*nindex+g)*inner_size+i]
	      = src[((b*outer_size+o)*gather_dim_size+j)*inner_size+i] ;
	}
      }
    }
  }

  return 0 ;
}
}

int op_BatchGather(const void* args, size_t len)
{
  LOG(LOG_TRACE) << __FUNCTION__ << " begin";

  struct Args {
    int dtype, idxtype;
    int64_t batch_size ;
    int64_t outer_size ;
    int64_t gather_dim_size;
    int64_t inner_size ;
    int64_t nindex ;
    uint64_t src_ptr, idx_ptr, dst_ptr ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << p->dtype << " idxtype=" << p->idxtype;

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    if ( p->idxtype == DT_INT32 ) {
      ret = batch_gather<float, int32_t> (p->batch_size, p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                          p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = batch_gather<float, int64_t> (p->batch_size, p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                          p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if ( p->idxtype == DT_INT32 ) {
      ret = batch_gather<double, int32_t> (p->batch_size, p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                           p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = batch_gather<double, int64_t> (p->batch_size, p->outer_size, p->gather_dim_size, p->inner_size, p->nindex,
                                           p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}
