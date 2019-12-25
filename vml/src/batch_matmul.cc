#include "vml/types.h"
#include "vml/log.h"
#include "vml.h"

#include <omp.h>
#define ADD_
#include <cblas_f77.h>
#undef ADD_


//
// BatchMatmul
//

namespace {

#define GEMM_ARGS(T) \
char* transa, char* transb, \
const int* M, const int* N, const int* K, \
const T* alpha, \
const T* A, const int* lda, \
const T* B, const int* ldb, \
const T* beta, \
T* C, const int* ldc

#define GEMM_REAL_ARGS \
transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc

template <typename T> void blas_gemm(GEMM_ARGS(T)) { assert(false && "blas_gemm: not implemented"); }
template<> void blas_gemm<float>(GEMM_ARGS(float)) { sgemm_(GEMM_REAL_ARGS); }

// C[M x N] = A[M x K] * B[K x N] ([rows x cols] in row-major)
//
// M[H x W] (rows x cols in row-major) = M[W x H] (rows x cols in col-major)
//
// C[M x N] (RxC in RM)
//   = C[N x M] (RxC in CM)
//   = B[N x K] (RxC in CM) * A[K x M] (RxC in CM)
//   = B[K x N] (RxC in RM) * A[M x K] (RxC in RM)
//
template<typename T, char TransA, char TransB>
int matmul(uint64_t c, uint64_t a, uint64_t b, int M, int N, int K)
{
  T* C = reinterpret_cast<T*>(c);
  const T* A = reinterpret_cast<const T*>(a);
  const T* B = reinterpret_cast<const T*>(b);

  T alpha = T(1);
  T beta = T(0);

  char transa = TransA;
  char transb = TransB;
  int lda = TransA == 'N' ? K : M;
  int ldb = TransB == 'N' ? N : K;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads() ;
    int threadid = omp_get_thread_num() ;

    int chunkSize = M / nthreads ;
    int remain    = M % nthreads ;

    int chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
    int myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

    int offset    = TransA == 'N' ? K : 1 ;

    if( myChunk > 0 ) {
      blas_gemm<T>(&transb, &transa, &N, &myChunk, &K, &alpha, B, &ldb, A+offset*chunkBegin, &lda, &beta, C+N*chunkBegin, &N);
    }
  }

  return 0;
}

template <typename T>
int fill0 (T* po, int64_t n)
{
  for(int64_t i=0; i<n; i++)
    po[i] = T(0) ;
  return 0 ;
}

template <typename T>
int BatchMatMul(
    vml::Tensor const & in_x,
    vml::Tensor const & in_y,
    vml::Tensor const & out,
    const bool adj_x,
    const bool adj_y
)
{
  T* po = reinterpret_cast<T*>(out.addr);
  const T* px = reinterpret_cast<const T*>(in_x.addr);
  const T* py = reinterpret_cast<const T*>(in_y.addr);

  if( in_x.nelems == 0 || in_y.nelems == 0 )
    return fill0<T>(po, out.nelems) ;

//  if(IsSameBatchShape) {}

  const int64_t dims = out.dims ;
  int64_t reshaped_x_dim_size[dims] ;
  int64_t reshaped_y_dim_size[dims] ;

#pragma _NEC novector
  for(int64_t dim=0; dim < dims; dim++) {
    reshaped_x_dim_size[dim] = dim-(dims-in_x.dims) >= 0 ? in_x.dim_size[dim-(dims-in_x.dims)] : 1 ;
    reshaped_y_dim_size[dim] = dim-(dims-in_y.dims) >= 0 ? in_y.dim_size[dim-(dims-in_y.dims)] : 1 ;
  }

  const int64_t x_matsize = reshaped_x_dim_size[dims-1] * reshaped_x_dim_size[dims-2] ;
  const int64_t y_matsize = reshaped_y_dim_size[dims-1] * reshaped_y_dim_size[dims-2] ;

  const int64_t out_row = out.dim_size[dims-2] ;
  const int64_t out_col = out.dim_size[dims-1] ;
  const int64_t out_matsize = out_row * out_col ;
  const int64_t out_batch   = out.nelems / out_matsize ;

  int (*mm)(uint64_t,uint64_t,uint64_t,int,int,int) ;
  int M, N, K ;

  if (!adj_x && !adj_y) {
    mm = matmul<T,'N','N'> ;
    M = reshaped_x_dim_size[dims-2] ;
    N = reshaped_y_dim_size[dims-1] ;
    K = reshaped_x_dim_size[dims-1] ;
  } else if (!adj_x && adj_y)  {
    mm = matmul<T,'N','T'> ;
    M = reshaped_x_dim_size[dims-2] ;
    N = reshaped_y_dim_size[dims-2] ;
    K = reshaped_x_dim_size[dims-1] ;
  } else if (adj_x && !adj_y)  {
    mm = matmul<T,'T','N'> ;
    M = reshaped_x_dim_size[dims-1] ;
    N = reshaped_y_dim_size[dims-1] ;
    K = reshaped_x_dim_size[dims-2] ;
  } else {
    mm = matmul<T,'T','T'> ;
    M = reshaped_x_dim_size[dims-1] ;
    N = reshaped_y_dim_size[dims-2] ;
    K = reshaped_x_dim_size[dims-2] ;
  }

  int64_t stO[dims-2] ;
  stO[dims-3] = 1 ;
#pragma _NEC novector
  for (int64_t dim = dims-4; dim >= 0; dim--) {
    stO[dim] = stO[dim+1] * out.dim_size[dim+1] ;
  }

  for(int64_t io=0; io<out_batch; io++) {
    int64_t tmp = io ;
    int64_t ix = 0 ;
    int64_t iy = 0 ;

#pragma _NEC novector
    for(int64_t dim=0; dim < dims-2; dim++) {
      int64_t tmp1 = tmp / stO[dim] ;
      ix = (ix * reshaped_x_dim_size[dim]) + tmp1 % reshaped_x_dim_size[dim];
      iy = (iy * reshaped_y_dim_size[dim]) + tmp1 % reshaped_y_dim_size[dim];
      tmp = tmp % stO[dim];
    }

    mm((uint64_t)(&po[io*out_matsize]), (uint64_t)(&px[ix*x_matsize]), (uint64_t)(&py[iy*y_matsize]), M, N, K) ;
    LOG(LOG_TRACE) << __FUNCTION__ << " ix=" << ix << " iy=" << iy << " io=" << io;
  }

  return 0 ;
}

}

int vml::batch_matmul(
    vml::Tensor const & in_x,
    vml::Tensor const & in_y,
    vml::Tensor const & out,
    const bool adj_x,
    const bool adj_y
)
{

  LOG(LOG_TRACE) << __FUNCTION__ << " begin";
  LOG(LOG_PARAM) << __FUNCTION__
    <<": in_x=" << in_x
    << " in_y=" << in_y
    << " out="  << out
    << " adj_x=" << adj_x
    << " adj_y=" << adj_y ;

  int ret = 1 ;

  if ( in_x.dtype != out.dtype || in_x.dtype != out.dtype ) {
    LOG(LOG_ERROR) << __FUNCTION__ << " dtype of tensors must be same.";
    return 1 ;
  }

  switch(out.dtype) {
  case DT_FLOAT :
    ret = BatchMatMul<float>(in_x, in_y, out, adj_x, adj_y) ;
    break ;
  default :
    LOG(LOG_ERROR) << __FUNCTION__ << " unsupported data type on VE.";
    return 1 ;
  }

  LOG(LOG_TRACE) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}


