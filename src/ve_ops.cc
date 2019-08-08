#include <cstdint>
#include "asl.h"
#include "types.h"
#include <sstream>
#include "ve_ops_common.h"
#include "vml.h"

#ifdef LIBVETF_INTRINSIC
#include "libvetfkernel.h"
#endif

namespace {

#if 0
std::string vml::TensorToString(const vml::Tensor *t)
{
    std::stringstream s;

    s << "[dtype=" << t->dtype
      << ",dims=" << t->dims
      << "[";
    for (int i = 0; i < t->dims; ++i) {
        s << " " << t->dim_size[i];
    }
    s  << " ],nelems=" << t->nelems
      << "]";
    return s.str();
}
#endif


template <typename T>
int op_select_nn(uint64_t out,
                 uint64_t cond,
                 uint64_t then,
                 uint64_t else_,
                 size_t n)
{
    T* po = reinterpret_cast<T*>(out);
    const bool* pc = reinterpret_cast<const bool*>(cond);
    const T* pt = reinterpret_cast<const T*>(then);
    const T* pe = reinterpret_cast<const T*>(else_);

    for (size_t i = 0; i < n; ++i) {
        po[i] = pc[i] ? pt[i] : pe[i];
    }
    return 0;
}

int op_select(const VEOpArgs& args)
{
    // LOG(LOG_PARAM) << __FUNCTION__ << ": ninputs=" << args.ninputs() << " noutputs=" << args.noutputs();
    if (args.nArguments() != 4)
        return 1;

    const vml::Tensor *t0 = args.arg<vml::Tensor>(0) ;
    const vml::Tensor *t1 = args.arg<vml::Tensor>(1) ;
    const vml::Tensor *t2 = args.arg<vml::Tensor>(2) ;
    const vml::Tensor *t3 = args.arg<vml::Tensor>(3) ;

    LOG(LOG_PARAM) << __FUNCTION__ << ":"
	   << " in0=" << t0->to_s()
	   << " in1=" << t1->to_s()
	   << " in2=" << t2->to_s()
	   << " out=" << t3->to_s();

    if (t0->dtype == DT_BOOL) {
        if (t1->dtype == DT_FLOAT && t2->dtype == DT_FLOAT &&
	    t3->dtype == DT_FLOAT) {
	    if (t0->nelems == t1->nelems && t0->nelems == t2->nelems) {
	        return op_select_nn<float>(t3->addr, t0->addr,
					   t1->addr, t2->addr,
					   t0->nelems);
	    }
        } else if (t1->dtype == DT_DOUBLE && t2->dtype == DT_DOUBLE &&
		   t3->dtype == DT_DOUBLE) {
	    if (t0->nelems == t1->nelems && t0->nelems == t2->nelems) {
	        return op_select_nn<double>(t3->addr, t0->addr,
					   t1->addr, t2->addr,
					   t0->nelems);
	    }
        } else if (t1->dtype == DT_INT32 && t2->dtype == DT_INT32 &&
		   t3->dtype == DT_INT32) {
	    if (t0->nelems == t1->nelems && t0->nelems == t2->nelems) {
	        return op_select_nn<int32_t>(t3->addr, t0->addr,
					     t1->addr, t2->addr,
					     t0->nelems);
	    }
        } else if (t1->dtype == DT_INT64 && t2->dtype == DT_INT64 &&
		   t3->dtype == DT_INT64) {
	    if (t0->nelems == t1->nelems && t0->nelems == t2->nelems) {
	        return op_select_nn<int64_t>(t3->addr, t0->addr,
					     t1->addr, t2->addr,
					     t0->nelems);
	    }
        }
    }

    //LOG(LOG_DETAIL) << __FUNCTION__ << ": return1";
    return 1;
}

int op_randomUniform(const VEOpArgs& args)
{
    if (args.nArguments() != 1)
        return 1;

    const vml::Tensor* t = args.arg<vml::Tensor>(0);

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << t->dtype << " nelems=" << t->nelems;

    if (t->dtype == DT_FLOAT) {
        float* p = reinterpret_cast<float*>(t->addr);
        ASL::getRandom(t->nelems, p) ;
    }

    return 0;
}

} // namespace

DEFINE_KERNEL(Select, op_select);
DEFINE_KERNEL(RandomUniform, op_randomUniform);

//
// Cast
//

namespace {

template <typename TO, typename TI>
void cast(const vml::Tensor* to, const vml::Tensor* ti) {
    TO* po = reinterpret_cast<TO*>(to->addr);
    const TI* pi = reinterpret_cast<const TI*>(ti->addr);

    for (size_t i = 0; i < ti->nelems; ++i)
        po[i] = pi[i];
}

template <typename TO>
void cast_from_bool(const vml::Tensor* to, const vml::Tensor* ti) {
    TO* po = reinterpret_cast<TO*>(to->addr);
    const bool* pi = reinterpret_cast<const bool*>(ti->addr);

#if 0 // original ( partially vectorized ) 
    for (size_t i = 0; i < ti->nelems; ++i)
        po[i] = pi[i];
#else
    const size_t n = ti->nelems ;

    const size_t vloop_begin =  (to->addr) & 0x3 ;
    const size_t vloop_end   =  n   & 0xFFFFFFFFFFFFFFFC ;

#pragma novector
    for(size_t i=0; i < vloop_begin ; i++) {
        po[i] = pi[i] ;
    }

    const int*  pi_i = reinterpret_cast<const int*>(&pi[vloop_begin]);
    for(size_t j=0; j < (vloop_end - vloop_begin)>>2 ; j++) {
        const int32_t b  = pi_i[j] ;

        const int32_t b0 =   b        & 0xFF ;
        const int32_t b1 = ( b >>  8) & 0xFF ;
        const int32_t b2 = ( b >> 16) & 0xFF ;
        const int32_t b3 = ( b >> 24)        ;

        po[vloop_begin+4*j+0] = b0 ;
        po[vloop_begin+4*j+1] = b1 ;
        po[vloop_begin+4*j+2] = b2 ;
        po[vloop_begin+4*j+3] = b3 ;
    }


#pragma novector
    for(size_t i=vloop_end; i < n ; i++) {
        po[i] = pi[i] ;
    }
#endif
}

int op_cast(const VEOpArgs& args)
{
    if (args.nArguments() != 2)
        return 1;
    const vml::Tensor* ti = args.arg<vml::Tensor>(0);
    const vml::Tensor* to = args.arg<vml::Tensor>(1);

    LOG(LOG_PARAM) << __FUNCTION__ << " ti=" << ti << " to=" << to;

    if (!ti || !to)
        return 1;

    LOG(LOG_PARAM) << __FUNCTION__ << " ti=" << ti->to_s() << " to=" << to->to_s();

    if (ti->nelems != to->nelems)
        return 1;

    if (ti->dtype == DT_BOOL && to->dtype == DT_FLOAT) {
        cast_from_bool<float>(to, ti);
    } else if (ti->dtype == DT_INT32 && to->dtype == DT_FLOAT) {
        cast<float, int32_t>(to, ti);
    } else if (ti->dtype == DT_INT64 && to->dtype == DT_FLOAT) {
        cast<float, int64_t>(to, ti);
    } else if (ti->dtype == DT_BOOL && to->dtype == DT_INT32) {
        cast_from_bool<int32_t>(to, ti);
    } else if (ti->dtype == DT_UINT16 && to->dtype == DT_INT32) {
        cast<int32_t, uint16_t>(to, ti);
    } else if (ti->dtype == DT_INT8 && to->dtype == DT_BOOL) {
        cast<bool, int8_t>(to, ti);
    } else {
        return 1;
    }

    return 0;
}

} // namespace

DEFINE_KERNEL(Cast, op_cast);

//
// Tile
//

namespace {

template<typename T>
int tile_dim3(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_DETAIL) << __FUNCTION__;
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;

    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        const size_t ix0 = i0 ;
        const size_t iy0 = i0 % sy[0] ;
        for (size_t i1 = 0; i1 < sx[1]; ++i1) {
            const size_t ix1 = ix0 * sx[1] + i1 ;
            const size_t iy1 = iy0 * sy[1] + (i1 % sy[1]) ;
            for (size_t i2 = 0; i2 < sx[2]; ++i2) {
                const size_t ix2 = ix1 * sx[2] + i2 ;
                const size_t iy2 = iy1 * sy[2] + (i2 % sy[2]) ;
                px[ix2] = py[iy2] ;
            }
        }
    }

    return 0;
}

// X = TILE(Y)
// X = [d0, d1, d2, d3]
// Y = [e0, e1, 1, 1]
template<typename T>
int tile_dim4_11(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_DETAIL) << __FUNCTION__;
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;

#pragma _NEC novector
    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        const size_t ix0 = i0 ;
        const size_t iy0 = i0 % sy[0] ;
#pragma _NEC novector
        for (size_t i1 = 0; i1 < sx[1]; ++i1) {
            const size_t ix1 = ix0 * sx[1] + i1 ;
            const size_t iy1 = iy0 * sy[1] + (i1 % sy[1]) ;
            for (size_t i23 = 0; i23 < sx[2] * sx[3] ; ++i23) {
                const size_t ix23 = ix1 * sx[2] * sx[3] + i23 ;
                px[ix23] = py[iy1] ;
            }
        }
    }

    return 0;
}

template<typename T>
int tile_dim4(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_DETAIL) << __FUNCTION__;
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;

    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        const size_t ix0 = i0 ;
        const size_t iy0 = i0 % sy[0] ;
        for (size_t i1 = 0; i1 < sx[1]; ++i1) {
            const size_t ix1 = ix0 * sx[1] + i1 ;
            const size_t iy1 = iy0 * sy[1] + (i1 % sy[1]) ;
            for (size_t i2 = 0; i2 < sx[2]; ++i2) {
                const size_t ix2 = ix1 * sx[2] + i2 ;
                const size_t iy2 = iy1 * sy[2] + (i2 % sy[2]) ;
                for (size_t i3 = 0; i3 < sx[3]; ++i3) {
                    const size_t ix3 = ix2 * sx[3] + i3 ;
                    const size_t iy3 = iy2 * sy[3] + (i3 % sy[3]) ;
                    px[ix3] = py[iy3] ;
                }
            }
        }
    }

    return 0;
}

// X = TILE(Y)
// X = [d0, d1, d2, d3, d4]
// Y = [e0, e1, e2, 1, 1]
template<typename T>
int tile_dim5_11(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_DETAIL) << __FUNCTION__;
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;

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
#ifdef LIBVETF_INTRINSIC
template<>
int tile_dim5_11<float>(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_DETAIL) << __FUNCTION__ << "(intrinsic)";
    float* px = (float*)(X.addr);
    float const* py = (float*)(Y.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;

    return tile_dim5_11_f32(px,py,sx,sy);
}
#endif

template<typename T>
int tile_dim5(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_DETAIL) << __FUNCTION__;
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;

    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        const size_t ix0 = i0 ;
        const size_t iy0 = i0 % sy[0] ;
        for (size_t i1 = 0; i1 < sx[1]; ++i1) {
            const size_t ix1 = ix0 * sx[1] + i1 ;
            const size_t iy1 = iy0 * sy[1] + (i1 % sy[1]) ;
            for (size_t i2 = 0; i2 < sx[2]; ++i2) {
                const size_t ix2 = ix1 * sx[2] + i2 ;
                const size_t iy2 = iy1 * sy[2] + (i2 % sy[2]) ;
                for (size_t i3 = 0; i3 < sx[3]; ++i3) {
                    const size_t ix3 = ix2 * sx[3] + i3 ;
                    const size_t iy3 = iy2 * sy[3] + (i3 % sy[3]) ;
                    for (size_t i4 = 0; i4 < sx[4]; ++i4) {
                        const size_t ix4 = ix3 * sx[4] + i4 ;
                        const size_t iy4 = iy3 * sy[4] + (i4 % sy[4]) ;
                        px[ix4] = py[iy4] ;
                    }
                }
            }
        }
    }

    return 0;
}


int op_tile(const VEOpArgs& args)
{
    if (args.nArguments() != 2)
        return 1;
    const vml::Tensor* ti = args.arg<vml::Tensor>(0);
    const vml::Tensor* to = args.arg<vml::Tensor>(1);

    LOG(LOG_PARAM) << __FUNCTION__ << " ti=" << ti->to_s() << " to=" << to->to_s();

    int rc = 1 ;

    if (ti->dtype == DT_FLOAT && to->dtype == DT_FLOAT) {
        const float* pi = reinterpret_cast<const float*>(ti->addr);
        float* po = reinterpret_cast<float*>(to->addr);
        if (ti->nelems == 1) {
            for (size_t i = 0; i < to->nelems; ++i) {
                po[i] = pi[0];
            }
            rc = 0 ;
        } else if (ti->dims == 2 && to->dims == 2
                   && ti->dim_size[0] == to->dim_size[0]
                   && ti->dim_size[1] == 1) {
            for (size_t i = 0; i < ti->dim_size[0]; ++i) {
                for (size_t j = 0; j < to->dim_size[1]; ++j) {
                    po[i * to->dim_size[1] + j] = pi[i];
                }
            }
            rc = 0 ;
        } else if (ti->dims == 2 && to->dims == 2
                   && ti->dim_size[1] == to->dim_size[1]
                   && ti->dim_size[0] == 1) {
            for (size_t i = 0; i < to->dim_size[0]; ++i) {
                for (size_t j = 0; j < to->dim_size[1]; ++j) {
                    po[i * to->dim_size[1] + j] = pi[j];
                }
            }
            rc = 0 ;
        } else if ( ti->dims == to->dims && ti->dims == 3 ) {
            rc = tile_dim3<float>(*to, *ti) ;
        } else if ( ti->dims == to->dims && ti->dims == 4 ) {
            if( ti->dim_size[2]==1 && ti->dim_size[3]==1 ) {
                rc = tile_dim4_11<float>(*to, *ti) ;
            }
            else {
                rc = tile_dim4<float>(*to, *ti) ;
            }
        } else if ( ti->dims == to->dims && ti->dims == 5 ) {
            if( ti->dim_size[3]==1 && ti->dim_size[4]==1 ) {
                rc = tile_dim5_11<float>(*to, *ti) ;
            }
            else {
                rc = tile_dim5<float>(*to, *ti) ;
            }
        }
    }

    return rc;
}
} // namespace

DEFINE_KERNEL(Tile, op_tile);


//
// SoftmaxXentWithLogits
//

template<typename T>
int softmax_xent_with_logits_same_shape(
        int64_t logits_addr,
        int64_t labels_addr,
        int64_t scratch_addr,
        int64_t loss_addr,
        int64_t back_addr,
        size_t batch_size,
        size_t num_classes )
{
    T* logits  = reinterpret_cast<T*>(logits_addr);
    T* labels  = reinterpret_cast<T*>(labels_addr);
    T* scratch = reinterpret_cast<T*>(scratch_addr);
    T* loss    = reinterpret_cast<T*>(loss_addr);
    T* back    = reinterpret_cast<T*>(back_addr);

#if 1 /* optimized version */
    for(int64_t i=0; i<batch_size; i++) {
        T max_logits = T(0.) ;
        for(int64_t j=0; j<num_classes; j++) {
            if(max_logits < logits[i*num_classes+j]) max_logits = logits[i*num_classes+j] ;
        }

        T sum_exp_logits = T(0.) ;
        for(int64_t j=0; j<num_classes; j++) {
            const T logit = logits[i*num_classes+j] - max_logits;
            sum_exp_logits += std::exp(logit) ;
            back[i*num_classes+j] = logit ;
        }

        T l = T(0.) ;
        for(int64_t j=0; j<num_classes; j++) {
            const T logit = back[i*num_classes+j] ;
            const T label = labels[i*num_classes+j] ;

            l += label * (std::log(sum_exp_logits) - logit);
            back[i*num_classes+j] = std::exp(logit) / sum_exp_logits - label ;
        }
        loss[i] = l ;
    }
#else /* original version */
    // max_logits along classes.
    for(int64_t i=0; i<batch_size; i++) {
        T max_logits = T(0.) ;
        for(int64_t j=0; j<num_classes; j++) {
            if(max_logits < logits[i*num_classes+j]) max_logits = logits[i*num_classes+j] ;
        }
        scratch[i] = max_logits ;
    }

    // logits - max_logits.
    for(int64_t i=0; i<batch_size; i++) {
        const T max_logits = scratch[i] ;
        for(int64_t j=0; j<num_classes; j++) {
            back[i*num_classes+j] = logits[i*num_classes+j] - max_logits;
        }
    }

    // sum(exp(logits - max_logits)) along classes.
    for(int64_t i=0; i<batch_size; i++) {
        T sum_exp_logits = T(0.) ;
        for(int64_t j=0; j<num_classes; j++) {
            sum_exp_logits += std::exp(back[i*num_classes+j]) ;
        }
        scratch[i] = sum_exp_logits ;
    }


    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    for(int64_t i=0; i<batch_size; i++) {
        const T sum_exp_logits = scratch[i] ;
        T l = T(0.) ;
        for(int64_t j=0; j<num_classes; j++) {
            l += labels[i*num_classes+j] * (std::log(sum_exp_logits) - back[i*num_classes+j]);
        }
        loss[i] = l ;
    }

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    for(int64_t i=0; i<batch_size; i++) {
        const T sum_exp_logits = scratch[i] ;
        for(int64_t j=0; j<num_classes; j++) {
            back[i*num_classes+j] = std::exp(back[i*num_classes+j]) / sum_exp_logits - labels[i*num_classes+j] ;
        }
    }
#endif

    return 0 ;
}

namespace {
int op_softmax_xent_with_logits(const VEOpArgs& args)
{
    if (args.nArguments() != 5)
        return 5;

    const vml::Tensor* logits_in = args.arg<vml::Tensor>(0);
    const vml::Tensor* labels_in = args.arg<vml::Tensor>(1);
    const vml::Tensor* scratch = args.arg<vml::Tensor>(2);
    const vml::Tensor* loss_out = args.arg<vml::Tensor>(3);
    const vml::Tensor* back_out = args.arg<vml::Tensor>(4);

    LOG(LOG_PARAM) << __FUNCTION__
           << " logits_in=" << logits_in->to_s()
           << " labels_in=" << labels_in->to_s()
           << " scratch="   << scratch->to_s()
           << " loss_out="  << loss_out->to_s()
           << " back_out="  << back_out->to_s() ;

    if ( logits_in->dtype == DT_FLOAT
         && labels_in->dtype == DT_FLOAT
         && scratch->dtype == DT_FLOAT
         && loss_out->dtype == DT_FLOAT
         && back_out->dtype == DT_FLOAT ) {

        int r=1;

        // TODO : add other patterns (ex:n1,1n)
        if (logits_in->dims == 2 && labels_in->dims == 2
                && logits_in->dim_size[0] == labels_in->dim_size[0]
                && logits_in->dim_size[1] == labels_in->dim_size[1] ) {
            r = softmax_xent_with_logits_same_shape<float>(
                        logits_in->addr, labels_in->addr,
                        scratch->addr, loss_out->addr, back_out->addr,
                        logits_in->dim_size[0], labels_in->dim_size[1] ) ;
        }

        return r;
    }
    return 1;
}
} // namespace

DEFINE_KERNEL(SoftmaxXentWithLogits, op_softmax_xent_with_logits);


//
// Concat
//
template<typename T>
int concat(uint64_t n, uint64_t dim0, uint64_t out,
           uint64_t *ins, uint64_t *dim1_offset )
{
    const T** pi = reinterpret_cast<const T**>(ins);
    const uint64_t* dim1ost = reinterpret_cast<const uint64_t*>(dim1_offset);
    T* po = reinterpret_cast<T*>(out);

    if ( dim0 == 1 ) {
#pragma omp parallel for
      for(int64_t j=0; j<n; j++) {
	int64_t idim1 = dim1ost[j+1] - dim1ost[j] ;
	for(int64_t k=0; k<idim1; k++) {
	  po[dim1ost[j]+k] = pi[j][k] ;
	}
      }
    }
    else {
#pragma omp parallel for
      for(int64_t i=0; i<dim0; i++) {
	T* po1 = po + i*dim1ost[n] ;
	for(int64_t j=0; j<n; j++) {
	  int64_t idim1 = dim1ost[j+1] - dim1ost[j] ;
	  for(int64_t k=0; k<idim1; k++) {
	    po1[k] = pi[j][i*idim1+k] ;
	  }
	  po1+=idim1 ;
	}
      }
    }

    return 0 ;
}

namespace {
int op_Concat(const VEOpArgs& args)
{
    int ret=1;

    int narg = 0 ;
    const int64_t dtype              = *args.arg<int64_t>(narg++) ;
    const uint64_t n_input           = *args.arg<uint64_t>(narg++) ;
    const uint64_t outputs_flat_dim0 = *args.arg<uint64_t>(narg++) ;
    const uint64_t output_ptr        = *args.arg<uint64_t>(narg++) ;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

    uint64_t ins[n_input] ;
    uint64_t dim1_offset[n_input+1] ;
    dim1_offset[0] = *args.arg<uint64_t>(narg++) ;

    for(int64_t i=0; i<n_input; i++) {
        ins[i]   = *args.arg<uint64_t>(narg++) ;
        dim1_offset[i+1] = *args.arg<uint64_t>(narg++) ;
    }

    if (dtype == DT_FLOAT) {
        ret = concat<float>(n_input, outputs_flat_dim0, output_ptr, ins, dim1_offset ) ;
    }
    else if (dtype == DT_INT32) {
        ret = concat<int32_t>(n_input, outputs_flat_dim0, output_ptr, ins, dim1_offset ) ;
    }
    else if (dtype == DT_DOUBLE) {
        ret = concat<double>(n_input, outputs_flat_dim0, output_ptr, ins, dim1_offset ) ;
    }

    return ret;
}
}

DEFINE_KERNEL(Concat, op_Concat);


//
// Split
//
template<typename T>
int split(const int64_t  num_split,
          const int64_t  prefix_dim_size,
          const int64_t  split_dim_size,
          const int64_t  suffix_dim_size,
          const uint64_t input_addr,
          uint64_t       *output_addrs)
{
    const T* pi = reinterpret_cast<const T*>(input_addr);
    T** po = reinterpret_cast<T**>(output_addrs);

    const int64_t split_dim_output_size = split_dim_size / num_split ;

    for(int64_t i=0; i<prefix_dim_size; i++) {
        for(int64_t n=0; n<num_split; n++) {
            for(int64_t j=0; j<split_dim_output_size*suffix_dim_size; j++) {
                po[n][(i*split_dim_output_size)*suffix_dim_size+j]
                        = pi[(i*split_dim_size+(n*split_dim_output_size))*suffix_dim_size+j] ;
            }
        }
    }
    return 0 ;
}

namespace {
int op_Split(const VEOpArgs& args)
{
    int ret=1;

    int narg = 0 ;
    const int64_t num_split       = *args.arg<int64_t>(narg++) ;
    const int64_t prefix_dim_size = *args.arg<int64_t>(narg++) ;
    const int64_t split_dim_size  = *args.arg<int64_t>(narg++) ;
    const int64_t suffix_dim_size = *args.arg<int64_t>(narg++) ;

    const vml::Tensor *input_tensor = args.arg<vml::Tensor>(narg++) ;
    const uint64_t input_addr  = input_tensor->addr ;

    uint64_t output_addrs[num_split] ;
    for(int64_t i=0; i<num_split; i++) {
        const vml::Tensor *result = args.arg<vml::Tensor>(narg++) ;
        output_addrs[i] = result->addr ;
    }

    const int dtype = input_tensor->dtype ;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

    if (dtype == DT_FLOAT) {
        ret = split<float>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
                           input_addr, output_addrs) ;
    }
    else if (dtype == DT_DOUBLE) {
        ret = split<double>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
                            input_addr, output_addrs) ;
    }

    return ret;
}
}

DEFINE_KERNEL(Split, op_Split);


//
// SplitV
//
template<typename T>
int split_v(const int64_t  num_split,
            const int64_t  prefix_dim_size,
            const int64_t  split_dim_size,
            const int64_t  suffix_dim_size,
            const int64_t  *split_sizes,
            const uint64_t input_addr,
            uint64_t       *output_addrs)
{
    const T* pi = reinterpret_cast<const T*>(input_addr);
    T** po = reinterpret_cast<T**>(output_addrs);


    for(int64_t i=0; i<prefix_dim_size; i++) {
        int64_t offset = 0 ;
        for(int64_t n=0; n<num_split; n++) {
            const int64_t split_size = split_sizes[n] ;
            for(int64_t j=0; j<split_size*suffix_dim_size; j++) {
                po[n][i*split_size*suffix_dim_size+j]
                        = pi[(i*split_dim_size+offset)*suffix_dim_size+j] ;
            }
            offset += split_size ;
        }
    }
    return 0 ;
}

namespace {
int op_SplitV(const VEOpArgs& args)
{
    int ret=1;

    int narg = 0 ;
    const int64_t num_split       = *args.arg<int64_t>(narg++) ;
    const int64_t prefix_dim_size = *args.arg<int64_t>(narg++) ;
    const int64_t split_dim_size  = *args.arg<int64_t>(narg++) ;
    const int64_t suffix_dim_size = *args.arg<int64_t>(narg++) ;

    const vml::Tensor *input_tensor = args.arg<vml::Tensor>(narg++) ;
    const uint64_t input_addr  = input_tensor->addr ;

    uint64_t output_addrs[num_split] ;
    int64_t  split_sizes[num_split] ;
    for(int64_t i=0; i<num_split; i++) {
        const vml::Tensor *result = args.arg<vml::Tensor>(narg++) ;
        output_addrs[i] = result->addr ;
        split_sizes[i] = *args.arg<int64_t>(narg++) ;
    }

    const int dtype = input_tensor->dtype ;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

    if (dtype == DT_FLOAT) {
        ret = split_v<float>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
                             split_sizes, input_addr, output_addrs) ;
    }
    else if (dtype == DT_DOUBLE) {
        ret = split_v<double>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
                              split_sizes, input_addr, output_addrs) ;
    }

    return ret;
}
}

DEFINE_KERNEL(SplitV, op_SplitV);


//
// StridedSlice
//
#define STRIDED_SLICE_MAX_HANDLE_DIM 7

template<typename T>
int strided_slice1(const int64_t* begin_di,
                   const int64_t* end_di,
                   const int64_t* stride_di,
                   const vml::Tensor*  input_tensor,
                   vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(input_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = input_tensor->dim_size[0] ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        *po = pi[i0] ; po++ ;
    }

    return 0 ;
}


template<typename T>
int strided_slice2(const int64_t* begin_di,
                   const int64_t* end_di,
                   const int64_t* stride_di,
                   const vml::Tensor*  input_tensor,
                   vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(input_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = input_tensor->dim_size[0] ;
    const int64_t d1 = input_tensor->dim_size[1] ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
            *po = pi[i0*d1+i1] ; po++ ;
        }
    }

    return 0 ;
}

template<typename T>
int strided_slice3(const int64_t* begin_di,
                   const int64_t* end_di,
                   const int64_t* stride_di,
                   const vml::Tensor*  input_tensor,
                   vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(input_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = input_tensor->dim_size[0] ;
    const int64_t d1 = input_tensor->dim_size[1] ;
    const int64_t d2 = input_tensor->dim_size[2] ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
            for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
                *po = pi[(i0*d1+i1)*d2+i2] ; po++ ;
            }
        }
    }

    return 0 ;
}

template<typename T>
int strided_slice4(const int64_t* begin_di,
                   const int64_t* end_di,
                   const int64_t* stride_di,
                   const vml::Tensor*  input_tensor,
                   vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(input_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = input_tensor->dim_size[0] ;
    const int64_t d1 = input_tensor->dim_size[1] ;
    const int64_t d2 = input_tensor->dim_size[2] ;
    const int64_t d3 = input_tensor->dim_size[3] ;

    LOG(3) << __FUNCTION__ << ": d0=" << d0 << " d1=" << d1 << " d2=" << d2 << " d3=" << d3;

    LOG(3) << __FUNCTION__ << ": B " << begin_di[0] << " E " << end_di[0] << " S " << stride_di[0];
    LOG(3) << __FUNCTION__ << ": B " << begin_di[1] << " E " << end_di[1] << " S " << stride_di[1];
    LOG(3) << __FUNCTION__ << ": B " << begin_di[2] << " E " << end_di[2] << " S " << stride_di[2];
    LOG(3) << __FUNCTION__ << ": B " << begin_di[3] << " E " << end_di[3] << " S " << stride_di[3];

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
            for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
              for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
                *po = pi[((i0*d1+i1)*d2+i2)*d3+i3] ; po++ ;
              }
            }
        }
    }

    LOG(3) << __FUNCTION__ << ": done";

    return 0 ;
}

template<typename T>
int strided_slice5(const int64_t* begin_di,
                   const int64_t* end_di,
                   const int64_t* stride_di,
                   const vml::Tensor*  input_tensor,
                   vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(input_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = input_tensor->dim_size[0] ;
    const int64_t d1 = input_tensor->dim_size[1] ;
    const int64_t d2 = input_tensor->dim_size[2] ;
    const int64_t d3 = input_tensor->dim_size[3] ;
    const int64_t d4 = input_tensor->dim_size[4] ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
    for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
    for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
    for(int64_t i4=begin_di[4]; i4<end_di[4] ; i4+=stride_di[4] ) {
        *po = pi[(((i0*d1+i1)*d2+i2)*d3+i3)*d4+i4] ; po++ ;
    } } } } }

    return 0 ;
}

template<typename T>
int strided_slice6(const int64_t* begin_di,
                   const int64_t* end_di,
                   const int64_t* stride_di,
                   const vml::Tensor*  input_tensor,
                   vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(input_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = input_tensor->dim_size[0] ;
    const int64_t d1 = input_tensor->dim_size[1] ;
    const int64_t d2 = input_tensor->dim_size[2] ;
    const int64_t d3 = input_tensor->dim_size[3] ;
    const int64_t d4 = input_tensor->dim_size[4] ;
    const int64_t d5 = input_tensor->dim_size[5] ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
    for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
    for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
    for(int64_t i4=begin_di[4]; i4<end_di[4] ; i4+=stride_di[4] ) {
    for(int64_t i5=begin_di[5]; i5<end_di[5] ; i5+=stride_di[5] ) {
        *po = pi[((((i0*d1+i1)*d2+i2)*d3+i3)*d4+i4)*d5+i5] ; po++ ;
    } } } } } }

    return 0 ;
}

template<typename T>
int strided_slice7(const int64_t* begin_di,
                   const int64_t* end_di,
                   const int64_t* stride_di,
                   const vml::Tensor*  input_tensor,
                   vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(input_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = input_tensor->dim_size[0] ;
    const int64_t d1 = input_tensor->dim_size[1] ;
    const int64_t d2 = input_tensor->dim_size[2] ;
    const int64_t d3 = input_tensor->dim_size[3] ;
    const int64_t d4 = input_tensor->dim_size[4] ;
    const int64_t d5 = input_tensor->dim_size[5] ;
    const int64_t d6 = input_tensor->dim_size[6] ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
    for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
    for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
    for(int64_t i4=begin_di[4]; i4<end_di[4] ; i4+=stride_di[4] ) {
    for(int64_t i5=begin_di[5]; i5<end_di[5] ; i5+=stride_di[5] ) {
    for(int64_t i6=begin_di[6]; i6<end_di[6] ; i6+=stride_di[6] ) {
        *po = pi[(((((i0*d1+i1)*d2+i2)*d3+i3)*d4+i4)*d5+i5)*d6+i6] ; po++ ;
    } } } } } } }

    return 0 ;
}

namespace {
int op_StridedSlice(const VEOpArgs& args)
{
    int ret=1;

    int narg = 0 ;
    const int64_t processing_dims = *args.arg<int64_t>(narg++) ;

    const vml::Tensor *input_tensor  = args.arg<vml::Tensor>(narg++) ;
    const vml::Tensor *result_tensor = args.arg<vml::Tensor>(narg++) ;

    int64_t begin_di[STRIDED_SLICE_MAX_HANDLE_DIM] ;
    int64_t end_di[STRIDED_SLICE_MAX_HANDLE_DIM] ;
    int64_t stride_di[STRIDED_SLICE_MAX_HANDLE_DIM] ;

    for(int64_t i=0; i<processing_dims; i++) {
        begin_di[i] = *args.arg<int64_t>(narg++) ;
    }
    for(int64_t i=0; i<processing_dims; i++) {
        end_di[i] = *args.arg<int64_t>(narg++) ;
    }
    for(int64_t i=0; i<processing_dims; i++) {
        stride_di[i] = *args.arg<int64_t>(narg++) ;
    }

    const int dtype = input_tensor->dtype ;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype << " processing_dims=" << processing_dims;

    if (dtype == DT_FLOAT) {
        switch(processing_dims) {
        case 1:
            ret = strided_slice1<float>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 2:
            ret = strided_slice2<float>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 3 :
            ret = strided_slice3<float>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 4 :
            ret = strided_slice4<float>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 5 :
            ret = strided_slice5<float>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 6 :
            ret = strided_slice6<float>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 7 :
            ret = strided_slice7<float>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        default :
            break ;
        }
    }
    else if (dtype == DT_DOUBLE) {
        switch(processing_dims) {
        case 1:
            ret = strided_slice1<double>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 2:
            ret = strided_slice2<double>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 3 :
            ret = strided_slice3<double>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 4 :
            ret = strided_slice4<double>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 5 :
            ret = strided_slice5<double>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 6 :
            ret = strided_slice6<double>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 7 :
            ret = strided_slice7<double>(begin_di, end_di, stride_di, input_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        default :
            break ;
        }
    }

    return ret;
}
}
#undef STRIDED_SLICE_MAX_HANDLE_DIM

DEFINE_KERNEL(StridedSlice, op_StridedSlice);


//
// StridedSliceGrad
//
#define STRIDED_SLICE_GRAD_MAX_HANDLE_DIM 7

template<typename T>
int strided_slice_grad1(const int64_t* begin_di,
                        const int64_t* end_di,
                        const int64_t* stride_di,
                        const vml::Tensor*  dy_tensor,
                        vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = result_tensor->dim_size[0] ;

    for(int64_t i=0; i<d0; i++) po[i] = T(0.) ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        po[i0] = *pi ; pi++ ;
    }

    return 0 ;
}


template<typename T>
int strided_slice_grad2(const int64_t* begin_di,
                        const int64_t* end_di,
                        const int64_t* stride_di,
                        const vml::Tensor*  dy_tensor,
                        vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = result_tensor->dim_size[0] ;
    const int64_t d1 = result_tensor->dim_size[1] ;

    for(int64_t i=0; i<d0*d1; i++) po[i] = T(0.) ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
            po[i0*d1+i1] = *pi ; pi++ ;
        }
    }

    return 0 ;
}

template<typename T>
int strided_slice_grad3(const int64_t* begin_di,
                        const int64_t* end_di,
                        const int64_t* stride_di,
                        const vml::Tensor*  dy_tensor,
                        vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = result_tensor->dim_size[0] ;
    const int64_t d1 = result_tensor->dim_size[1] ;
    const int64_t d2 = result_tensor->dim_size[2] ;

    for(int64_t i=0; i<d0*d1*d2; i++) po[i] = T(0.) ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
            for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
                po[(i0*d1+i1)*d2+i2] = *pi ; pi++ ;
            }
        }
    }

    return 0 ;
}

template<typename T>
int strided_slice_grad4(const int64_t* begin_di,
                        const int64_t* end_di,
                        const int64_t* stride_di,
                        const vml::Tensor*  dy_tensor,
                        vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = result_tensor->dim_size[0] ;
    const int64_t d1 = result_tensor->dim_size[1] ;
    const int64_t d2 = result_tensor->dim_size[2] ;
    const int64_t d3 = result_tensor->dim_size[2] ;

    for(int64_t i=0; i<d0*d1*d2*d3; i++) po[i] = T(0.) ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
        for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
            for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
              for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
                po[((i0*d1+i1)*d2+i2)*d3+i3] = *pi ; pi++ ;
              }
            }
        }
    }

    return 0 ;
}

template<typename T>
int strided_slice_grad5(const int64_t* begin_di,
                        const int64_t* end_di,
                        const int64_t* stride_di,
                        const vml::Tensor*  dy_tensor,
                        vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = result_tensor->dim_size[0] ;
    const int64_t d1 = result_tensor->dim_size[1] ;
    const int64_t d2 = result_tensor->dim_size[2] ;
    const int64_t d3 = result_tensor->dim_size[3] ;
    const int64_t d4 = result_tensor->dim_size[4] ;

    for(int64_t i=0; i<d0*d1*d2*d3*d4; i++) po[i] = T(0.) ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
    for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
    for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
    for(int64_t i4=begin_di[4]; i4<end_di[4] ; i4+=stride_di[4] ) {
        po[(((i0*d1+i1)*d2+i2)*d3+i3)*d4+i4] = *pi ; pi++ ;
    } } } } }

    return 0 ;
}

template<typename T>
int strided_slice_grad6(const int64_t* begin_di,
                        const int64_t* end_di,
                        const int64_t* stride_di,
                        const vml::Tensor*  dy_tensor,
                        vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = result_tensor->dim_size[0] ;
    const int64_t d1 = result_tensor->dim_size[1] ;
    const int64_t d2 = result_tensor->dim_size[2] ;
    const int64_t d3 = result_tensor->dim_size[3] ;
    const int64_t d4 = result_tensor->dim_size[4] ;
    const int64_t d5 = result_tensor->dim_size[5] ;

    for(int64_t i=0; i<d0*d1*d2*d3*d4*d5; i++) po[i] = T(0.) ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
    for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
    for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
    for(int64_t i4=begin_di[4]; i4<end_di[4] ; i4+=stride_di[4] ) {
    for(int64_t i5=begin_di[5]; i5<end_di[5] ; i5+=stride_di[5] ) {
        po[((((i0*d1+i1)*d2+i2)*d3+i3)*d4+i4)*d5+i5] = *pi ; pi++ ;
    } } } } } }

    return 0 ;
}

template<typename T>
int strided_slice_grad7(const int64_t* begin_di,
                        const int64_t* end_di,
                        const int64_t* stride_di,
                        const vml::Tensor*  dy_tensor,
                        vml::Tensor* result_tensor)
{
    const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
    T* po = reinterpret_cast<T*>(result_tensor->addr);

    const int64_t d0 = result_tensor->dim_size[0] ;
    const int64_t d1 = result_tensor->dim_size[1] ;
    const int64_t d2 = result_tensor->dim_size[2] ;
    const int64_t d3 = result_tensor->dim_size[3] ;
    const int64_t d4 = result_tensor->dim_size[4] ;
    const int64_t d5 = result_tensor->dim_size[5] ;
    const int64_t d6 = result_tensor->dim_size[6] ;

    for(int64_t i=0; i<d0*d1*d2*d3*d4*d5*d6; i++) po[i] = T(0.) ;

    for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
    for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
    for(int64_t i3=begin_di[3]; i3<end_di[3] ; i3+=stride_di[3] ) {
    for(int64_t i4=begin_di[4]; i4<end_di[4] ; i4+=stride_di[4] ) {
    for(int64_t i5=begin_di[5]; i5<end_di[5] ; i5+=stride_di[5] ) {
    for(int64_t i6=begin_di[6]; i6<end_di[6] ; i6+=stride_di[6] ) {
        po[(((((i0*d1+i1)*d2+i2)*d3+i3)*d4+i4)*d5+i5)*d6+i6] = *pi ; pi++ ;
    } } } } } } }

    return 0 ;
}

namespace {
int op_StridedSliceGrad(const VEOpArgs& args)
{
    int ret=1;

    int narg = 0 ;
    const int64_t processing_dims = *args.arg<int64_t>(narg++) ;

    const vml::Tensor *dy_tensor     = args.arg<vml::Tensor>(narg++) ;
    const vml::Tensor *result_tensor = args.arg<vml::Tensor>(narg++) ;

    int64_t begin_di[STRIDED_SLICE_GRAD_MAX_HANDLE_DIM] ;
    int64_t end_di[STRIDED_SLICE_GRAD_MAX_HANDLE_DIM] ;
    int64_t stride_di[STRIDED_SLICE_GRAD_MAX_HANDLE_DIM] ;

    for(int64_t i=0; i<processing_dims; i++) {
        begin_di[i] = *args.arg<int64_t>(narg++) ;
    }
    for(int64_t i=0; i<processing_dims; i++) {
        end_di[i] = *args.arg<int64_t>(narg++) ;
    }
    for(int64_t i=0; i<processing_dims; i++) {
        stride_di[i] = *args.arg<int64_t>(narg++) ;
    }

    const int dtype = dy_tensor->dtype ;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype << " processing_dims=" << processing_dims;

    if (dtype == DT_FLOAT) {
        switch(processing_dims) {
        case 1:
            ret = strided_slice_grad1<float>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 2:
            ret = strided_slice_grad2<float>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 3 :
            ret = strided_slice_grad3<float>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 4 :
            ret = strided_slice_grad4<float>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 5 :
            ret = strided_slice_grad5<float>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 6 :
            ret = strided_slice_grad6<float>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 7 :
            ret = strided_slice_grad7<float>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        default :
            break ;
        }
    }
    else if (dtype == DT_DOUBLE) {
        switch(processing_dims) {
        case 1:
            ret = strided_slice_grad1<double>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 2:
            ret = strided_slice_grad2<double>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 3 :
            ret = strided_slice_grad3<double>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 4 :
            ret = strided_slice_grad4<double>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 5 :
            ret = strided_slice_grad5<double>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 6 :
            ret = strided_slice_grad6<double>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        case 7 :
            ret = strided_slice_grad7<double>(begin_di, end_di, stride_di, dy_tensor, (vml::Tensor*)result_tensor) ;
            break ;
        default :
            break ;
        }
    }

    return ret;
}
}
#undef STRIDED_SLICE_GRAD_MAX_HANDLE_DIM

DEFINE_KERNEL(StridedSliceGrad, op_StridedSliceGrad);


//
// L2Loss
//
template<typename T>
int l2loss(const int64_t  input_length,
           const uint64_t input_addr,
           const uint64_t output_addr)
{
    const T* pi = reinterpret_cast<const T*>(input_addr);
    T* po = reinterpret_cast<T*>(output_addr);

    T sum = T(0.) ;
    for(int64_t i=0; i<input_length; i++) {
        const T v = pi[i] ;
        sum += v*v ;
    }
    po[0] = sum * T(0.5) ;

    return 0 ;
}

namespace {
int op_L2Loss(const VEOpArgs& args)
{
    if (args.nArguments() != 2)
        return 1 ;

    int ret=1;

    const vml::Tensor *input_tensor  = args.arg<vml::Tensor>(0) ;
    const vml::Tensor *output_tensor = args.arg<vml::Tensor>(1) ;

    const int dtype = input_tensor->dtype ;

    const int64_t input_length = input_tensor->nelems ;

    const int64_t input_addr  = input_tensor->addr  ;
    const int64_t output_addr = output_tensor->addr ;

    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << dtype;

    if (dtype == DT_FLOAT) {
        ret = l2loss<float>(input_length, input_addr, output_addr) ;
    }
    else if (dtype == DT_DOUBLE) {
        ret = l2loss<double>(input_length, input_addr, output_addr) ;
    }

    return ret;
}
}

DEFINE_KERNEL(L2Loss, op_L2Loss);
