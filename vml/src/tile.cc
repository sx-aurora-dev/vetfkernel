#include <vml.h>

#include "types.h"
#include "log.h"

#include "intrinsic/intrinsic.h"

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

} // namespace

int vml::tile(vml::Tensor const& X, vml::Tensor const& Y)
{
    LOG(LOG_PARAM) << __FUNCTION__ << " Y=" << Y << " X=" << X;

    int rc = 1 ;

    if (Y.dtype == DT_FLOAT && X.dtype == DT_FLOAT) {
        const float* pi = reinterpret_cast<const float*>(Y.addr);
        float* po = reinterpret_cast<float*>(X.addr);
        if (Y.nelems == 1) {
            for (size_t i = 0; i < X.nelems; ++i) {
                po[i] = pi[0];
            }
            rc = 0 ;
        } else if (Y.dims == 2 && X.dims == 2
                   && Y.dim_size[0] == X.dim_size[0]
                   && Y.dim_size[1] == 1) {
            for (size_t i = 0; i < Y.dim_size[0]; ++i) {
                for (size_t j = 0; j < X.dim_size[1]; ++j) {
                    po[i * X.dim_size[1] + j] = pi[i];
                }
            }
            rc = 0 ;
        } else if (Y.dims == 2 && X.dims == 2
                   && Y.dim_size[1] == X.dim_size[1]
                   && Y.dim_size[0] == 1) {
            for (size_t i = 0; i < X.dim_size[0]; ++i) {
                for (size_t j = 0; j < X.dim_size[1]; ++j) {
                    po[i * X.dim_size[1] + j] = pi[j];
                }
            }
            rc = 0 ;
        } else if ( Y.dims == X.dims && Y.dims == 3 ) {
            rc = tile_dim3<float>(X, Y) ;
        } else if ( Y.dims == X.dims && Y.dims == 4 ) {
            if( Y.dim_size[2]==1 && Y.dim_size[3]==1 ) {
                rc = tile_dim4_11<float>(X, Y) ;
            }
            else {
                rc = tile_dim4<float>(X, Y) ;
            }
        } else if ( Y.dims == X.dims && Y.dims == 5 ) {
            if( Y.dim_size[3]==1 && Y.dim_size[4]==1 ) {
                rc = tile_dim5_11<float>(X, Y) ;
            }
            else {
                rc = tile_dim5<float>(X, Y) ;
            }
        }
    }

    return rc;
}

