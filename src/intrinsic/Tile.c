#include <stdint.h>
#include <float.h>

#include <stdio.h>




#include "libvetfkernel.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

int tile_dim5_11_f32(float* px, float const*py, int64_t const* sx, int64_t const* sy)
{
    //    printf("tile5_11: x %d %d %d %d %d\n",sx[0],sx[1],sx[2],sx[3],sx[4],sx[5]);
    //    printf("tile5_11: y %d %d %d %d %d\n",sy[0],sy[1],sy[2],sy[3],sy[4],sy[5]);


#if 1

#if 0
#pragma _NEC novector
    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        const size_t ix0 = i0 ;
        const size_t iy0 = i0 % sy[0] ;
        size_t sx12 = sx[1] * sx[2];
        size_t sx34 = sx[3] * sx[4];
        for (size_t i12 = 0; i12 < sx12; ++i12) {

            const size_t i1 = i12/sx[2];
            const size_t i2 = i12%sx[2];
            const size_t iy1 = iy0 * sy[1] + (i1 % sy[1]) ;
            const size_t iy2 = iy1 * sy[2] + (i2 % sy[2]) ;

            __vr vr_iy2 = _vel_vbrds_vsl(py[iy2] , VLEN);
            for (size_t i34 = 0; i34 < sx34 ; i34+=VLEN) {
                const int64_t vlen = (sx34-i34 < VLEN ? sx34-i34 : VLEN);
                _vel_vstu_vssl(vr_iy2, 4, px + (ix0 * sx12 + i12 ) * sx34 + i34, vlen);
            }


        }

    }

#elif 0
#pragma _NEC novector
    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        const size_t ix0 = i0 ;
        const size_t iy0 = i0 % sy[0] ;
        for (size_t i12 = 0; i12 < sx[1]*sx[2]; ++i12) {

            const size_t i1 = i12/sx[2];
            const size_t i2 = i12%sx[2];


            const size_t iy1 = iy0 * sy[1] + (i1 % sy[1]) ;
            const size_t ix2 = ix0 * sx[1] *sx[2] + i12 ;
            const size_t iy2 = iy1 * sy[2] + (i2 % sy[2]) ;

            __vr vr_iy2 = _vel_vbrds_vsl(py[iy2] , VLEN);
            size_t sx34 = sx[3] * sx[4];
            for (size_t i34 = 0; i34 < sx34 ; i34+=VLEN) {
                const int64_t vlen = (sx34-i34 < VLEN ? sx34-i34 : VLEN);
                _vel_vstu_vssl(vr_iy2, 4, px + ix2 * sx34 + i34, vlen);
            }


        }

    }

#elif 1
    size_t sx0 = sx[0];
    size_t sx1 = sx[1];
    size_t sx2 = sx[2];
    size_t sx3 = sx[3];
    size_t sx4 = sx[4];
    size_t sy0 = sy[0];
    size_t sy1 = sy[1];
    size_t sy2 = sy[2];
    size_t sy3 = sy[3];
    size_t sy4 = sy[4];
    if((sx1==sy1)&&(sx2==sy2)){
        size_t sx34 = sx3 * sx4;
        if(sx34<=VLEN){

            if(sx2%4==0){
#pragma _NEC novector
                for (size_t i0 = 0; i0 < sx0; ++i0) {
                    const size_t ix0 = i0 ;
                    const size_t iy0 = i0 % sy0;
                    const int64_t vlen = sx34;
#pragma _NEC novector
                    for (size_t i1 = 0; i1 < sx1; ++i1) {
                        const size_t ix1 = ix0 * sx1 + i1 ;
                        const size_t iy1 = iy0 * sy1 + i1;
#pragma _NEC novector
                        for (size_t i2 = 0; i2 < (sx2>>2); ++i2) {
                            const size_t ix2a = ix1 * sx2 + 4*i2 ;
                            const size_t ix2b = ix1 * sx2 + 4*i2+1;
                            const size_t ix2c = ix1 * sx2 + 4*i2+2;
                            const size_t ix2d = ix1 * sx2 + 4*i2+3;
                            const size_t iy2a = iy1 * sy2 + 4*i2 ;
                            const size_t iy2c = iy1 * sy2 + 4*i2+2;

                            const long *lpya = &py[iy2a];
                            const long *lpyc = &py[iy2c];
                            __vr vr_iy2a = _vel_vbrdl_vsl(lpya[0] , vlen);
                            __vr vr_iy2c = _vel_vbrdl_vsl(lpyc[0] , vlen);


                            _vel_vstl_vssl(vr_iy2a, 4, px + ix2a * sx34, vlen);
                            _vel_vstu_vssl(vr_iy2a, 4, px + ix2b * sx34, vlen);
                            _vel_vstl_vssl(vr_iy2c, 4, px + ix2c * sx34, vlen);
                            _vel_vstu_vssl(vr_iy2c, 4, px + ix2d * sx34, vlen);

                        }
                    }
                }
                return 0;
            }
            if(sx2%2==0){
#pragma _NEC novector
                for (size_t i0 = 0; i0 < sx0; ++i0) {
                    const size_t ix0 = i0 ;
                    const size_t iy0 = i0 % sy0;
#pragma _NEC novector
                    for (size_t i1 = 0; i1 < sx1; ++i1) {
                        const size_t ix1 = ix0 * sx1 + i1 ;
                        const size_t iy1 = iy0 * sy1 + i1;
#pragma _NEC novector
                        for (size_t i2 = 0; i2 < (sx2>>1); ++i2) {
                            const size_t ix2a = ix1 * sx2 + 2*i2 ;
                            const size_t ix2b = ix1 * sx2 + 2*i2+1;
                            const size_t iy2 = iy1 * sy2 + 2*i2 ;
                            const int64_t vlen = sx34;
                            const long *lpy = &py[iy2];
                            __vr vr_iy2 = _vel_vbrdl_vsl(lpy[0] , vlen);

                            _vel_vstl_vssl(vr_iy2, 4, px + ix2a * sx34, vlen);
                            _vel_vstu_vssl(vr_iy2, 4, px + ix2b * sx34, vlen);

                        }
                    }
                }
                return 0;
            }


#pragma _NEC novector
            for (size_t i0 = 0; i0 < sx0; ++i0) {
                const size_t ix0 = i0 ;
                const size_t iy0 = i0 % sy0;
#pragma _NEC novector
                for (size_t i1 = 0; i1 < sx1; ++i1) {
                    const size_t ix1 = ix0 * sx1 + i1 ;
                    const size_t iy1 = iy0 * sy1 + i1;
#pragma _NEC novector
                    for (size_t i2 = 0; i2 < sx2; ++i2) {
                        const size_t ix2 = ix1 * sx2 + i2 ;
                        const size_t iy2 = iy1 * sy2 + i2 ;
                        const int64_t vlen = sx34;
                        __vr vr_iy2 = _vel_vbrds_vsl(py[iy2] , vlen);


                        _vel_vstu_vssl(vr_iy2, 4, px + ix2 * sx34, vlen);

                    }
                }
            }
            return 0;
        }

        if(sx2%4==0){
#pragma _NEC novector
            for (size_t i0 = 0; i0 < sx0; ++i0) {
                const size_t ix0 = i0 ;
                const size_t iy0 = i0 % sy0 ;
#pragma _NEC novector
                for (size_t i1 = 0; i1 < sx1; ++i1) {
                    const size_t ix1 = ix0 * sx1 + i1 ;
                    const size_t iy1 = iy0 * sy1 + i1 ;
#pragma _NEC novector
                    for (size_t i2 = 0; i2 < (sx2>>2); ++i2) {
                        const size_t ix2a = ix1 * sx2 + 4*i2 ;
                        const size_t ix2b = ix1 * sx2 + 4*i2 +1;
                        const size_t ix2c = ix1 * sx2 + 4*i2 +2;
                        const size_t ix2d = ix1 * sx2 + 4*i2 +3;
                        const size_t iy2a = iy1 * sy2 + 4*i2;
                        const size_t iy2c= iy1 * sy2 + 4*i2+2;
                        const long* lpya = &py[iy2a];
                        const long* lpyc = &py[iy2c];
                        __vr vr_iy2a = _vel_vbrdl_vsl(lpya[0] , VLEN);
                        __vr vr_iy2c = _vel_vbrdl_vsl(lpyc[0] , VLEN);

                        for (size_t i34 = 0; i34 < sx34 ; i34+=VLEN) {
                            const int64_t vlen = (sx34-i34 < VLEN ? sx34-i34 : VLEN);
                            _vel_vstl_vssl(vr_iy2a, 4, px + ix2a * sx34 + i34, vlen);
                            _vel_vstu_vssl(vr_iy2a, 4, px + ix2b * sx34 + i34, vlen);
                            _vel_vstl_vssl(vr_iy2c, 4, px + ix2c * sx34 + i34, vlen);
                            _vel_vstu_vssl(vr_iy2c, 4, px + ix2d * sx34 + i34, vlen);
                        }
                    }
                }
            }
            return 0;
        }

#pragma _NEC novector
        for (size_t i0 = 0; i0 < sx0; ++i0) {
            const size_t ix0 = i0 ;
            const size_t iy0 = i0 % sy0 ;
#pragma _NEC novector
            for (size_t i1 = 0; i1 < sx1; ++i1) {
                const size_t ix1 = ix0 * sx1 + i1 ;
                const size_t iy1 = iy0 * sy1 + i1 ;
#pragma _NEC novector
                for (size_t i2 = 0; i2 < sx2; ++i2) {
                    const size_t ix2 = ix1 * sx2 + i2 ;
                    const size_t iy2 = iy1 * sy2 + i2;

                    __vr vr_iy2 = _vel_vbrds_vsl(py[iy2] , VLEN);

                    for (size_t i34 = 0; i34 < sx34 ; i34+=VLEN) {
                        const int64_t vlen = (sx34-i34 < VLEN ? sx34-i34 : VLEN);
                        _vel_vstu_vssl(vr_iy2, 4, px + ix2 * sx34 + i34, vlen);
                    }
                }
            }
        }
        return 0;
    }



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

                __vr vr_iy2 = _vel_vbrds_vsl(py[iy2] , VLEN);
                size_t sx34 = sx[3] * sx[4];
                for (size_t i34 = 0; i34 < sx34 ; i34+=VLEN) {
                    const int64_t vlen = (sx34-i34 < VLEN ? sx34-i34 : VLEN);
                    _vel_vstu_vssl(vr_iy2, 4, px + ix2 * sx34 + i34, vlen);
                }


            }
        }
    }
    return 0;
#endif

#else
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
#endif
}









