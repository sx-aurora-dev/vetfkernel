#include <stdint.h>
#include <float.h>

#include <stdio.h>




#include "libvetfkernel.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))



int mean_d3a02_f32(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    float* po = (float *)(out);
    const float* pi = (const float *)(in);

    size_t dim12 = dim1 * dim2;
    float idim02 = 1.0f/(dim0*dim2);
    //    printf("mean ve d3a02, %d %d %d\n",dim0,dim1,dim2);

#if 1
#if 1
    if((dim2>VLEN*1.5)&&((dim2%2)==0)){
        __vr vr_idim02 = _vel_vbrds_vsl(idim02, VLEN);
        for (size_t j = 0; j < dim1; ++j) {
            __vr vr_sum = _vel_vbrds_vsl(0.f, VLEN);
            for (size_t k = 0; k < dim2; k+=VLEN*2) {
                const int64_t vlen = (dim2-k < VLEN*2 ? dim2-k : VLEN*2) >> 1;
                for (size_t i = 0; i < dim0; ++i) {
                    __vr vr_vl = _vel_vld_vssl(8,pi + i * dim12 + j * dim2 +k,vlen);
                    vr_sum = _vel_pvfadd_vvvvl(vr_sum, vr_vl, vr_sum, vlen);
                }
            }
            vr_sum = _vel_vfadds_vvvl(vr_sum,
                                      _vel_vsll_vvsl(vr_sum, 32, VLEN),
                                      VLEN);
            vr_sum = _vel_vfsums_vvl(vr_sum,VLEN);
            vr_sum = _vel_vfmuls_vvvvl(vr_sum, vr_idim02, vr_sum, 1);
            _vel_vstu_vssl(vr_sum, 4, po + j, 1);
        }
    }else{
        __vr vr_idim02 = _vel_vbrds_vsl(idim02, VLEN);
        for (size_t j = 0; j < dim1; ++j) {
            __vr vr_sum = _vel_vbrds_vsl(0.f, VLEN);
            for (size_t k = 0; k < dim2; k+=VLEN) {
                const int64_t vlen = (dim2-k < VLEN ? dim2-k : VLEN);
                for (size_t i = 0; i < dim0; ++i) {
                    __vr vr_vl = _vel_vldu_vssl(4,pi + i * dim12 + j * dim2 +k,vlen);
                    vr_sum = _vel_vfadds_vvvvl(vr_sum, vr_vl, vr_sum, vlen);
                }
            }
            vr_sum = _vel_vfsums_vvl(vr_sum,VLEN);
            vr_sum = _vel_vfmuls_vvvvl(vr_sum, vr_idim02, vr_sum, 1);
            _vel_vstu_vssl(vr_sum, 4, po + j, 1);
        }
    }
#else
    for (size_t j = 0; j < dim1; ++j) {
        __vr vr_sum = _vel_vbrds_vsl(0.f, VLEN);
        for (size_t k = 0; k < dim2; k+=VLEN) {
            const int64_t vlen = (dim2-k < VLEN ? dim2-k : VLEN);
            for (size_t i = 0; i < dim0; ++i) {
                __vr vr_vl = _vel_vldu_vssl(4,pi + i * dim12 + j * dim2 +k,vlen);
                vr_sum = _vel_vfadds_vvvvl(vr_sum, vr_vl, vr_sum, vlen);
            }
        }
        vr_sum = _vel_vfsums_vvl(vr_sum,VLEN);
        _vel_vstu_vssl(vr_sum, 4, po + j, 1);
    }
    __vr vr_idim02 = _vel_vbrds_vsl(idim02, VLEN);
    for (size_t j = 0; j < dim1; j+=VLEN) {
        const int64_t vlen = (dim1-j < VLEN ? dim1-j : VLEN);
        __vr vr_vl = _vel_vldu_vssl(4,po + j,vlen);
        vr_vl = _vel_vfmuls_vvvvl(vr_vl, vr_idim02, vr_vl, vlen);
        _vel_vstu_vssl(vr_vl, 4, po + j, vlen);
    }
#endif

#else
    for (size_t j = 0; j < dim1; ++j) {
        float s = (float)(0);
        for (size_t i = 0; i < dim0; ++i) {
            for (size_t k = 0; k < dim2; ++k) {
                s += pi[i * dim12 + j * dim2 + k];
            }
        }
        po[j] = s / (dim0*dim2);
    }
#endif
    return 0;
}



int mean_d2a0_f32(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
    float* po = (float *)(out);
    const float* pi = (const float *)(in);

    //    printf("mean ve d2a0, %d %d\n",dim0,dim1);

#if 1
    float idim0 = 1.0f/(dim0);
    __vr vr_idim0 = _vel_vbrds_vsl(idim0, VLEN);
    for (size_t j = 0; j < dim1; j+=VLEN) {
        const int64_t vlen = (dim1-j < VLEN ? dim1-j : VLEN);
        __vr vr_sum = _vel_vbrds_vsl(0.f, vlen);
        for (size_t i = 0; i < dim0; ++i) {
            __vr vr_vl = _vel_vldu_vssl(4,pi + i * dim1 + j,vlen);
            vr_sum = _vel_vfadds_vvvvl(vr_sum, vr_vl, vr_sum, vlen);
        }
        vr_sum = _vel_vfmuls_vvvvl(vr_sum, vr_idim0, vr_sum, vlen);
        _vel_vstu_vssl(vr_sum, 4, po + j, vlen);
    }


#else
    for (size_t j = 0; j < dim1; ++j) {
        T s = T(0);
        for (size_t i = 0; i < dim0; ++i) {
            s += pi[i * dim1 + j];
        }
        po[j] = s / dim0 ;
    }
#endif

    return 0;
}






