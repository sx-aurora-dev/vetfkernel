#include <stdint.h>
#include <float.h>

#include <stdio.h>




#include "libvetfkernel.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

#if 0
int add_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
    float* po = (float*)(out);
    const float* pi0 = (const float*)(in0);
    float i1 = *((const float*)(in1));



    const uint64_t alignIn = ((const uint64_t)in0) & 0x07;
    const uint64_t alignOut = ((const uint64_t)out) & 0x07;


    if((alignIn==0)&&(alignOut==0)&&(n%2==0)&&(n>256)){
        unsigned long int li1 = _ve_pack_f32a(&i1);
        if(n%(8*VLEN)==0){
            _ve_lvl(VLEN);
            for (size_t i = 0; i < n; i+=8*VLEN) {
                __vr vr_pin1 = _ve_vld_vss(8,pi0+i+2*VLEN*0);
                __vr vr_pin2 = _ve_vld_vss(8,pi0+i+2*VLEN*1);
                __vr vr_pin3 = _ve_vld_vss(8,pi0+i+2*VLEN*2);
                __vr vr_pin4 = _ve_vld_vss(8,pi0+i+2*VLEN*3);
                __vr vr_sum1 = _ve_pvfadd_vsv(li1, vr_pin1);
                __vr vr_sum2 = _ve_pvfadd_vsv(li1, vr_pin2);
                __vr vr_sum3 = _ve_pvfadd_vsv(li1, vr_pin3);
                __vr vr_sum4 = _ve_pvfadd_vsv(li1, vr_pin4);
                _ve_vst_vss(vr_sum1,8,po+i+2*VLEN*0);
                _ve_vst_vss(vr_sum2,8,po+i+2*VLEN*1);
                _ve_vst_vss(vr_sum3,8,po+i+2*VLEN*2);
                _ve_vst_vss(vr_sum4,8,po+i+2*VLEN*3);
            }
        }else if(n%8==0){
            for (size_t i = 0; i < n; i+=8*VLEN) {
                const int64_t vlen = (n-i < 8*VLEN ? n-i : 8*VLEN) >> 3;
                _ve_lvl(vlen);
                __vr vr_pin1 = _ve_vld_vss(8,pi0+i+2*vlen*0);
                __vr vr_pin2 = _ve_vld_vss(8,pi0+i+2*vlen*1);
                __vr vr_pin3 = _ve_vld_vss(8,pi0+i+2*vlen*2);
                __vr vr_pin4 = _ve_vld_vss(8,pi0+i+2*vlen*3);
                __vr vr_sum1 = _ve_pvfadd_vsv(li1, vr_pin1);
                __vr vr_sum2 = _ve_pvfadd_vsv(li1, vr_pin2);
                __vr vr_sum3 = _ve_pvfadd_vsv(li1, vr_pin3);
                __vr vr_sum4 = _ve_pvfadd_vsv(li1, vr_pin4);
                _ve_vst_vss(vr_sum1,8,po+i+2*vlen*0);
                _ve_vst_vss(vr_sum2,8,po+i+2*vlen*1);
                _ve_vst_vss(vr_sum3,8,po+i+2*vlen*2);
                _ve_vst_vss(vr_sum4,8,po+i+2*vlen*3);
            }
        }else{
            for (size_t i = 0; i < n; i+=2*VLEN) {
                const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
                _ve_lvl(vlen);
                __vr vr_pin = _ve_vld_vss(8,pi0+i);
                __vr vr_sum = _ve_pvfadd_vsv(li1, vr_pin);
                _ve_vst_vss(vr_sum,8,po+i);
            }
        }
    }else if(n>256){
        for (size_t i = 0; i < n; i+=VLEN) {
            const int64_t vlen = n-i < VLEN ? n-i : VLEN;
            _ve_lvl(vlen);
            __vr vr_pin = _ve_vldu_vss(4,pi0+i);
            __vr vr_sum = _ve_vfadds_vsv(i1, vr_pin);
            _ve_vstu_vss(vr_sum,4,po+i);
        }
    }else if(n<17){
        for (size_t i = 0; i < n; i++)
            po[i] = pi0[i] + i1;

    }else{
        _ve_lvl(n);
        __vr vr_pin = _ve_vldu_vss(4,pi0);
        __vr vr_sum = _ve_vfadds_vsv(i1, vr_pin);
        _ve_vstu_vss(vr_sum,4,po);
    }
    return 0;
}
#endif

int mean_d3a02_f32(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2)
{
    float* po = (float *)(out);
    const float* pi = (const float *)(in);

    size_t dim12 = dim1 * dim2;
    float idim02 = 1.0f/(dim0*dim2);
//    printf("mean ve d3a02, %d %d %d\n",dim0,dim1,dim2);

#if 1
    for (size_t j = 0; j < dim1; ++j) {
        __vr vr_sum = _vel_vbrds_vsl(0.f, VLEN);
        for (size_t k = 0; k < dim2; k+=VLEN) {
            const int64_t vlen = (dim2-k < VLEN ? dim2-k : VLEN);
            for (size_t i = 0; i < dim0; ++i) {
                __vr vr_vl = _vel_vldu_vssl(4,pi + i * dim12 + j * dim2,vlen);
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

