#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "intrinsic.h"

#include "velintrin.h"
#define VLEN	(256)


static inline void _apply_adam_f32_packed(
  float* var, float* m, float *v,
  const float beta1, const float beta2, 
  const float epsilon, const float k,
  const int64_t numElement, 
  const float *grd )
{
  const float one_minus_beta1 = 1.f - beta1 ;
  const float one_minus_beta2 = 1.f - beta2 ;
  const float minus_k         = -k ;

  const uint64_t one_minus_beta1_packed = _vel_pack_f32a(&one_minus_beta1) ;
  const uint64_t one_minus_beta2_packed = _vel_pack_f32a(&one_minus_beta2) ;
  const uint64_t minus_k_packed         = _vel_pack_f32a(&minus_k) ;
  const uint64_t epsilon_packed         = _vel_pack_f32a(&epsilon) ;

  const uint64_t alignVar = ((const uint64_t)var) & 0x07;
  if( alignVar ) {
    m[0] = m[0] + one_minus_beta1 * (grd[0] - m[0]) ;
    v[0] = v[0] + one_minus_beta2 * (grd[0]*grd[0] - v[0]) ;
    var[0] -= k * m[0] / (epsilon + sqrtf(v[0])) ;    
  }

  const int64_t j = alignVar ? 1 : 0 ;
  const int64_t halfElement = (numElement - j) >> 1 ;  
  for(int64_t i=0; i<halfElement; i+=VLEN) {
    const int64_t vl = halfElement - i < VLEN ? halfElement - i : VLEN ;

     ;

    __vr vrm   = _vel_vld_vssl(8, m+2*i+j, vl) ;
    __vr vrv   = _vel_vld_vssl(8, v+2*i+j, vl) ;
    __vr vrgrd = _vel_vld_vssl(8, grd+2*i+j, vl) ;
    __vr vrvar = _vel_vld_vssl(8, var+2*i+j, vl) ;

    vrm = _vel_pvfmad_vvsvl(vrm,
                          one_minus_beta1_packed,
                          _vel_pvfsub_vvvl(vrgrd, vrm, vl), vl) ;
    vrv = _vel_pvfmad_vvsvl(vrv,
                          one_minus_beta2_packed,
                          _vel_pvfmsb_vvvvl(vrv, vrgrd,vrgrd, vl), vl) ;

    __vr sqrt_vrv = _vel_vshf_vvvsl(_vel_vfsqrts_vvl(vrv, vl),
                                  _vel_vfsqrts_vvl(_vel_vsll_vvsl(vrv,32, vl), vl) ,
                                  VE_VSHUFFLE_YUZU , vl) ;

    vrvar = _vel_pvfmad_vvsvl(vrvar,
                            minus_k_packed,
                            _vel_approx_pvfdiv_vvvl(vrm,
                                            _vel_pvfadd_vsvl(epsilon_packed,
                                                           sqrt_vrv, vl), vl), vl) ;
   
    _vel_vst_vssl(vrm, 8, m+2*i+j, vl) ; 
    _vel_vst_vssl(vrv, 8, v+2*i+j, vl) ; 
    _vel_vst_vssl(vrvar, 8, var+2*i+j, vl) ; 
  }
  
  if( ( !alignVar && (numElement & 0x01)==1 )
      || ( alignVar && (numElement & 0x01)==0 ) ) {
    const int64_t idx = numElement - 1 ; 
    m[idx] = m[idx] + one_minus_beta1 * (grd[idx] - m[idx]) ;
    v[idx] = v[idx] + one_minus_beta2 * (grd[idx]*grd[idx] - v[idx]) ;
    var[idx] -= k * m[idx] / (epsilon + sqrtf(v[idx])) ;    
  }
}

static inline void _apply_adam_f32_defualt(
  float* var, float* m, float *v,
  const float beta1, const float beta2, 
  const float epsilon, const float k,
  const int64_t numElement, 
  const float *grd )
{
  for(int64_t i=0; i<numElement; i+=VLEN) {
    const int64_t vl = numElement - i < VLEN ? numElement - i : VLEN ;

     ;

    __vr vrm   = _vel_vldu_vssl(4, m+i, vl) ;
    __vr vrv   = _vel_vldu_vssl(4, v+i, vl) ;
    __vr vrgrd = _vel_vldu_vssl(4, grd+i, vl) ;
    __vr vrvar = _vel_vldu_vssl(4, var+i, vl) ;

    vrm = _vel_vfmads_vvsvl(vrm,
                          1.f - beta1,
                          _vel_vfsubs_vvvl(vrgrd, vrm, vl), vl) ;
    vrv = _vel_vfmads_vvsvl(vrv,
                          1.f - beta2,
                          _vel_vfmsbs_vvvvl(vrv, vrgrd,vrgrd, vl), vl) ;
    vrvar = _vel_vfmads_vvsvl(vrvar,
                            -k,
                            _vel_approx_vfdivs_vvvl(vrm,
                                            _vel_vfadds_vsvl(epsilon,
                                                           _vel_vfsqrts_vvl(vrv, vl), vl), vl), vl) ;
   
    _vel_vstu_vssl(vrm, 4, m+i, vl) ; 
    _vel_vstu_vssl(vrv, 4, v+i, vl) ; 
    _vel_vstu_vssl(vrvar, 4, var+i, vl) ; 

  }
}

void _apply_adam_f32(float* var, float* m, float *v,
                     const float beta1, const float beta2, 
                     const float epsilon, const float k,
                     const int64_t numElement, 
                     const float *grd ) 
{
  const uint64_t alignVar = ((const uint64_t)var) & 0x07;
  const uint64_t alignM   = ((const uint64_t)m  ) & 0x07;
  const uint64_t alignV   = ((const uint64_t)v  ) & 0x07;
  const uint64_t alignGrd = ((const uint64_t)grd) & 0x07;
  
  if ( (numElement >= 2*VLEN)
       && (alignVar == alignM) 
       && (alignVar == alignV)
       && (alignVar == alignGrd) )
  {
     _apply_adam_f32_packed(var, m, v, beta1, beta2, epsilon, k, 
                            numElement, grd) ;

  }
  else {
     _apply_adam_f32_defualt(var, m, v, beta1, beta2, epsilon, k, 
                             numElement, grd) ;
  }
}
