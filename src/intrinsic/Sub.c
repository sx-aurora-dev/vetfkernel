#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "intrinsic.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


int sub_nn_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	const float* pi1 = (const float*)(in1);

	const uint64_t alignIn0 = ((const uint64_t)in0) & 0x07;
	const uint64_t alignIn1 = ((const uint64_t)in1) & 0x07;
	const uint64_t alignOut = ((const uint64_t)out) & 0x07;


	if((alignIn0==0)&&(alignIn1==0)&&(alignOut==0)&&(n%2==0)){
		for (size_t i = 0; i < n; i+=2*VLEN) {
			const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
			 ;
			__vr vr_pin0 = _vel_vld_vssl(8,pi0+i, vlen);
			__vr vr_pin1 = _vel_vld_vssl(8,pi1+i, vlen);
			__vr vr_sub = _vel_pvfsub_vvvl(vr_pin0, vr_pin1, vlen);
			_vel_vst_vssl(vr_sub,8,po+i, vlen);
		}
	}else if(n>7){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			 ;
			__vr vr_pin0 = _vel_vldu_vssl(4,pi0+i, vlen);
			__vr vr_pin1 = _vel_vldu_vssl(4,pi1+i, vlen);
			__vr vr_sub = _vel_vfsubs_vvvl(vr_pin0, vr_pin1, vlen);
			_vel_vstu_vssl(vr_sub,4,po+i, vlen);
		}
	}else{
		for (size_t i = 0; i < n; i++) {
			po[i] = pi0[i] - pi1[i];
		}
	}	



	return 0;
}

