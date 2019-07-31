#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


int neg(uint64_t out, uint64_t in, size_t n)
{

	const float* pi = (const float*)(in);
	float* po = (float*)(out);


	if(VLEN<n){
		const uint64_t alignIn = ((const uint64_t)in) & 0x07;
		const uint64_t alignOut = ((const uint64_t)out) & 0x07;


		if((alignIn==0)&&(alignOut==0)&&(n%2==0)){
			for (size_t i = 0; i < n; i+=2*VLEN) {
				const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
				;
				__vr vr_pin = _vel_vld_vssl(8,pi+i, vlen);
				__vr vr_sub = _vel_pvfsub_vsvl(0UL, vr_pin, vlen);
				_vel_vst_vssl(vr_sub,8,po+i, vlen);
			}
		}else{
			for (size_t i = 0; i < n; i+=VLEN) {
				const int64_t vlen = n-i < VLEN ? n-i : VLEN;
				;
				__vr vr_pin = _vel_vldu_vssl(4,pi+i, vlen);
				__vr vr_sub = _vel_vfsubs_vsvl(0.f, vr_pin, vlen);
				_vel_vstu_vssl(vr_sub,4,po+i, vlen);
			}
		}
	}else if(n<=VLEN){
		;
		__vr vr_pin = _vel_vldu_vssl(4,pi, n);
		__vr vr_sub = _vel_vfsubs_vsvl(0.f, vr_pin, n);
		_vel_vstu_vssl(vr_sub,4,po, n);
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = - pi[i];
	}

	return 0;
}


