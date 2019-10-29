#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "intrinsic.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


int sqrt_f32(float* po, float const* pi, size_t n)
{
	if(VLEN<n){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			;
			__vr vr_pin = _vel_vldu_vssl(4,pi+i, vlen);
			__vr vr_s = _vel_vfsqrts_vvl(vr_pin, vlen);
			_vel_vstu_vssl(vr_s,4,po+i, vlen);
		}
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = sqrtf(pi[i]);

	}else{
		;
		__vr vr_pin = _vel_vldu_vssl(4,pi, n);
		__vr vr_s = _vel_vfsqrts_vvl(vr_pin, n);
		_vel_vstu_vssl(vr_s,4,po, n);
	}
	return 0;
}


