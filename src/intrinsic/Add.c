#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

int add_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	float i1 = *((const float*)(in1));



	const uint64_t alignIn = ((const uint64_t)in0) & 0x07;
	const uint64_t alignOut = ((const uint64_t)out) & 0x07;


	if((alignIn==0)&&(alignOut==0)&&(n%2==0)&&(n>256)){
		unsigned long int li1 = _vel_pack_f32a(&i1);
		if(n%(8*VLEN)==0){
			;
			for (size_t i = 0; i < n; i+=8*VLEN) {
				__vr vr_pin1 = _vel_vld_vssl(8,pi0+i+2*VLEN*0, VLEN);
				__vr vr_pin2 = _vel_vld_vssl(8,pi0+i+2*VLEN*1, VLEN);
				__vr vr_pin3 = _vel_vld_vssl(8,pi0+i+2*VLEN*2, VLEN);
				__vr vr_pin4 = _vel_vld_vssl(8,pi0+i+2*VLEN*3, VLEN);
				__vr vr_sum1 = _vel_pvfadd_vsvl(li1, vr_pin1, VLEN);
				__vr vr_sum2 = _vel_pvfadd_vsvl(li1, vr_pin2, VLEN);
				__vr vr_sum3 = _vel_pvfadd_vsvl(li1, vr_pin3, VLEN);
				__vr vr_sum4 = _vel_pvfadd_vsvl(li1, vr_pin4, VLEN);
				_vel_vst_vssl(vr_sum1,8,po+i+2*VLEN*0, VLEN);
				_vel_vst_vssl(vr_sum2,8,po+i+2*VLEN*1, VLEN);
				_vel_vst_vssl(vr_sum3,8,po+i+2*VLEN*2, VLEN);
				_vel_vst_vssl(vr_sum4,8,po+i+2*VLEN*3, VLEN);
			}
		}else if(n%8==0){
			for (size_t i = 0; i < n; i+=8*VLEN) {
				const int64_t vlen = (n-i < 8*VLEN ? n-i : 8*VLEN) >> 3;
				;
				__vr vr_pin1 = _vel_vld_vssl(8,pi0+i+2*vlen*0, vlen);
				__vr vr_pin2 = _vel_vld_vssl(8,pi0+i+2*vlen*1, vlen);
				__vr vr_pin3 = _vel_vld_vssl(8,pi0+i+2*vlen*2, vlen);
				__vr vr_pin4 = _vel_vld_vssl(8,pi0+i+2*vlen*3, vlen);
				__vr vr_sum1 = _vel_pvfadd_vsvl(li1, vr_pin1, vlen);
				__vr vr_sum2 = _vel_pvfadd_vsvl(li1, vr_pin2, vlen);
				__vr vr_sum3 = _vel_pvfadd_vsvl(li1, vr_pin3, vlen);
				__vr vr_sum4 = _vel_pvfadd_vsvl(li1, vr_pin4, vlen);
				_vel_vst_vssl(vr_sum1,8,po+i+2*vlen*0, vlen);
				_vel_vst_vssl(vr_sum2,8,po+i+2*vlen*1, vlen);
				_vel_vst_vssl(vr_sum3,8,po+i+2*vlen*2, vlen);
				_vel_vst_vssl(vr_sum4,8,po+i+2*vlen*3, vlen);
			}
		}else{
			for (size_t i = 0; i < n; i+=2*VLEN) {
				const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
				;
				__vr vr_pin = _vel_vld_vssl(8,pi0+i, vlen);
				__vr vr_sum = _vel_pvfadd_vsvl(li1, vr_pin, vlen);
				_vel_vst_vssl(vr_sum,8,po+i, vlen);
			}
		}
	}else if(n>256){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			;
			__vr vr_pin = _vel_vldu_vssl(4,pi0+i, vlen);
			__vr vr_sum = _vel_vfadds_vsvl(i1, vr_pin, vlen);
			_vel_vstu_vssl(vr_sum,4,po+i, vlen);
		}
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = pi0[i] + i1;

	}else{
		;
		__vr vr_pin = _vel_vldu_vssl(4,pi0, n);
		__vr vr_sum = _vel_vfadds_vsvl(i1, vr_pin, n);
		_vel_vstu_vssl(vr_sum,4,po, n);
	}
	return 0;
}

int add_nn_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
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
			__vr vr_sum = _vel_pvfadd_vvvl(vr_pin0, vr_pin1, vlen);
			_vel_vst_vssl(vr_sum,8,po+i, vlen);
		}
	}else{
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			 ;
			__vr vr_pin0 = _vel_vldu_vssl(4,pi0+i, vlen);
			__vr vr_pin1 = _vel_vldu_vssl(4,pi1+i, vlen);
			__vr vr_sum = _vel_vfadds_vvvl(vr_pin0, vr_pin1, vlen);
			_vel_vstu_vssl(vr_sum,4,po+i, vlen);
		}
	}





	return 0;
}


