#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "intrinsic.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


int div_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	float i1 = 1.f/(*(const float*)(in1));


	if(n>VLEN*2){
		const uint64_t alignIn = ((const uint64_t)in0) & 0x07;
		const uint64_t alignOut = ((const uint64_t)out) & 0x07;
		if((alignIn==0)&&(alignOut==0)&&(n%2==0)){
			unsigned long int li1 = _vel_pack_f32a(&i1);
			if(n%(8*VLEN)==0){
				;
				for (size_t i = 0; i < n; i+=8*VLEN) {
					__vr vr_pin1 = _vel_vld_vssl(8,pi0+i+2*VLEN*0, VLEN);
					__vr vr_pin2 = _vel_vld_vssl(8,pi0+i+2*VLEN*1, VLEN);
					__vr vr_pin3 = _vel_vld_vssl(8,pi0+i+2*VLEN*2, VLEN);
					__vr vr_pin4 = _vel_vld_vssl(8,pi0+i+2*VLEN*3, VLEN);
					__vr vr_mul1 = _vel_pvfmul_vsvl(li1, vr_pin1, VLEN);
					__vr vr_mul2 = _vel_pvfmul_vsvl(li1, vr_pin2, VLEN);
					__vr vr_mul3 = _vel_pvfmul_vsvl(li1, vr_pin3, VLEN);
					__vr vr_mul4 = _vel_pvfmul_vsvl(li1, vr_pin4, VLEN);
					_vel_vst_vssl(vr_mul1,8,po+i+2*VLEN*0, VLEN);
					_vel_vst_vssl(vr_mul2,8,po+i+2*VLEN*1, VLEN);
					_vel_vst_vssl(vr_mul3,8,po+i+2*VLEN*2, VLEN);
					_vel_vst_vssl(vr_mul4,8,po+i+2*VLEN*3, VLEN);
				}
			}else if((n%8==0)&&(n>VLEN*8)){
				for (size_t i = 0; i < n; i+=8*VLEN) {
					const int64_t vlen = (n-i < 8*VLEN ? n-i : 8*VLEN) >> 3;
					;
					__vr vr_pin1 = _vel_vld_vssl(8,pi0+i+2*vlen*0, vlen);
					__vr vr_pin2 = _vel_vld_vssl(8,pi0+i+2*vlen*1, vlen);
					__vr vr_pin3 = _vel_vld_vssl(8,pi0+i+2*vlen*2, vlen);
					__vr vr_pin4 = _vel_vld_vssl(8,pi0+i+2*vlen*3, vlen);
					__vr vr_mul1 = _vel_pvfmul_vsvl(li1, vr_pin1, vlen);
					__vr vr_mul2 = _vel_pvfmul_vsvl(li1, vr_pin2, vlen);
					__vr vr_mul3 = _vel_pvfmul_vsvl(li1, vr_pin3, vlen);
					__vr vr_mul4 = _vel_pvfmul_vsvl(li1, vr_pin4, vlen);
					_vel_vst_vssl(vr_mul1,8,po+i+2*vlen*0, vlen);
					_vel_vst_vssl(vr_mul2,8,po+i+2*vlen*1, vlen);
					_vel_vst_vssl(vr_mul3,8,po+i+2*vlen*2, vlen);
					_vel_vst_vssl(vr_mul4,8,po+i+2*vlen*3, vlen);
				}
			}else{
				for (size_t i = 0; i < n; i+=2*VLEN) {
					const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
					;
					__vr vr_pin = _vel_vld_vssl(8,pi0+i, vlen);
					__vr vr_mul = _vel_pvfmul_vsvl(li1, vr_pin, vlen);
					_vel_vst_vssl(vr_mul,8,po+i, vlen);
				}
			}
		}else{
			for (size_t i = 0; i < n; i+=VLEN) {
				const int64_t vlen = n-i < VLEN ? n-i : VLEN;
				;
				__vr vr_pin = _vel_vldu_vssl(4,pi0+i, vlen);
				__vr vr_mul = _vel_vfmuls_vsvl(i1, vr_pin, vlen);
				_vel_vstu_vssl(vr_mul,4,po+i, vlen);
			}
		}
	}else if(n>VLEN){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			;
			__vr vr_pin = _vel_vldu_vssl(4,pi0+i, vlen);
			__vr vr_mul = _vel_vfmuls_vsvl(i1, vr_pin, vlen);
			_vel_vstu_vssl(vr_mul,4,po+i, vlen);
		}
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = pi0[i] * i1;
	}else{
		;
		__vr vr_pin = _vel_vldu_vssl(4,pi0, n);
		__vr vr_mul = _vel_vfmuls_vsvl(i1, vr_pin, n);
		_vel_vstu_vssl(vr_mul,4,po, n);
	}

	return 0;
}


int div2_nn_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n0, size_t n1)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	const float* pi1 = (const float*)(in1);

	if((n0>VLEN/3)&&(n0%4==0)){
		if(n1<=VLEN){
			if(n0<=VLEN){
				float temp[VLEN];
				
				__vr vr_pin = _vel_vldu_vssl(4,pi1, n0);
				__vr vr_div = _vel_vfdivs_vsvl(1.0f, vr_pin, n0);
				_vel_vstu_vssl(vr_div,4,temp, n0);

				for (int ii = 0 ; ii < n0; ii+=4) {
					float dval1 = temp[ii+0];
					float dval2 = temp[ii+1];
					float dval3 = temp[ii+2];
					float dval4 = temp[ii+3];
					;
					for (size_t j = 0; j < n1; j+=VLEN) {
						__vr vr_pin1 = _vel_vldu_vssl(4,pi0+(ii+0)*n1+j, n1);
						__vr vr_pin2 = _vel_vldu_vssl(4,pi0+(ii+1)*n1+j, n1);
						__vr vr_pin3 = _vel_vldu_vssl(4,pi0+(ii+2)*n1+j, n1);
						__vr vr_pin4 = _vel_vldu_vssl(4,pi0+(ii+3)*n1+j, n1);
						__vr vr_mul1 = _vel_vfmuls_vsvl(dval1, vr_pin1, n1);
						__vr vr_mul2 = _vel_vfmuls_vsvl(dval2, vr_pin2, n1);
						__vr vr_mul3 = _vel_vfmuls_vsvl(dval3, vr_pin3, n1);
						__vr vr_mul4 = _vel_vfmuls_vsvl(dval4, vr_pin4, n1);
						_vel_vstu_vssl(vr_mul1,4,po+(ii+0)*n1+j, n1);
						_vel_vstu_vssl(vr_mul2,4,po+(ii+1)*n1+j, n1);
						_vel_vstu_vssl(vr_mul3,4,po+(ii+2)*n1+j, n1);
						_vel_vstu_vssl(vr_mul4,4,po+(ii+3)*n1+j, n1);
					}
				}
#if 1
                }else{
                                float temp[VLEN];
                                for (size_t i = 0; i < n0; i+=VLEN) {
                                        const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
                                        ;
                                        __vr vr_pin = _vel_vldu_vssl(4,pi1+i, vlen);
                                        __vr vr_div = _vel_vfdivs_vsvl(1.0f, vr_pin, vlen);
                                        _vel_vstu_vssl(vr_div,4,temp, vlen);

                                        for (int ii = i ; ii < vlen+i; ii+=4) {
                                                float dval1 = temp[ii-i+0];
                                                float dval2 = temp[ii-i+1];
                                                float dval3 = temp[ii-i+2];
                                                float dval4 = temp[ii-i+3];
                                                
                                                for (size_t j = 0; j < n1; j+=VLEN) {
                                                        __vr vr_pin1 = _vel_vldu_vssl(4,pi0+(ii+0)*n1+j, n1);
                                                        __vr vr_pin2 = _vel_vldu_vssl(4,pi0+(ii+1)*n1+j, n1);
                                                        __vr vr_pin3 = _vel_vldu_vssl(4,pi0+(ii+2)*n1+j, n1);
                                                        __vr vr_pin4 = _vel_vldu_vssl(4,pi0+(ii+3)*n1+j, n1);
                                                        __vr vr_mul1 = _vel_vfmuls_vsvl(dval1, vr_pin1, n1);
                                                        __vr vr_mul2 = _vel_vfmuls_vsvl(dval2, vr_pin2, n1);
                                                        __vr vr_mul3 = _vel_vfmuls_vsvl(dval3, vr_pin3, n1);
                                                        __vr vr_mul4 = _vel_vfmuls_vsvl(dval4, vr_pin4, n1);
                                                        _vel_vstu_vssl(vr_mul1,4,po+(ii+0)*n1+j, n1);
                                                        _vel_vstu_vssl(vr_mul2,4,po+(ii+1)*n1+j, n1);
                                                        _vel_vstu_vssl(vr_mul3,4,po+(ii+2)*n1+j, n1);
                                                        _vel_vstu_vssl(vr_mul4,4,po+(ii+3)*n1+j, n1);
                                                }
                                        }
                                }
                        }

#else
			}else{
				float temp[VLEN];
				for (size_t i = 0; i < n0; i+=VLEN) {
					const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
					
					__vr vr_pin = _vel_vldu_vssl(4,pi1+i, vlen);
					__vr vr_div = _vel_vfdivs_vsvl(1.0f, vr_pin, vlen);
					_vel_vstu_vssl(vr_div,4,temp, vlen);

					for (int ii = i ; ii < vlen+i; ii+=4) {
						float dval1 = temp[ii-i+0];
						float dval2 = temp[ii-i+1];
						float dval3 = temp[ii-i+2];
						float dval4 = temp[ii-i+3];
						;
						for (size_t j = 0; j < n1; j+=VLEN) {
							__vr vr_pin1 = _vel_vldu_vssl(4,pi0+(ii+0)*n1+j, n1);
							__vr vr_pin2 = _vel_vldu_vssl(4,pi0+(ii+1)*n1+j, n1);
							__vr vr_pin3 = _vel_vldu_vssl(4,pi0+(ii+2)*n1+j, n1);
							__vr vr_pin4 = _vel_vldu_vssl(4,pi0+(ii+3)*n1+j, n1);
							__vr vr_mul1 = _vel_vfmuls_vsvl(dval1, vr_pin1, n1);
							__vr vr_mul2 = _vel_vfmuls_vsvl(dval2, vr_pin2, n1);
							__vr vr_mul3 = _vel_vfmuls_vsvl(dval3, vr_pin3, n1);
							__vr vr_mul4 = _vel_vfmuls_vsvl(dval4, vr_pin4, n1);
							_vel_vstu_vssl(vr_mul1,4,po+(ii+0)*n1+j, n1);
							_vel_vstu_vssl(vr_mul2,4,po+(ii+1)*n1+j, n1);
							_vel_vstu_vssl(vr_mul3,4,po+(ii+2)*n1+j, n1);
							_vel_vstu_vssl(vr_mul4,4,po+(ii+3)*n1+j, n1);
						}
					}
				}
			}
#endif
		}else{
			float temp[VLEN];
			for (size_t i = 0; i < n0; i+=VLEN) {
				const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
				
				__vr vr_pin = _vel_vldu_vssl(4,pi1+i, vlen);
				__vr vr_div = _vel_vfdivs_vsvl(1.0f, vr_pin, vlen);
				_vel_vstu_vssl(vr_div,4,temp, vlen);

				for (int ii = i ; ii < vlen+i; ii+=4) {
					float dval1 = temp[ii-i+0];
					float dval2 = temp[ii-i+1];
					float dval3 = temp[ii-i+2];
					float dval4 = temp[ii-i+3];
					for (size_t j = 0; j < n1; j+=VLEN) {
						const int64_t vlen1 = n1-j < VLEN ? n1-j : VLEN;
						;

						__vr vr_pin1 = _vel_vldu_vssl(4,pi0+(ii+0)*n1+j, vlen1);
						__vr vr_pin2 = _vel_vldu_vssl(4,pi0+(ii+1)*n1+j, vlen1);
						__vr vr_pin3 = _vel_vldu_vssl(4,pi0+(ii+2)*n1+j, vlen1);
						__vr vr_pin4 = _vel_vldu_vssl(4,pi0+(ii+3)*n1+j, vlen1);
						__vr vr_mul1 = _vel_vfmuls_vsvl(dval1, vr_pin1, vlen1);
						__vr vr_mul2 = _vel_vfmuls_vsvl(dval2, vr_pin2, vlen1);
						__vr vr_mul3 = _vel_vfmuls_vsvl(dval3, vr_pin3, vlen1);
						__vr vr_mul4 = _vel_vfmuls_vsvl(dval4, vr_pin4, vlen1);
						_vel_vstu_vssl(vr_mul1,4,po+(ii+0)*n1+j, vlen1);
						_vel_vstu_vssl(vr_mul2,4,po+(ii+1)*n1+j, vlen1);
						_vel_vstu_vssl(vr_mul3,4,po+(ii+2)*n1+j, vlen1);
						_vel_vstu_vssl(vr_mul4,4,po+(ii+3)*n1+j, vlen1);
					}
				}
			}
		}
	}else if(n0<n1){
		for (size_t i = 0; i < n0; ++i) {
			float dval = 1.0f / pi1[i];
			for (size_t j = 0; j < n1; j+=VLEN) {
				const int64_t vlen = n1-j < VLEN ? n1-j : VLEN;
				
				__vr vr_pin = _vel_vldu_vssl(4,pi0+i*n1+j, vlen);
				__vr vr_mul = _vel_vfmuls_vsvl(dval, vr_pin, vlen);
				_vel_vstu_vssl(vr_mul,4,po+i*n1+j, vlen);
			}
		}
	}else{
		for (size_t i = 0; i < n0; i+=VLEN) {
			const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
			
			__vr vr_pi1 = _vel_vldu_vssl(4,pi1+i, vlen);
			__vr dval = _vel_vfdivs_vsvl(1.0f,vr_pi1, vlen);
			for (size_t j = 0; j < n1; j++) {
				__vr vr_pin = _vel_vldu_vssl(4*n1,pi0+i*n1+j, vlen);
				__vr vr_mul = _vel_vfmuls_vvvl(dval, vr_pin, vlen);
				_vel_vstu_vssl(vr_mul,4*n1,po+i*n1+j, vlen);
			}
		}
	}

	return 0;
}


