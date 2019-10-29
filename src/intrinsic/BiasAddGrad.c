#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "intrinsic.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))




int BiasAddGrad_NHWC(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
	float* pout = (float*)(output);
	const float* pin = (const float*)(output_backprop);


	const uint64_t alignOut = ((const uint64_t)output) & 0x07;
	const uint64_t alignIn = ((const uint64_t)output_backprop) & 0x07;

	if((alignIn==0)&&(alignOut==0)&&(channel%2==0)&&(channel>256)){
		for (int c = 0; c < channel; c+=2*VLEN) {
			const int64_t vlen = (channel-c < 2*VLEN ? channel-c : 2*VLEN) >> 1;
			;
			__vr vr_sum = _vel_vbrdd_vsl(0.f, vlen);
			for (int b = 0; b < batch; ++b) {
				for (int xy = 0; xy < width*height; ++xy) {
					int pos = b * height * width * channel
						+ xy * channel;
					__vr vr_pin = _vel_vld_vssl(8,pin+pos+c, vlen);
					vr_sum = _vel_pvfadd_vvvl(vr_sum, vr_pin, vlen);
				}
			}
			_vel_vst_vssl(vr_sum,8,pout+c, vlen);
		}

	}else{
		for (int c = 0; c < channel; c+=VLEN) {
			const int64_t vlen = channel-c < VLEN ? channel-c : VLEN;
			;
			__vr vr_sum = _vel_vbrds_vsl(0.f, vlen);
			for (int b = 0; b < batch; ++b) {
				for (int xy = 0; xy < width*height; ++xy) {
					int pos = b * height * width * channel
						+ xy * channel;
					__vr vr_pin = _vel_vldu_vssl(4,pin+pos+c, vlen);
					vr_sum = _vel_vfadds_vvvl(vr_sum, vr_pin, vlen);
				}
			}
			_vel_vstu_vssl(vr_sum,4,pout+c, vlen);
		}
	}



	return 0;
}


int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
	float* pout = (float*)(output);
	const float* pin = (const float*)(output_backprop);

	const uint64_t alignIn = ((const uint64_t)output_backprop) & 0x07;

	if(((width*height)<64)&&(width*height<channel)){
		for (int c = 0; c < channel; c+=VLEN) {
			const int64_t vlen = channel-c < VLEN ? channel-c : VLEN;
			 ;
			__vr vr_sum = _vel_vbrds_vsl(0.f, vlen);
			for (int b = 0; b < batch; ++b) {
				int pos = b * channel * height * width + c * height * width;
				for (int i = 0; i < width * height; i++) {
					__vr vr_pin = _vel_vldu_vssl(4*width*height,pin+pos+i, vlen);
					vr_sum = _vel_vfadds_vvvl(vr_sum, vr_pin, vlen);
				}
			}
			_vel_vstu_vssl(vr_sum,4,pout+c, vlen);
		}
	}else{
		if((alignIn==0)&&(width*height%2==0)){
			if(channel%4==0){
				for (int c = 0; c < channel; c+=4) {
					;
					__vr vr_sum1 = _vel_vbrdd_vsl(0, 256);
					__vr vr_sum2 = _vel_vbrdd_vsl(0, 256);
					__vr vr_sum3 = _vel_vbrdd_vsl(0, 256);
					__vr vr_sum4 = _vel_vbrdd_vsl(0, 256);
					for (int b = 0; b < batch; ++b) {
						int pos1 = b * channel * height * width + (c+0) * height * width;
						int pos2 = b * channel * height * width + (c+1) * height * width;
						int pos3 = b * channel * height * width + (c+2) * height * width;
						int pos4 = b * channel * height * width + (c+3) * height * width;
						for (int i = 0; i < width * height; i+=2*VLEN) {
							const int64_t vlen = (width*height-i < 2*VLEN ? width*height-i : 2*VLEN) >> 1;
							 ;
							__vr vr_pin1 = _vel_vld_vssl(8,pin+pos1+i, vlen);
							__vr vr_pin2 = _vel_vld_vssl(8,pin+pos2+i, vlen);
							__vr vr_pin3 = _vel_vld_vssl(8,pin+pos3+i, vlen);
							__vr vr_pin4 = _vel_vld_vssl(8,pin+pos4+i, vlen);
							vr_sum1 = _vel_pvfadd_vvvl(vr_sum1, vr_pin1, vlen);
							vr_sum2 = _vel_pvfadd_vvvl(vr_sum2, vr_pin2, vlen);
							vr_sum3 = _vel_pvfadd_vvvl(vr_sum3, vr_pin3, vlen);
							vr_sum4 = _vel_pvfadd_vvvl(vr_sum4, vr_pin4, vlen);
						}
					}
					;
					vr_sum1 = _vel_vfadds_vvvl(vr_sum1, _vel_vsll_vvsl(vr_sum1,32, 256), 256);
					vr_sum2 = _vel_vfadds_vvvl(vr_sum2, _vel_vsll_vvsl(vr_sum2,32, 256), 256);
					vr_sum3 = _vel_vfadds_vvvl(vr_sum3, _vel_vsll_vvsl(vr_sum3,32, 256), 256);
					vr_sum4 = _vel_vfadds_vvvl(vr_sum4, _vel_vsll_vvsl(vr_sum4,32, 256), 256);
					vr_sum1 = _vel_vfsums_vvl(vr_sum1, 256);
					vr_sum2 = _vel_vfsums_vvl(vr_sum2, 256);
					vr_sum3 = _vel_vfsums_vvl(vr_sum3, 256);
					vr_sum4 = _vel_vfsums_vvl(vr_sum4, 256);
					;
					_vel_vstu_vssl(vr_sum1,4,pout+c+0, 1);
					_vel_vstu_vssl(vr_sum2,4,pout+c+1, 1);
					_vel_vstu_vssl(vr_sum3,4,pout+c+2, 1);
					_vel_vstu_vssl(vr_sum4,4,pout+c+3, 1);
				}

			}else{
				for (int c = 0; c < channel; ++c) {
					;
					__vr vr_sum = _vel_vbrdd_vsl(0, 256);
					for (int b = 0; b < batch; ++b) {
						int pos = b * channel * height * width + c * height * width;
						for (int i = 0; i < width * height; i+=2*VLEN) {
							const int64_t vlen = (width*height-i < 2*VLEN ? width*height-i : 2*VLEN) >> 1;
							 ;
							__vr vr_pin = _vel_vld_vssl(8,pin+pos+i, vlen);
							vr_sum = _vel_pvfadd_vvvl(vr_sum, vr_pin, vlen);
						}
					}
					;
					vr_sum = _vel_vfadds_vvvl(vr_sum, _vel_vsll_vvsl(vr_sum,32, 256), 256);
					vr_sum = _vel_vfsums_vvl(vr_sum, 256);
					;
					_vel_vstu_vssl(vr_sum,4,pout+c, 1);
				}
			}
		}else{
			for (int c = 0; c < channel; ++c) {
				;		
				__vr vr_sum = _vel_vbrds_vsl(0.f, 256);
				for (int b = 0; b < batch; ++b) {
					int pos = b * channel * height * width + c * height * width;
					for (int i = 0; i < width * height; i+=VLEN) {
						const int64_t vlen = width*height-i < VLEN ? width*height-i : VLEN;
						 ;
						__vr vr_pin = _vel_vldu_vssl(4,pin+pos+i, vlen);
						vr_sum = _vel_vfadds_vvvl(vr_sum, vr_pin, vlen);
					}
				}
				;
				vr_sum = _vel_vfsums_vvl(vr_sum, 256);
				;
				_vel_vstu_vssl(vr_sum,4,pout+c, 1);
			}
		}
	}

	return 0;
}



