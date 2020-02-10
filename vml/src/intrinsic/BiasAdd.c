#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "intrinsic.h"

#include "velintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


static inline void biasadd_nhwc_vectc_cu1mvl(
    float *pout,
    const float *pin,
    const float *pbias,
    const size_t bhw,
    const size_t channel,
    const size_t c,
    const size_t remvl )
{
  __vr vrb0 = _vel_vldu_vssl(4, pbias+c+0*256, remvl);
  for (size_t b = 0; b < bhw; ++b) {
    __vr vri0  = _vel_vldu_vssl(4, pin+b*channel+c+0*256, remvl);

    __vr vro0 = _vel_vfadds_vvvl(vrb0, vri0, remvl);

    _vel_vstu_vssl(vro0, 4, pout+b*channel+c+0*256, remvl);
  }
}

static inline void biasadd_nhwc_vectc_cu2mvl(
    float *pout,
    const float *pin,
    const float *pbias,
    const size_t bhw,
    const size_t channel,
    const size_t c,
    const size_t remvl )
{
  __vr vrb0 = _vel_vldu_vssl(4, pbias+c+0*256, 256);
  __vr vrb1 = _vel_vldu_vssl(4, pbias+c+1*256, remvl);
  for (size_t b = 0; b < bhw; ++b) {
    __vr vri0  = _vel_vldu_vssl(4, pin+b*channel+c+0*256, 256);
    __vr vri1  = _vel_vldu_vssl(4, pin+b*channel+c+1*256, remvl);

    __vr vro0 = _vel_vfadds_vvvl(vrb0, vri0, 256);
    __vr vro1 = _vel_vfadds_vvvl(vrb1, vri1, remvl);

    _vel_vstu_vssl(vro0, 4, pout+b*channel+c+0*256, 256);
    _vel_vstu_vssl(vro1, 4, pout+b*channel+c+1*256, remvl);
  }
}

static inline void biasadd_nhwc_vectc_cu3mvl(
    float *pout,
    const float *pin,
    const float *pbias,
    const size_t bhw,
    const size_t channel,
    const size_t c,
    const size_t remvl )
{
  __vr vrb0 = _vel_vldu_vssl(4, pbias+c+0*256, 256);
  __vr vrb1 = _vel_vldu_vssl(4, pbias+c+1*256, 256);
  __vr vrb2 = _vel_vldu_vssl(4, pbias+c+2*256, remvl);
  for (size_t b = 0; b < bhw; ++b) {
    __vr vri0  = _vel_vldu_vssl(4, pin+b*channel+c+0*256, 256);
    __vr vri1  = _vel_vldu_vssl(4, pin+b*channel+c+1*256, 256);
    __vr vri2  = _vel_vldu_vssl(4, pin+b*channel+c+2*256, remvl);

    __vr vro0 = _vel_vfadds_vvvl(vrb0, vri0, 256);
    __vr vro1 = _vel_vfadds_vvvl(vrb1, vri1, 256);
    __vr vro2 = _vel_vfadds_vvvl(vrb2, vri2, remvl);

    _vel_vstu_vssl(vro0, 4, pout+b*channel+c+0*256, 256);
    _vel_vstu_vssl(vro1, 4, pout+b*channel+c+1*256, 256);
    _vel_vstu_vssl(vro2, 4, pout+b*channel+c+2*256, remvl);
  }
}

static inline void biasadd_nhwc_vectc_cu4mvl(
    float *pout,
    const float *pin,
    const float *pbias,
    const size_t bhw,
    const size_t channel,
    const size_t c,
    const size_t remvl )
{
  __vr vrb0 = _vel_vldu_vssl(4, pbias+c+0*256, 256);
  __vr vrb1 = _vel_vldu_vssl(4, pbias+c+1*256, 256);
  __vr vrb2 = _vel_vldu_vssl(4, pbias+c+2*256, 256);
  __vr vrb3 = _vel_vldu_vssl(4, pbias+c+3*256, remvl);
  for (size_t b = 0; b < bhw; ++b) {
    __vr vri0  = _vel_vldu_vssl(4, pin+b*channel+c+0*256, 256);
    __vr vri1  = _vel_vldu_vssl(4, pin+b*channel+c+1*256, 256);
    __vr vri2  = _vel_vldu_vssl(4, pin+b*channel+c+2*256, 256);
    __vr vri3  = _vel_vldu_vssl(4, pin+b*channel+c+3*256, remvl);

    __vr vro0 = _vel_vfadds_vvvl(vrb0, vri0, 256);
    __vr vro1 = _vel_vfadds_vvvl(vrb1, vri1, 256);
    __vr vro2 = _vel_vfadds_vvvl(vrb2, vri2, 256);
    __vr vro3 = _vel_vfadds_vvvl(vrb3, vri3, remvl);

    _vel_vstu_vssl(vro0, 4, pout+b*channel+c+0*256, 256);
    _vel_vstu_vssl(vro1, 4, pout+b*channel+c+1*256, 256);
    _vel_vstu_vssl(vro2, 4, pout+b*channel+c+2*256, 256);
    _vel_vstu_vssl(vro3, 4, pout+b*channel+c+3*256, remvl);
  }
}

int BiasAdd_NHWC_f32(uint64_t out, uint64_t in, uint64_t bias, int batch,
    int width, int height, int channel) {
  float* pout = (float*) (out);
  const float* pin = (const float*) (in);
  const float* pbias = (const float*) (bias);

  if ((channel < 64) && (width * height > channel)) {
    for (int xy = 0; xy < width * height; xy += VLEN) {
      const int64_t vlen =
	  width * height - xy < VLEN ? width * height - xy : VLEN;

      for (int b = 0; b < batch; ++b) {
	int i = b * height * width * channel + xy * channel;
	for (int c = 0; c < channel; ++c) {
	  __vr vr_pin = _vel_vldu_vssl(4 * channel, pin + i + c, vlen);
	  __vr vr_sum = _vel_vfadds_vsvl(pbias[c], vr_pin, vlen);
	  _vel_vstu_vssl(vr_sum, 4 * channel, pout + i + c, vlen);
	}
      }
    }
  } else {
    size_t bhw = ((size_t) batch) * width * height ;
    size_t cremain = channel % (4*256) ;

    size_t c = 0 ;
    if ( cremain > 0 ) {
      if( cremain <= 1*256 ) {
	size_t remvl = cremain % 256 == 0 ? 256 : cremain % 256 ;
	biasadd_nhwc_vectc_cu1mvl(pout, pin, pbias, bhw, channel, c, remvl ) ;
	c += cremain ;
      }
      else if( cremain <= 2*256 ) {
	size_t remvl = cremain % 256 == 0 ? 256 : cremain % 256 ;
	biasadd_nhwc_vectc_cu2mvl(pout, pin, pbias, bhw, channel, c, remvl ) ;
	c += cremain ;
      }
      else if( cremain <= 3*256 ) {
	size_t remvl = cremain % 256 == 0 ? 256 : cremain % 256 ;
	biasadd_nhwc_vectc_cu3mvl(pout, pin, pbias, bhw, channel, c, remvl ) ;
	c += cremain ;
      }
      else { // cremain  < 4*256
	size_t remvl = cremain % 256 == 0 ? 256 : cremain % 256 ;
	biasadd_nhwc_vectc_cu4mvl(pout, pin, pbias, bhw, channel, c, remvl ) ;
	c += cremain ;
      }
    }

    while( c < channel ) {
      biasadd_nhwc_vectc_cu4mvl(pout, pin, pbias, bhw, channel, c, 256 ) ;
      c += 4*256 ;
    }
  }

#if 0
  fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
  return 0;
}






int BiasAdd_NCHW_f32(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
	float* pout = (float*)(out);
	const float* pin = (const float*)(in);
	const float* pbias = (const float*)(bias);

	int wh = width*height;
	const uint64_t alignIn = ((const uint64_t)in) & 0x07;
	const uint64_t alignOut = ((const uint64_t)out) & 0x07;


	if((alignIn==0)&&(alignOut==0)&&(wh%2==0)){
		if(channel%5==0){
			for (int b = 0; b < batch; ++b) {
				for (int c = 0; c < channel; c+=5) {
					for (int xy = 0; xy < width*height; xy+=2*VLEN) {
						const int64_t vlen = (width*height-xy < 2*VLEN ? width*height-xy : 2*VLEN) >> 1;
						 ;
						int i1 = b * height * width * channel + c * height * width;
						int i2 = b * height * width * channel + (c+1) * height * width;
						int i3 = b * height * width * channel + (c+2) * height * width;
						int i4 = b * height * width * channel + (c+3) * height * width;
						int i5 = b * height * width * channel + (c+4) * height * width;
						__vr vr_pin1 = _vel_vld_vssl(8,pin+i1+xy, vlen);
						__vr vr_pin2 = _vel_vld_vssl(8,pin+i2+xy, vlen);
						__vr vr_pin3 = _vel_vld_vssl(8,pin+i3+xy, vlen);
						__vr vr_pin4 = _vel_vld_vssl(8,pin+i4+xy, vlen);
						__vr vr_pin5 = _vel_vld_vssl(8,pin+i5+xy, vlen);

						__vr vr_sum1 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c), vr_pin1, vlen);
						__vr vr_sum2 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c+1), vr_pin2, vlen);
						__vr vr_sum3 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c+2), vr_pin3, vlen);
						__vr vr_sum4 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c+3), vr_pin4, vlen);
						__vr vr_sum5 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c+4), vr_pin5, vlen);

						_vel_vst_vssl(vr_sum1,8,pout+i1+xy, vlen);
						_vel_vst_vssl(vr_sum2,8,pout+i2+xy, vlen);
						_vel_vst_vssl(vr_sum3,8,pout+i3+xy, vlen);
						_vel_vst_vssl(vr_sum4,8,pout+i4+xy, vlen);
						_vel_vst_vssl(vr_sum5,8,pout+i5+xy, vlen);
					}
				}
			}

		}
		else if(channel%4==0){
			for (int b = 0; b < batch; ++b) {
				for (int c = 0; c < channel; c+=4) {
					for (int xy=0; xy < width*height; xy+=2*VLEN) {
						const int64_t vlen = (width*height-xy < 2*VLEN ? width*height-xy : 2*VLEN) >> 1;
						;
						int i1 = b * height * width * channel + c * height * width;
						int i2 = b * height * width * channel + (c+1) * height * width;
						int i3 = b * height * width * channel + (c+2) * height * width;
						int i4 = b * height * width * channel + (c+3) * height * width;
						__vr vr_pin1 = _vel_vld_vssl(8,pin+i1+xy, vlen);
						__vr vr_pin2 = _vel_vld_vssl(8,pin+i2+xy, vlen);
						__vr vr_pin3 = _vel_vld_vssl(8,pin+i3+xy, vlen);
						__vr vr_pin4 = _vel_vld_vssl(8,pin+i4+xy, vlen);
						__vr vr_sum1 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c), vr_pin1, vlen);
						__vr vr_sum2 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c+1), vr_pin2, vlen);
						__vr vr_sum3 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c+2), vr_pin3, vlen);
						__vr vr_sum4 = _vel_pvfadd_vsvl(_vel_pack_f32a(pbias+c+3), vr_pin4, vlen);
						_vel_vst_vssl(vr_sum1,8,pout+i1+xy, vlen);
						_vel_vst_vssl(vr_sum2,8,pout+i2+xy, vlen);
						_vel_vst_vssl(vr_sum3,8,pout+i3+xy, vlen);
						_vel_vst_vssl(vr_sum4,8,pout+i4+xy, vlen);
					}
				}
			}
		}else{

			for (int b = 0; b < batch; ++b) {
				for (int c = 0; c < channel; ++c) {
					 ;
					unsigned long p_bias= _vel_pack_f32a(pbias+c);
					for (int xy = 0; xy < width*height; xy+=2*VLEN) {
						const int64_t vlen = (width*height-xy < 2*VLEN ? width*height-xy : 2*VLEN) >> 1;
						 ;
						int i = b * height * width * channel
							+ c * height * width;
						__vr vr_pin = _vel_vld_vssl(8,pin+i+xy, vlen);
						__vr vr_sum = _vel_pvfadd_vsvl(p_bias, vr_pin, vlen);
						_vel_vst_vssl(vr_sum,8,pout+i+xy, vlen);
					}
				}
			}
		}
	}else{
		for (int b = 0; b < batch; ++b) {
			for (int c = 0; c < channel; ++c) {
				for (int xy = 0; xy < width*height; xy+=VLEN) {
					const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
					 ;
					int i = b * height * width * channel
						+ c * height * width;
					__vr vr_pin = _vel_vldu_vssl(4,pin+i+xy, vlen);
					__vr vr_sum = _vel_vfadds_vsvl(pbias[c], vr_pin, vlen);
					_vel_vstu_vssl(vr_sum,4,pout+i+xy, vlen);
				}
			}
		}
	}









#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}





