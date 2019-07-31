#ifndef __LIBVETFKERNEL__
#define __LIBVETFKERNEL__
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int BiasAdd_NHWC_f32(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel);
int BiasAdd_NCHW_f32(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel);

int BiasAddGrad_NHWC(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel);
int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel);

int transpose4_0231_f32(uint64_t out, uint64_t in, const int32_t* dim_size);
int transpose4_0312_f32(uint64_t out, uint64_t in, const int32_t* dim_size);



int add_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n);
int add_nn_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n);

int sub_nn_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n);

int mul_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n);
int mul_nn_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n);

int div_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n);
int div2_nn_n1_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n0, size_t n1);

int sqrt_(uint64_t out, uint64_t in, size_t n);
int rsqrt(uint64_t out, uint64_t in, size_t n);
int square(uint64_t out, uint64_t in, size_t n);

int neg(uint64_t out, uint64_t in, size_t n);

void _apply_adam_f32(float* var, float* m, float *v,
                     const float beta1, const float beta2, 
                     const float epsilon, const float k,
                     const int64_t numElement, 
                     const float *grd ) ;




int sum_d3a02_f32(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2);
int sum_d2a0_f32(uint64_t out, uint64_t in, size_t dim0, size_t dim1);

int mean_d3a02_f32(uint64_t out, uint64_t in, size_t dim0, size_t dim1, size_t dim2);
int mean_d2a0_f32(uint64_t out, uint64_t in, size_t dim0, size_t dim1);


int tile_dim5_11_f32(float* px, float const*py, int64_t const* sx, int64_t const* sy);



#ifdef __cplusplus
}
#endif

#endif
