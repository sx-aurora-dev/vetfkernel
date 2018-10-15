#ifndef __LIBVETFKERNEL__
#define __LIBVETFKERNEL__
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int BiasAdd_NHWC(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel);
int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel);

int BiasAddGrad_NHWC(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel);
int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel);

int transpose4_0231(uint64_t out, uint64_t in, const int32_t* dim_size);
int transpose4_0312(uint64_t out, uint64_t in, const int32_t* dim_size);



int add_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n);
int add_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n);

int sub_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n);

int mul_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n);
int mul_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n);

int div_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n);
int div2_nn_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n0, size_t n1);

int sqrt_(uint64_t out, uint64_t in, size_t n);
int rsqrt(uint64_t out, uint64_t in, size_t n);
int square(uint64_t out, uint64_t in, size_t n);

void neg(uint64_t out, uint64_t in, size_t n);

#ifdef __cplusplus
}
#endif


#endif
