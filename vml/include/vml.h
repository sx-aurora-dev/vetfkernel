#pragma once

// VML: VE Machile Learning Kernels

#include <vector>
#include <ostream>

namespace vml
{

int initialize();
int finalize();

// Tensor with variable number of dimensions
// typical usage: Tensor* t = reinterpret_cast<Tensor*>(ptr)
struct Tensor {
  int32_t dtype;
  uint64_t addr;
  int32_t dims;
  int64_t nelems;
  int64_t dim_size[0];

  template <typename T> T ptr() const { return reinterpret_cast<T>(addr); }
} __attribute__((__packed__));

template <int N>
struct TensorDesc : public Tensor {
  int64_t _dim_size_buf[N];
} __attribute__((__packed__));

std::ostream& operator<<(std::ostream& s, Tensor const& t);

struct PoolingParam {
  int64_t ksize[4];
  int64_t stride[4];
  int32_t data_format;
  int32_t padding; // 1=VALID, 2=SAME
  int64_t pad_rows;
  int64_t pad_cols;
} __attribute__((__packed__));

int avgpool(Tensor const& out, Tensor const& in, PoolingParam const& param);
int avgpoolgrad(Tensor const& out, Tensor const& in, PoolingParam const& param);

// activation op
int relu6(vml::Tensor const& out, vml::Tensor const& in);
int relu6_grad(vml::Tensor const& backprops,
               vml::Tensor const& gradients,
	       vml::Tensor const& features ) ;
int leaky_relu(vml::Tensor const& out, vml::Tensor const& in, double alpha);
int leaky_relu_grad(vml::Tensor const& backprops,
                    vml::Tensor const& gradients,
                    vml::Tensor const& features,
                    double alpha);

// unary op
int abs(vml::Tensor const& out, vml::Tensor const& in);
int sign(vml::Tensor const& out, vml::Tensor const& in);
int exp(vml::Tensor const& out, vml::Tensor const& in);
int expm1(vml::Tensor const& out, vml::Tensor const& in);
int floor(vml::Tensor const& out, vml::Tensor const& in);
int log(vml::Tensor const& out, vml::Tensor const& in);
int log1p(vml::Tensor const& out, vml::Tensor const& in);
int neg(vml::Tensor const& out, vml::Tensor const& in);
int reciprocal(vml::Tensor const& out, vml::Tensor const& in);
int rsqrt(vml::Tensor const& out, vml::Tensor const& in);
int sigmoid(vml::Tensor const& out, vml::Tensor const& in);
int sqrt(vml::Tensor const& out, vml::Tensor const& in);
int square(vml::Tensor const& out, vml::Tensor const& in);
int sin(vml::Tensor const& out, vml::Tensor const& in);
int cos(vml::Tensor const& out, vml::Tensor const& in);
int tan(vml::Tensor const& out, vml::Tensor const& in);
int sinh(vml::Tensor const& out, vml::Tensor const& in);
int cosh(vml::Tensor const& out, vml::Tensor const& in);
int tanh(vml::Tensor const& out, vml::Tensor const& in);
int asinh(vml::Tensor const& out, vml::Tensor const& in);
int acosh(vml::Tensor const& out, vml::Tensor const& in);
int atanh(vml::Tensor const& out, vml::Tensor const& in);

// binary op
int add(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int div(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int mul(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int sub(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int divnonan(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int pow(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int sqdiff(vml::Tensor const& X, vml::Tensor const& Y, vml::Tensor const& Z);
int rsqrt_grad(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int minimum(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int maximum(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int equal(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int notEqual(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int less(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int lessEqual(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int greater(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);
int greaterEqual(vml::Tensor const& out, vml::Tensor const& in0, vml::Tensor const& in1);

int randomUniform(vml::Tensor const& t);
int tile(vml::Tensor const& out, vml::Tensor const& in);

// reduction
int mean(vml::Tensor const& out, vml::Tensor const& in, std::vector<int> const& axis);

// train op
int ApplyGradientDescent(
    vml::Tensor const& var,
    vml::Tensor const& alpha,	// scalar
    vml::Tensor const& delta
) ;
int ApplyAdadelta(
    vml::Tensor const& var,
    vml::Tensor const& accum,
    vml::Tensor const& accum_update,
    vml::Tensor const& lr,		// scalar
    vml::Tensor const& rho,		// scalar
    vml::Tensor const& epsilon,		// scalar
    vml::Tensor const& grad
) ;
int applyMomentum(
    vml::Tensor const& var,
    vml::Tensor const& accum,
    vml::Tensor const& lr,		// scalar
    vml::Tensor const& grad,
    vml::Tensor const& momentum,	// scalar
    const bool use_nesterov
) ;
int applyAdam(
    vml::Tensor const& var,
    vml::Tensor const& m,
    vml::Tensor const& v,
    vml::Tensor const& beta1_power,	// scalar
    vml::Tensor const& beta2_power,	// scalar
    vml::Tensor const& lr,		// scalar
    vml::Tensor const& beta1,		// scalar
    vml::Tensor const& beta2,		// scalar
    vml::Tensor const& epsilon,		// scalar
    vml::Tensor const& grad,
    const bool use_nesterov
) ;

int conv2d(vml::Tensor const& in,
           vml::Tensor const& filter,
           vml::Tensor const& out,
           std::vector<int> params); // stride[2],dilation[2],padding[2],data_format

// pad op
int pad(
    vml::Tensor const& out,
    vml::Tensor const& in,
    float pad_value,          // 
    int32_t *padding          // padding range
);
int pad(
    vml::Tensor const& out,
    vml::Tensor const& in,
    double pad_value,         // 
    int32_t *padding          // padding range
);

}; // namespace vml
