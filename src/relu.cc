#include <cstdio>
#include "ve_ops_common.h"
#include "types.h"
#include "log.h"
#include "vml.h"

namespace {
template <typename T>
int leaky_relu_(vml::Tensor const& a, vml::Tensor const& f, T alpha)
{
  T* pa = reinterpret_cast<T*>(a.addr);
  T const* pf = reinterpret_cast<T const*>(f.addr);

  for (size_t i = 0; i < a.nelems; ++i) {
    pa[i] = pf[i] > T(0) ? pf[i] : pf[i] * alpha;
  }

  return 0;
}

} // namespace

int vml::leaky_relu(vml::Tensor const& a, vml::Tensor const& f, double alpha)
{
  if (a.nelems != f.nelems)
    return 1;

  if (a.dtype == DT_FLOAT && f.dtype == DT_FLOAT)
    return leaky_relu_<float>(a, f, (float)alpha);

  return 1;
}

namespace {
template <typename T>
int leaky_relu_grad_(vml::Tensor const& backprops,
                     vml::Tensor const& gradients,
                     vml::Tensor const& features,
                     T alpha)
{
  T* b = reinterpret_cast<T*>(backprops.addr);
  T const* g = reinterpret_cast<T const*>(gradients.addr);
  T const* f = reinterpret_cast<T const*>(features.addr);

  for (size_t i = 0; i < backprops.nelems; ++i) {
    b[i] = f[i] > T(0) ? g[i] : g[i] * alpha;
  }

  return 0;
}

} // namespace


int vml::leaky_relu_grad(vml::Tensor const& backprops,
                         vml::Tensor const& gradients,
                         vml::Tensor const& features,
                         double alpha)
{
  if (backprops.nelems != gradients.nelems
      || backprops.nelems != features.nelems)
    return 1;

  if (backprops.dtype == DT_FLOAT 
      && gradients.dtype == DT_FLOAT
      && features.dtype == DT_FLOAT)
    return leaky_relu_grad_<float>(backprops, gradients, features, (float)alpha);

  return 1;
}



int op_leaky_relu(const VEOpArgs& args)
{
  const vml::Tensor* f = args.arg<vml::Tensor>(0);
  const vml::Tensor* a = args.arg<vml::Tensor>(1);
  double alpha = *args.arg<double>(2);

  return vml::leaky_relu(*a, *f, alpha);
}

int op_leaky_relu_grad(const VEOpArgs& args)
{
  const vml::Tensor* gradients = args.arg<vml::Tensor>(0);
  const vml::Tensor* features = args.arg<vml::Tensor>(1);
  const vml::Tensor* backprops = args.arg<vml::Tensor>(2);
  double alpha = *args.arg<double>(3);

  return vml::leaky_relu_grad(*backprops, *gradients, *features, alpha);
}

DEFINE_KERNEL(LeakyRelu, op_leaky_relu);
DEFINE_KERNEL(LeakyReluGrad, op_leaky_relu_grad);
