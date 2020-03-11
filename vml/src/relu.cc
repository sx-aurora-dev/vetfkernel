#include <cstdio>
#include "types.h"
#include "log.h"
#include "vml.h"

/* Relu6 / Relu6Grad */

namespace {
template <typename T>
int relu6_(vml::Tensor const& a, vml::Tensor const& f)
{
  T* pa = reinterpret_cast<T*>(a.addr);
  T const* pf = reinterpret_cast<T const*>(f.addr);

  for (size_t i = 0; i < a.nelems; ++i) {
    const T positive = pf[i] > T(0) ? pf[i] : T(0) ;
    pa[i] = positive < T(6) ? positive : T(6) ;
  }

  return 0;
}

} // namespace

int vml::relu6(vml::Tensor const& a, vml::Tensor const& f)
{
  if (a.nelems != f.nelems)
    return 1;

  if (a.dtype == DT_FLOAT && f.dtype == DT_FLOAT)
    return relu6_<float>(a, f) ;

  return 1;
}

namespace {
template <typename T>
int relu6_grad_(vml::Tensor const& backprops,
                vml::Tensor const& gradients,
                vml::Tensor const& features )
{
  T* b = reinterpret_cast<T*>(backprops.addr);
  T const* g = reinterpret_cast<T const*>(gradients.addr);
  T const* f = reinterpret_cast<T const*>(features.addr);

  for (size_t i = 0; i < backprops.nelems; ++i) {
    b[i] = ( f[i] > T(0)  && f[i] < T(6) ) ? g[i] : T(0) ;
  }

  return 0;
}

} // namespace


int vml::relu6_grad(vml::Tensor const& backprops,
                    vml::Tensor const& gradients,
                    vml::Tensor const& features )
{
  if (backprops.nelems != gradients.nelems
      || backprops.nelems != features.nelems)
    return 1;

  if (backprops.dtype == DT_FLOAT 
      && gradients.dtype == DT_FLOAT
      && features.dtype == DT_FLOAT)
    return relu6_grad_<float>(backprops, gradients, features) ;

  return 1;
}


/* LeakyRelu / LeakyReluGrad */

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


/* Selu / SeluGrad */

namespace {
template <typename T>
int selu_(vml::Tensor const& a, vml::Tensor const& f)
{
  T* pa = reinterpret_cast<T*>(a.addr);
  T const* pf = reinterpret_cast<T const*>(f.addr);

  const T scale       = static_cast<T>(1.0507009873554804934193349852946);
  const T scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
  const T one         = static_cast<T>(1);
  const T zero        = static_cast<T>(0);

  for (size_t i = 0; i < a.nelems; ++i) {
    const T feature = pf[i] ;
    pa[i] = feature < zero ? scale_alpha * (std::exp(feature) - one) : scale * feature ;
  }

  return 0;
}

} // namespace

int vml::selu(vml::Tensor const& a, vml::Tensor const& f)
{
  if (a.nelems != f.nelems)
    return 1;

  if (a.dtype == DT_FLOAT && f.dtype == DT_FLOAT)
    return selu_<float>(a, f);

  return 1;
}

namespace {
template <typename T>
int selu_grad_(vml::Tensor const& backprops,
               vml::Tensor const& gradients,
               vml::Tensor const& features )
{
  T* b = reinterpret_cast<T*>(backprops.addr);
  T const* g = reinterpret_cast<T const*>(gradients.addr);
  T const* f = reinterpret_cast<T const*>(features.addr);

  const T scale       = static_cast<T>(1.0507009873554804934193349852946);
  const T scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
  const T zero        = static_cast<T>(0.) ;

  for (size_t i = 0; i < backprops.nelems; ++i) {
    const T activation = f[i] ;
    const T gradient   = g[i] ;
    b[i] =  activation < zero ? gradient * (activation + scale_alpha) : gradient * scale ;
  }

  return 0;
}

} // namespace


int vml::selu_grad(vml::Tensor const& backprops,
                         vml::Tensor const& gradients,
                         vml::Tensor const& features )
{
  if (backprops.nelems != gradients.nelems
      || backprops.nelems != features.nelems)
    return 1;

  if (backprops.dtype == DT_FLOAT
      && gradients.dtype == DT_FLOAT
      && features.dtype == DT_FLOAT)
    return selu_grad_<float>(backprops, gradients, features) ;

  return 1;
}
