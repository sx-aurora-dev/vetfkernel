#include <cstdio>
#include "ve_ops_common.h"
#include "vml.h"

/* Relu6 / Relu6Grad */

int op_relu6(const VEOpArgs& args)
{
  const vml::Tensor* f = args.arg<vml::Tensor>(0);
  const vml::Tensor* a = args.arg<vml::Tensor>(1);

  return vml::relu6(*a, *f) ;
}

int op_relu6_grad(const VEOpArgs& args)
{
  const vml::Tensor* gradients = args.arg<vml::Tensor>(0);
  const vml::Tensor* features = args.arg<vml::Tensor>(1);
  const vml::Tensor* backprops = args.arg<vml::Tensor>(2);

  return vml::relu6_grad(*backprops, *gradients, *features) ;
}

DEFINE_KERNEL(Relu6, op_relu6);
DEFINE_KERNEL(Relu6Grad, op_relu6_grad);


/* LeakyRelu / LeakyReluGrad */

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


/* Elu / EluGrad */

int op_elu(const VEOpArgs& args)
{
  const vml::Tensor* f = args.arg<vml::Tensor>(0);
  const vml::Tensor* a = args.arg<vml::Tensor>(1);

  return vml::elu(*a, *f);
}

int op_elu_grad(const VEOpArgs& args)
{
  const vml::Tensor* gradients = args.arg<vml::Tensor>(0);
  const vml::Tensor* features = args.arg<vml::Tensor>(1);
  const vml::Tensor* backprops = args.arg<vml::Tensor>(2);

  return vml::elu_grad(*backprops, *gradients, *features);
}

DEFINE_KERNEL(Elu, op_elu);
DEFINE_KERNEL(EluGrad, op_elu_grad);



/* Selu / SeluGrad */

int op_selu(const VEOpArgs& args)
{
  const vml::Tensor* f = args.arg<vml::Tensor>(0);
  const vml::Tensor* a = args.arg<vml::Tensor>(1);

  return vml::selu(*a, *f);
}

int op_selu_grad(const VEOpArgs& args)
{
  const vml::Tensor* gradients = args.arg<vml::Tensor>(0);
  const vml::Tensor* features = args.arg<vml::Tensor>(1);
  const vml::Tensor* backprops = args.arg<vml::Tensor>(2);

  return vml::selu_grad(*backprops, *gradients, *features);
}

DEFINE_KERNEL(Selu, op_selu);
DEFINE_KERNEL(SeluGrad, op_selu_grad);
