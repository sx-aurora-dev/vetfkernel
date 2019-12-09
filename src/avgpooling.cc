#include <cstdint>
#include "ve_ops_common.h"
#include "vml.h"

namespace {

int op_avgpool(const VEOpArgs& args)
{
  if (args.nArguments() != 3)
    return 1;

  // ndims == 4 (checked by tf)

  const vml::Tensor* out = args.arg<vml::Tensor>(0);
  const vml::Tensor* in = args.arg<vml::Tensor>(1);
  const vml::PoolingParam* param = args.arg<vml::PoolingParam>(2);

  return vml::avgpool(*out, *in, *param);
}

int op_avgpoolgrad(const VEOpArgs& args)
{
  if (args.nArguments() != 3)
    return 1;

  // ndims == 4 (checked by tf)

  const vml::Tensor* out = args.arg<vml::Tensor>(0);
  const vml::Tensor* in = args.arg<vml::Tensor>(1);
  const vml::PoolingParam* param = args.arg<vml::PoolingParam>(2);

  return vml::avgpoolgrad(*out, *in, *param);
}

} // namespace

DEFINE_KERNEL(AvgPool, op_avgpool);
DEFINE_KERNEL(AvgPoolGrad, op_avgpoolgrad);
