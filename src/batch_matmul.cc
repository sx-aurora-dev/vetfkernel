#include "ve_ops_common.h"
#include "vml.h"


//
// BatchMatMul
//

namespace {

int op_batch_matmul(const VEOpArgs& args)
{
    if (args.nArguments() != 5)
        return 1;

    const vml::Tensor* in_x = args.arg<vml::Tensor>(0);
    const vml::Tensor* in_y = args.arg<vml::Tensor>(1);
    const vml::Tensor* out = args.arg<vml::Tensor>(2);
    const bool adj_x = *args.arg<int32_t>(3) ? true : false ;
    const bool adj_y = *args.arg<int32_t>(4) ? true : false ;

    return vml::batch_matmul(*in_x, *in_y, *out, adj_x, adj_y);
}

} // namespace

DEFINE_KERNEL(BatchMatMul, op_batch_matmul);
