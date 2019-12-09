#include <vml.h>
#include "asl.h"

#include "types.h"
#include "log.h"

int vml::randomUniform(vml::Tensor const& t)
{
    LOG(LOG_PARAM) << __FUNCTION__ << ": dtype=" << t.dtype << " nelems=" << t.nelems;

    if (t.dtype == DT_FLOAT) {
        float* p = reinterpret_cast<float*>(t.addr);
        ASL::getRandom(t.nelems, p) ;
    }

    return 0;
}

