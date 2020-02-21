#include <vml.h>
#include <vml/log.h>
#include "asl.h"

namespace vml
{

int initialize() {
  LOG(LOG_TRACE) << __FUNCTION__;
#ifdef __ve__
  ASL::initialize();
#endif
  return 0;
}

int finalize() {
  LOG(LOG_TRACE) << __FUNCTION__;
#ifdef __ve__
  ASL::finalize();
#endif
  return 0;
}

std::ostream& operator<<(std::ostream& s, Tensor const& t)
{
  s << "[dtype=" << t.dtype
    << ",addr=" << t.addr
    << ",dims=" << t.dims
    << ",nelems=" << t.nelems
    << ",dim_size=[";

  for (size_t i = 0; i < t.dims; ++i) {
    s << t.dim_size[i];
    if (i < t.dims - 1)
      s << ",";
  }
  s << "]]";
  return s;
}

} // namespace vml


