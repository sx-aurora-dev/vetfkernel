#include "vml.h"
#include "asl.h"

namespace vml
{

int initialize() {
  ASL::initialize();
  return 0;
}

int finalize() {
  ASL::finalize();
  return 0;
}

std::ostream& operator<<(std::ostream& s, Tensor const& t)
{
  s << "[dtype=" << t.dtype
    << ",dims=" << t.dims
    << ",nelems=" << t.nelems
    << ",dim_size=[";

  for (size_t i = 0; i < t.dims; ++i) {
    s << t.dim_size[i];
    if (i < t.dims - 1)
      s << ",";
  }
  s << "]]";
}

} // namespace vml


