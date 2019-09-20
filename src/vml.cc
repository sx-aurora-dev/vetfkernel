#include "vml.h"
namespace vml
{

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


