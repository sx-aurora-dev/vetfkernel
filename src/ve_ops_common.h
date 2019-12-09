#ifndef VE_OPS_COMMON_H_
#define VE_OPS_COMMON_H_

#include <cstdint>
#include "vml/log.h"
#include "kernel.h"

#define DEFINE_KERNEL(NAME, FUNC) \
  REGISTER_KERNEL(#NAME, "op_" # NAME); \
  extern "C" { \
    int op_##NAME(const void* args, size_t len) { \
      return op_Kernel(args, len, FUNC, "op_" # NAME); \
    } \
  }

namespace {

class VEOpArgs  {
  public:
    VEOpArgs(const void* buf) : buf_(buf) {
      pHeader_ = reinterpret_cast<const Header*>(buf);
      pVariable_ = reinterpret_cast<uintptr_t>(buf) + sizeof(Header);

#if 0
      LOG(LOG_DETAIL) << __FUNCTION__ << ": buf=" << buf << " pHeader_=" << pHeader_ << " pTensor_=" << pTensor_ << " (", pTensor_ - reinterpret_cast<uintptr_t>(pHeader_) <<  ")", 
#endif

      const int* p = reinterpret_cast<const int*>(pVariable_);
#if 0
      LOG(LOG_DETAIL) << __FUNCTION__ << ": *p=" << *p;
#endif

#if 0
      tensor_size_ = sizeof(Tensor) + sizeof(int64_t) * (pHeader_->max_dim_size - 1);
#endif
    }

    int64_t nArguments() const { return pHeader_->nArguments; }

    template<typename T>
    const T* arg(int i) const {
      uintptr_t p = pVariable_;
      for (int j = 0; j < i; ++j) {
        const size_t size  = *reinterpret_cast<size_t*>(p);
        p += sizeof(size_t) + size;
      }
      return reinterpret_cast<const T*>(p+sizeof(size_t));
    }

  private:
    const void* buf_;
    uintptr_t pVariable_;
    struct Header {
      int64_t nArguments;
    };
    const Header* pHeader_;

    size_t tensor_size_;
};

int op_Kernel(const void* args, size_t len,
              int (*func)(const VEOpArgs&),
              const char* name)
{
  LOG(LOG_TRACE) << name << " begin";
  int ret = 1;

  VEOpArgs tmp(args);

  LOG(LOG_PARAM) << name << ": nVariable=" << tmp.nArguments();

  // TODO: check length

  if (func)
    ret = func(tmp);

  LOG(LOG_TRACE) << name << " end. ret=" << ret;
  return ret;
}

}



#endif /* VE_OPS_COMMON_H_ */
