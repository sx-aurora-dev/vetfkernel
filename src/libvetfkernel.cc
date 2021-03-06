#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

#include <sched.h>
#include <omp.h>

#ifdef USE_VEDNN
#include <vednn.h>
#if VEDNN_VERSION < 1001
#error Require vednn version 1.0.1 or higher
#endif
#endif

#include <vml.h>
#include "kernel.h"
#include "vml/log.h"
#include "revision.h"

#define MAX_KERNEL 1024

#ifdef __ve__
#define USE_DMA
#ifdef USE_DMA
extern "C" {
#include <vhshm.h>
#include <vedma.h>
}
#endif
#endif // __ve__

#define _STR(v) __STR(v)
#define __STR(v) #v

#define PROF
#ifdef PROF
static bool _is_profiler_enabled = false;
#endif


class InitVETFKernel
{
public :
  InitVETFKernel() {
    LOG(LOG_TRACE) << "InitVETFKernel";
    LOG(LOG_INFO) << "vetfkernel revision: " << VETFKERNEL_REVISION;
    LOG(LOG_INFO) << "vednn revision: " << VEDNN_REVISION;
    setaffinity() ;
    vml::initialize();

#ifdef PROF
    if (const char* tmp = getenv("VE_PROF")) {
      if (tmp[0] == '1')
        _is_profiler_enabled = true;
      LOG(LOG_PROFILE) << "VE_PROF=" << tmp << " _is_profiler_enabled=" << _is_profiler_enabled;
#endif
    }
  }

  ~InitVETFKernel() {
    LOG(LOG_TRACE) << "~InitVETFKernel";
    vml::finalize();
  }

private :
  const int num_cores = 8 ;

  void setaffinity() {
    int core_offset = 0 ;
    if (const char* tmp = getenv("TF_VE_CORE_OFFSET")) {
      core_offset = atoi(tmp);
    }

    int num_threads ;
#pragma omp parallel
    {
#pragma omp_master
      {
	num_threads = omp_get_num_threads() ;
      }
    }

    cpu_set_t mask[num_threads] ;
    // ncc does not create parallel routine in the function using CPU_SET...
    set_mask(num_threads, core_offset, mask) ;


#pragma omp parallel
    {
      int threadid = omp_get_thread_num() ;
      sched_setaffinity(0, sizeof(cpu_set_t), &mask[threadid] ) ;
    }
  }

  void set_mask(int num_threads, int core_offset, cpu_set_t* mask)
  {
    for(int i=0; i<num_threads ; i++) {
      int coreid   = (core_offset + i) % num_cores ;

      CPU_ZERO(&mask[i]) ;
      CPU_SET(coreid, &mask[i]) ;
    }
  }


} _InitVETFKernel ;


extern "C" {
    int get_num_kernels();
    uint64_t get_kernel_table_addr();

    uint64_t vetfkl_entry(const void* arg, size_t len);
    uint64_t vetfkl_entry_prof(const void* argIn, size_t lenIn, void* argOut, size_t lenOut);
    int vetfkl_get_timestamp(void* arg, size_t len);

#ifdef USE_DMA
    int vetfkl_dma_init(void* arg, size_t len);
    int vetfkl_dma_read(void* arg, size_t len);
#endif

    int op_Assign(const void* arg, size_t len);

}

struct Kernel
{
    char name[256];
    char func[256];
} table_[MAX_KERNEL];

static int kernel_index = 0;

int get_num_kernels()
{
    return kernel_index;
}

uint64_t get_kernel_table_addr()
{
    return (uint64_t)table_;
}

void register_kernel(char const* name, char const* func)
{
    // FIXME: check MAX_KERNEL
  LOG(LOG_INFO) << __FUNCTION__ << ":"
    << " kernel_index=" << kernel_index
    << " kernel_name=" << name
    << " func=" << func;
  Kernel& k = table_[kernel_index++];

  strcpy(k.name, name);
  strcpy(k.func, func);
}

#ifdef __ve__
static inline unsigned long int __ve_get_usrcc() {
  unsigned long int val;
  asm volatile("smir %0,%usrcc\n" : "=r"(val));
  return val;
}

static inline unsigned long long __veperf_get_stm() {
  void *vehva = (void *)0x1000;
  unsigned long long val;
  asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
  return val;
}

static inline unsigned long long get_timestamp() {
  return __veperf_get_stm();
}

static inline double get_resolution() {
  return 800e6;
}

int vetfkl_get_timestamp(void* arg, size_t len)
{
  struct tmp {
    uint64_t ts;
    double resolution;
  }* p = reinterpret_cast<tmp*>(arg);

  //fprintf(stderr, "vetfkl_get_timestamp: len=%lu sizeof(tmp)=%lu\n", len, sizeof(tmp));

  if (len < sizeof(tmp))
    return 1;

  p->ts = get_timestamp();
  p->resolution = get_resolution();

  //fprintf(stderr, "vetfkl_get_timestamp: resolution=%lf\n", p->resolution);

  return 0;

#if 0
#if 1
  *reinterpret_cast<uint64_t*>(arg) = get_timestamp();
  return 0;
#else
  uint64_t t = get_timestamp();
  fprintf(stderr, "vetfkl_get_timestamp: t=%lu\n", t);
  *reinterpret_cast<uint64_t*>(arg) = t;
  return 0;
#endif
#endif
}

uint64_t vetfkl_entry(const void* arg, size_t len)
{
#if 0
  fprintf(stderr, "vetfkl_entry: len=%lu\n", len);
#endif

  //const void* curr = arg;

  uint64_t end = reinterpret_cast<uintptr_t>(arg) + len;
  uintptr_t curr = reinterpret_cast<uintptr_t>(arg);

  int32_t num_kernels = *reinterpret_cast<int32_t*>(curr);
  curr += sizeof(int32_t);

  LOG(LOG_PARAM) << __FUNCTION__ << ": num_kernels=" << num_kernels;

#if 0
  fprintf(stderr, "%s: num_kernels=%d\n", __FUNCTION__, num_kernels);
#endif

  typedef int (*func_t)(const void* arg, size_t len);

  for (int i = 0; i < num_kernels; ++i) {
    uint64_t sym = *reinterpret_cast<uint64_t*>(curr);
    curr += sizeof(uint64_t);
    func_t func = reinterpret_cast<func_t>(sym);
    size_t len0 = *reinterpret_cast<size_t*>(curr);
    curr += sizeof(size_t);
    const void* arg0 = reinterpret_cast<const void*>(curr);
    curr += len0;

    LOG(LOG_TRACE) << __FUNCTION__ << ": i=" << i << "/" << num_kernels;
#if 0
    fprintf(stderr, "vetfkl_entry: i=%d/%d func=%p args0=%p len=%lu\n",
            i, num_kernels, func, arg0, len0);
#endif
#ifdef PROF
    uint64_t t0;
    if (_is_profiler_enabled)
      t0 = get_timestamp();
#endif
    int ret = func(arg0, len0);
#ifdef PROF
    if (_is_profiler_enabled) {
      uint64_t t1 = get_timestamp();
      LOG(LOG_PROFILE) << __FUNCTION__ << ": profile" 
          << " " << (void*)func
          << " " << (t1 - t0);
    }
#endif
    LOG(LOG_TRACE) <<  __FUNCTION__ << ": i=" << i << "/" << num_kernels << " ret=" << ret;
#if 0
    fprintf(stderr, "vetfkl_entry: ret=%d\n", ret);
#endif
    if (ret != 0) {
      LOG(LOG_ERROR) << __FUNCTION__ << ": return error. i=" << i << " ret=" << ret;
      uint64_t retval = ((uint64_t)i) << 32 | ret;
      return retval;
    }
  }

  LOG(LOG_TRACE) << __FUNCTION__ << ": end. ret=0";
#if 0
  fprintf(stderr, "vetfkl_entry: end\n");
#endif
  return 0;
}

#ifdef USE_DMA
struct {
    uint64_t vehva_ve;
    uint64_t vehva_vh;
    void* buf;
} DMABuffer_;
#if 0
void* vemva_ = NULL; // FIXME: rename
void* vehva_vh_;
uint64_t vehva_ve_;
#endif

int dma_init(int32_t shmid, size_t size)
{
    LOG(LOG_TRACE) << "dma_init: shmid=" << shmid << " size=" << size;

    void* buf;
    uint64_t vehva_ve;

    ve_dma_init();

    size_t align = 64 * 1024 * 1024;
    if (posix_memalign(&buf, align, size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }
    LOG(LOG_DETAIL) << "dma_init: buf=" << buf;
    vehva_ve = ve_register_mem_to_dmaatb(buf, size);
    LOG(LOG_DETAIL) << "dma_init: vehva_ve=" << (void*)vehva_ve;
    if (vehva_ve == (uint64_t)-1) {
        fprintf(stderr, "ve_register_mem_to_dmaatb failed\n");
        return 1;
    }

    void* vehva_vh;
    void* tmp = vh_shmat(shmid, NULL, 0, &vehva_vh);
    if (tmp == (void*)-1) {
        perror("vh_shmget");
        return 1;
    }

    DMABuffer_.buf = buf;
    DMABuffer_.vehva_ve = vehva_ve;
    DMABuffer_.vehva_vh = reinterpret_cast<uint64_t>(vehva_vh);

    // TODO: vh_shmdt

    return 0;
}

int vetfkl_dma_init(void* arg, size_t len)
{
    struct tmp {
        int32_t shmid;
        uint64_t size;
    }* p = reinterpret_cast<tmp*>(arg);

    return dma_init(p->shmid, p->size);
}

int vetfkl_dma_read(void* arg, size_t len)
{
    struct tmp {
        uint64_t size;
        uint64_t addr;
    } const* p = reinterpret_cast<tmp*>(arg);

    LOG(LOG_DETAIL) << __FUNCTION__ << ": size=" << p->size
        << " addr=" << reinterpret_cast<void const*>(p->addr);

    // TODO: 
    // 1. DMA to pre-allocated buffer, then memcpy to destination address.
    // 2. register destination address, and DMA, and unregister. 
    //    (Address have to be 64b aligned)
    // Currently 1 is used. which is faster?

    const size_t max_dma_size = 64 * 1024 * 1024; // have to be less than 128MB

    uint64_t dst_hva = DMABuffer_.vehva_ve;
    uint64_t src_hva = DMABuffer_.vehva_vh;
    uint64_t dst = reinterpret_cast<uint64_t>(p->addr);
    uint64_t src = reinterpret_cast<uint64_t>(DMABuffer_.buf);
    size_t size = p->size;

    while (size > 0) {
        size_t l = size > max_dma_size ? max_dma_size : size;
        LOG(LOG_DETAIL) << __FUNCTION__ << ": call ve_dma_post_wait. transfer size is " 
            << l << " bytes";
        int ret = ve_dma_post_wait(dst_hva, src_hva, l);
        if (ret != 0)
            return 1;
        memcpy(reinterpret_cast<void*>(dst),
               reinterpret_cast<void const*>(src), l);

        dst_hva += l;
        src_hva += l;
        dst += l;
        src += l;
        size -= l;
    }

#if 0
    fprintf(stderr, "%16llu %s: done\n", __veperf_get_stm(), __FUNCTION__);
#endif
    return 0;
}
#endif

uint64_t vetfkl_entry_prof(const void* argIn, size_t lenIn, 
                           void* argOut, size_t lenOut)
{
#if 0
  fprintf(stderr, "vetfkl_entry_prof: argIn=%p lenIn=%lu argOut=%p lenOut=%lu\n",
          argIn, lenIn, argOut, lenOut);
#endif

  uint64_t end = reinterpret_cast<uintptr_t>(argIn) + lenIn;
  uintptr_t curr = reinterpret_cast<uintptr_t>(argIn);

  int32_t num_kernels = *reinterpret_cast<int32_t*>(curr);
  curr += sizeof(int32_t);

#if 0
  fprintf(stderr, "%s: num_kernels=%d\n", __FUNCTION__, num_kernels);
#endif

  typedef int (*func_t)(const void* arg, size_t len);

  uint64_t* pcyc = nullptr;
  if (lenOut > 0) {
    pcyc = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(argOut));
  }

  for (int i = 0; i < num_kernels; ++i) {
    uint64_t sym = *reinterpret_cast<uint64_t*>(curr);
    curr += sizeof(uint64_t);
    func_t func = reinterpret_cast<func_t>(sym);
    size_t len0 = *reinterpret_cast<size_t*>(curr);
    curr += sizeof(size_t);
    const void* arg0 = reinterpret_cast<const void*>(curr);
    curr += len0;

#if 0
    fprintf(stderr, "vetfkl_entry: i=%d/%d func=%p args0=%p len=%lu\n",
            i, num_kernels, func, arg0, len0);
#endif
    uint64_t t0 = get_timestamp();
    int ret = func(arg0, len0);
    uint64_t t1 = get_timestamp();
    if (pcyc) {
      pcyc[i*2] = t0;
      pcyc[i*2+1] = t1;
    }
    //fprintf(stderr, "vetfkl_entry_prof: i=%d cyc=%llu\n", i, pcyc[i]);
#if 0
    fprintf(stderr, "vetfkl_entry: ret=%d\n", ret);
#endif
    if (ret != 0) {
      uint64_t retval = ((uint64_t)i) << 32 | ret;
      return retval;
    }
  }

#if 0
  fprintf(stderr, "vetfkl_entry: end\n");
#endif

  return 0;
}

#endif // __ve__
