#pragma once

#ifdef USE_PROFILE

#include <cstdlib>
#include <string>
#include <set>

namespace {

inline unsigned long long __veperf_get_stm() {
        void *vehva = (void *)0x1000;
        unsigned long long val;
        asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
        return val;
}

inline double second() 
{
  return __veperf_get_stm() / 800e6;
}

std::set<std::string> init_prof_enabled()
{
  std::set<std::string> s;
  if (char* p = getenv("VML_PROF")) {
    std::string tmp;
    for (;;) {
      if (*p == ',' || *p == '\0') {
        s.insert(tmp);
        tmp.clear();
      } else {
        tmp += *p;
      }
      if (*p == '\0')
        break;
      ++p;
    }
  }
  return s;
}

bool is_prof_enabled(std::string const& name) { 
  static std::set<std::string> enabled = init_prof_enabled();
  return enabled.find(name) != enabled.end();
}

} // namespace

#define PROF_BEGIN(name) \
    double __prof_##name = second()

#define PROF_END(name) \
  if (is_prof_enabled(#name)) LogMessage() \
       << #name << ": " << second() - __prof_##name

#else // USE_PROFILE

#define PROF_BEGIN(name)
#define PROF_END(name) \
  if (false) LogMessage()

#endif // USE_PROFILE
