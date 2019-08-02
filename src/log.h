#ifndef LOG_H
#define LOG_H

#include <sstream>

static int init_min_log_level()
{
  if (const char* tmp = getenv("VE_LOG_LEVEL")) {
    return atoi(tmp);
  }
  return 0;
}

class LogMessage : public std::basic_ostringstream<char> {
  public:
    LogMessage() {}
    ~LogMessage() {
      fprintf(stderr, "%s\n", str().c_str());
    }

    static int getMinLogLevel() {
      static int min_log_level = init_min_log_level();
      return min_log_level;
    }
};

/*
 * Log level
 */
#define LOG_PROFILE	0
#define LOG_TIMER	0
#define LOG_MESSAGE	0	// starup message
#define LOG_ERROR	1
#define LOG_WARNING	2
#define LOG_INFO	3
#define LOG_TRACE	4	// kernel begin and end
#define LOG_PARAM	5	// kernel parameter
#define LOG_DETAIL	6	// more detail infomation
#define LOG_DEBUG	7


#ifndef NDEBUG
#define LOG(lvl) \
  if ((lvl) <= LogMessage::getMinLogLevel()) LogMessage()

#else
#define LOG(lvl) \
  if (false) LogMessage()
#endif

#endif
