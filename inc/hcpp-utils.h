#include <iostream>
#include <stdlib.h>
#include <string.h>
// Required for lambda as std::function
#include <list>
#include <functional>

extern "C" {
  #include "hclib.h"
  #include "runtime-hclib.h"
  #ifdef _PHASER_LIB_
  #include "phased.h"
  #endif
}

typedef struct ddf_st DDF_t;
#define ASYNC_COMM ((int) 0x2)
#define __FORASYNC__

#ifdef _PHASER_LIB_
typedef phaser_t PHASER_t;
typedef phaser_mode_t PHASER_m;
#endif

namespace hcpp {
  void finish(std::function<void()> lambda);
  void async_cpp_wrapper(void *args);
  void async(std::function<void()> &&lambda);
  void asyncComm(std::function<void()> &&lambda);
  int numWorkers();
  void forasync(int lowBound, int highBound, std::function<void(int)> lambda);
  DDF_t** DDF_LIST(int total_ddfs);
  DDF_t* DDF_CREATE();
  void DDF_PUT(DDF_t* ddf, void* value);
  void* DDF_GET(DDF_t* ddf);
#ifdef _PHASER_LIB_
  PHASER_t* PHASER_CREATE(PHASER_m mode);
  void DROP(PHASER_t* ph);
  void NEXT();
#endif
  void hcpp_lock();
  void hcpp_unlock();
}

#include "hcpp-asyncAwait.h"
#ifdef _PHASER_LIB_
#include "hcpp-asyncPhased.h"
#endif
