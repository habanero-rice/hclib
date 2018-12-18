#include "hclib_cpp.h"
#include <inttypes.h>
#include <sys/time.h>

#define THRESHOLD 10

// TIMING HELPER FUNCTIONS
long get_usecs () {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}

uint64_t fib_serial(uint64_t n) {
    if (n < 2) return n;
    return fib_serial(n-1) + fib_serial(n-2);
}

uint64_t fib(uint64_t n) {
  if (n < THRESHOLD) {
    return fib_serial(n);
  } 

  // compute f1 asynchronously
  hclib::future_t<uint64_t>* f1 = hclib::async_future([=]() { 
    uint64_t x = fib(n - 1);
    return x;
  });

  uint64_t y = fib(n - 2);
  // wait for dependences, before updating the result
  return y + f1->wait_and_get();
}

int main(int argc, char** argv) {
  uint64_t n = argc>1?atoi(argv[1]) : 40;
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    long start = get_usecs();
    uint64_t result = fib(n);
    long end = get_usecs();
    double dur = ((double)(end-start))/1000000;
    printf("Fibonacci of %" PRIu64 " is %" PRIu64 ".\n", n, result);
    printf("Time = %f\n",dur);
  });
  return 0;
}
