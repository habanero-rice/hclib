#include "hclib_cpp.h"
#include <inttypes.h>

/*
 * Parallel fibonacci number calculation using async_future
 */

#define THRESHOLD 10

/* sequential computation */
uint64_t fib_serial(uint64_t n) {
    if (n < 2) return n;
    return fib_serial(n-1) + fib_serial(n-2);
}

uint64_t fib(uint64_t n) {
  if (n < THRESHOLD) {
    return fib_serial(n);
  } 

  /* compute f1 asynchronously */
  hclib::future_t<uint64_t>* f1 = hclib::async_future([=]() { 
    uint64_t x = fib(n - 1);
    return x;
  });

  uint64_t y = fib(n - 2);
  /* wait for dependences, before updating the result */
  return y + f1->wait();
}

int main(int argc, char** argv) {
  uint64_t n = argc>1?atoi(argv[1]) : 40;
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    long start = hclib_current_time_ms();
    uint64_t result = fib(n);
    long end = hclib_current_time_ms();
    double dur = ((double)(end-start))/1000;
    printf("Fibonacci of %" PRIu64 " is %" PRIu64 ".\n", n, result);
    printf("Time = %f\n",dur);
  });
  return 0;
}
