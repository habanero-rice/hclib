#include "hclib_cpp.h"
#include <inttypes.h>

/* Parallel fibonacci number calculation using async-finish */

#define THRESHOLD 10

uint64_t fib_seq(int n) {
  if (n < 2) {
    return n;
  }
  else {
    return fib_seq(n-1) + fib_seq(n-2);
  }
}

uint64_t fib(uint64_t n) {
  if (n < THRESHOLD) { 
    return fib_seq(n);
  } else {
    uint64_t x, y;
    /* start the finish scope*/
    /* Note that this finish is going to create another finish scope (nested finish) */
    hclib::finish([&]() {
      /* Spawn a nested async */
      hclib::async([&x, n]() {
        x = fib(n-1);
      });
      y = fib(n-2);
    }); 
    return (x + y);
  }
}

int main (int argc, char ** argv) {
  char const *deps[] = { "system" }; 
  uint64_t n = argc>1?atoi(argv[1]) : 40;
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
