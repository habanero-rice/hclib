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
    // start the finish scope
    // Note that this finish is going to create another finish scope (nested finish)
    hclib::finish([&]() {
      //Spawn a nested async
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
    long start = get_usecs();
    uint64_t result = fib(n);
    long end = get_usecs();
    double dur = ((double)(end-start))/1000000;
    printf("Fibonacci of %" PRIu64 " is %" PRIu64 ".\n", n, result);
    printf("Time = %f\n",dur);
  });
  return 0;
}
