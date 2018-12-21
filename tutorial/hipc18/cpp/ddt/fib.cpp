#include "hclib_cpp.h"
#include <inttypes.h>

/*
 * Parallel fibonacci number calculation using promises, futures, and async_future
 */

#define THRESHOLD 10

/*Sequential execution once threshold is reached */
uint64_t fib_serial(uint64_t n) {
    if (n <= 2) return 1;
    return fib_serial(n-1) + fib_serial(n-2);
}

/* parallel execution */
void fib(uint64_t n, hclib::promise_t<uint64_t> *res) {
  if (n <= THRESHOLD) {
    uint64_t r = fib_serial(n);
    res->put(r);
    return;
  } 

  /* compute f1 asynchronously */
  hclib::promise_t<uint64_t> * f1 = new hclib::promise_t<uint64_t>();
  hclib::async([=]() { 
    fib(n - 1, f1);
  });

  /* compute f2 serially (f1 is done asynchronously). */
  hclib::promise_t<uint64_t>* f2 = new hclib::promise_t<uint64_t>();
  hclib::async([=]() { 
    fib(n - 2, f2);
  });

  /* wait for dependences, before updating the result */
  hclib::async_await([=] {
    uint64_t r = f1->get_future()->get() + f2->get_future()->get();
    res->put(r);
    delete(f1); delete(f2);
  }, f1->get_future(), f2->get_future());
}

int main(int argc, char** argv) {
  uint64_t n = argc>1?atoi(argv[1]) : 40;
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    long start = hclib_current_time_ms();
    hclib::promise_t<uint64_t> * promise = new hclib::promise_t<uint64_t>();
    hclib::finish([=] {
      fib(n, promise);
    });
    uint64_t res = promise->get_future()->get();
    delete(promise);
    long end = hclib_current_time_ms();
    double dur = ((double)(end-start))/1000;
    printf("Fibonacci of %" PRIu64 " is %" PRIu64 ".\n", n, res);
    printf("Time = %f\n",dur);
  });
  return 0;
}
