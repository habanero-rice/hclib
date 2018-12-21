#include "hclib.h"
#include <inttypes.h>

/*
 * This example calculates Nth fibonacci number in parallel
 * by using C based interfaces of async and finish
 */

/*
 * Structure containing input and output arguments to fib
 */
typedef struct {
    uint64_t n;
    uint64_t res;
} FibArgs;

/*
 * Stop tasks creation beyond this threshold
 */
#define THRESHOLD 10

/*
 * Sequential implementation to invoke once threshold is reached
 */
uint64_t fib_seq(int n) {
  if (n < 2) {
    return n;
  }
  else {
    return fib_seq(n-1) + fib_seq(n-2);
  }
}

/*
 * Parallel implememtation of fib
 */
void fib(void * raw_args) {
  FibArgs *args = raw_args;
  if (args->n < THRESHOLD) { 
    args->res = fib_seq(args->n);
  } else {
    FibArgs lhsArgs = { args->n - 1, 0 };
    FibArgs rhsArgs = { args->n - 2, 0 };

    hclib_start_finish();
    hclib_async(fib, &lhsArgs, NO_FUTURE, 0, ANY_PLACE);
    fib(&rhsArgs);
    hclib_end_finish();
    args->res = lhsArgs.res + rhsArgs.res;
  }
}

/*
 * Method to invoke from hclib_launch
 */
void taskMain(void* raw_args) {
  int n = *((int*)raw_args);
  long start = hclib_current_time_ms();
  FibArgs args = { n, 0 };
  fib(&args);
  long end = hclib_current_time_ms();
  double dur = ((double)(end-start))/1000;
  printf("Fibonacci of %" PRIu64 " is %" PRIu64 ".\n", args.n, args.res);
  printf("Time = %f\n",dur);
}
  
int main (int argc, char ** argv) {
  const int n = argc>1?atoi(argv[1]):40;
  char const *deps[] = { "system" }; 
  hclib_launch(taskMain, (void*)&n, deps, 1);;
  return 0;
}
