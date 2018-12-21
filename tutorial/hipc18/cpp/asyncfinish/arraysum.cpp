#include "hclib_cpp.h"

/*
 * Parallel array sum using async-finish
 */

/* Program input */
int SIZE=10000000; /*10M*/
int* array;

/* 
 * allocate and initialize the array
 */
void initialize() {
  array = new int[SIZE];
  for(int i=0; i<SIZE; i++) array[i] = 1;
}

int main(int argc, char **argv) {

  if(argc>1) SIZE=atoi(argv[1]);
  
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    initialize();
    double sum1 =0, sum2 = 0;
    long start = hclib_current_time_ms();
    /* start the finish scope */
    hclib::finish([&]() {
      /* spawn an async */
      hclib::async([&]() {
        for(int i=0; i<SIZE/2; i++) {
          sum1 += 1/array[i];
        }
      });
      /* 
       *This computation will happen in parallel 
       * to above if there are two HCLIB_WORKERS 
       */
      for(int i=SIZE/2; i<SIZE; i++) {
        sum2 += 1/array[i];
      }

    }); /*end of finish, i.e. wait until all the asyncs inside this finish scope have terminated. */
    double sum = sum1 + sum2;

    long end = hclib_current_time_ms();
    double dur = ((double)(end-start))/1000;
    const char* pass = sum==SIZE ? "true" : "false";
    printf("Test PASSED = %s\n",pass);
    printf("Time = %f\n",dur);
    delete array;
  });
  return 0;
}
