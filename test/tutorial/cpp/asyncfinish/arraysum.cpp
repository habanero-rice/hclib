#include "hclib_cpp.h"
#include <sys/time.h>

int SIZE=10000000; //10M
int* array;

// TIMING HELPER FUNCTIONS
long get_usecs ()
{
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

// allocate and initialize the array
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
    long start = get_usecs();
    // start the finish scope
    hclib::finish([&]() {
      // spawn an async
      hclib::async([&]() {
        for(int i=0; i<SIZE/2; i++) {
          sum1 += 1/array[i];
        }
      });
      // This computation will happen in parallel 
      // to above if there are two HCLIB_WORKERS
      for(int i=SIZE/2; i<SIZE; i++) {
        sum2 += 1/array[i];
      }

    }); //end of finish, i.e. wait until all the asyncs inside this finish scope have terminated.
    double sum = sum1 + sum2;

    long end = get_usecs();
    double dur = ((double)(end-start))/1000000;
    const char* pass = sum==SIZE ? "true" : "false";
    printf("Test PASSED = %s\n",pass);
    printf("Time = %f\n",dur);
    delete array;
  });
  return 0;
}
