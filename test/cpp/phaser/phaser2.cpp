#include <stdio.h>
#include "hclib_cpp.h"
#include "phaser-api.h"

/**
 * DESC: SIGNAL_WAIT/TRANSMIT_ALL with main activity participating
 */

int ub = 100;

void barrier_test(void* arg) {
  int k;
  // Wait for all asyncs to be here
  printf("Enter barrier_test\n");
  phaser_next();
  printf("All ready\n");
  for (k=0; k<=ub; k++){
    phaser_next(); 
  }
  printf("Done\n");
}

int main (int argc, char ** argv) {
    hclib_init(&argc, argv);
    // TODO should get number of workers here
    int nthreads = 3;
    int degree = nthreads;
    int i;
    phaser_t ph;
    phaser_create(&ph, SIGNAL_WAIT, degree);
    for(i = 1; i < nthreads; ++i) {
        printf("Create async %d\n", i);
        async(&barrier_test, NULL, NULL, NULL, PHASER_TRANSMIT_ALL); 
    }
    // Participate in the barrier
    barrier_test(NULL);
    // current worker enters in helper mode
    // and will pick up one of the tasks
    hclib_finalize();
    return 0;
}
