
#include <stdio.h>
#include "hclib_cpp.h"
#include "phased.h"

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
    printf("Hello\n");
    hclib_init(&argc, argv);
    phaser_t ph;
    phaser_create(&ph, SIGNAL_WAIT, 2);
    phaser_mode_t mode = SIGNAL_WAIT;
    phased_t phased;
    phased.count = 1;
    phased.phasers = &ph;
    phased.phasers_mode = &mode;
    async(&barrier_test, NULL, NULL, &phased, 0); 
    // Participate in the barrier
    barrier_test(NULL);
    // current worker enters in helper mode
    // and will pick up one of the tasks
    hclib_finalize();
    return 0;
}
