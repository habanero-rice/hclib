/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include "hclib_cpp.h"
#include "phaser-api.h"

/**
 * DESC: SIGNAL_WAIT/TRANSMIT_ALL test with main activity dropping
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
    for(i = 0; i < nthreads; ++i) {
        printf("Create async %d\n", i);
        async(&barrier_test, NULL, NULL, NULL, PHASER_TRANSMIT_ALL); 
    }
    printf("Dropping\n");
    // Dropping here unlocks child activities next
    phaser_drop(ph);
    // current worker enters in helper mode
    // and will pick up one of the tasks
    hclib_finalize();
    return 0;
}
