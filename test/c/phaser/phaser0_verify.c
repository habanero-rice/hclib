/***************************************************************************
 * *                                                                         *
 * *             OpenMP MicroBenchmark Suite - Version 2.0                   *
 * *                                                                         *
 * *                            produced by                                  *
 * *                                                                         *
 * *                     Mark Bull and Fiona Reid                            *
 * *                                                                         *
 * *                                at                                       *
 * *                                                                         *
 * *                Edinburgh Parallel Computing Centre                      *
 * *                                                                         *
 * *         email: markb@epcc.ed.ac.uk or fiona@epcc.ed.ac.uk               *
 * *                                                                         *
 * *                                                                         *
 * *      This version copyright (c) The University of Edinburgh, 2004.      *
 * *                         All rights reserved.                            *
 * *                                                                         *
 * **************************************************************************/

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <sys/time.h>
#include "hclib.h"
#include "phaser-api.h"

#define OUTERREPS 20
#define CONF95 1.96 


   int nthreads, delaylength, innerreps, degree; 
   long shared_val;

   void syncdelay(int);

   void checkbar(); 
   void testbar(); 
   void testnobar(); 

   typedef struct _data {
     char pad1[64];
     int val;
     char pad2[64];
   } data_t;
   data_t * gData;

int main (int argc, char **argv)
{ 
   if (argc != 1) {
    nthreads = atoi(argv[1]);
    degree = nthreads; // flat phaser by default
    if (argc > 2)
      degree = atoi(argv[2]);

    delaylength = 0;
    innerreps = 1000000;

    if (argc > 3)
      delaylength = atoi(argv[3]);
  } else {
    // TODO should get number of workers here
    int nthreads = 3;
    degree = nthreads; // flat phaser by default
    delaylength = 0;
    innerreps = 1000000;
  }

  printf(" Running phaser barrier benchmark on %d thread(s) and degree %d\n", nthreads, degree); 

  hclib_init(&argc, argv);
  /* TEST  BARRIER */ 
  checkbar(); 

  // /* TEST  BARRIER */ 
  testbar(); 

  // /* TEST NO BARRIER */ 
  testnobar(); 
  hclib_finalize();
  return 0;
} 

void barrier_test(void* arg) {
  int i,j,k; 
  int tid = *((int *)arg);
  phaser_next(); 

  for (k=0; k<=OUTERREPS; k++){
    if (tid == 0) {
      shared_val = 0;
    }
    phaser_next(); 

    for (j=0; j<nthreads; j++){
      if ((long)j == tid) {
        for (i=0; i<innerreps; i++) { 
          shared_val++;
        }
      }
      syncdelay(delaylength); 
      phaser_next(); 
    }     

    if (tid == 0) {
      printf("Result: %ld\n", shared_val);
    }
    phaser_next(); 
  }
}

void testbar()
{
  long i;
  int indices[nthreads];
  start_finish();
  phaser_t ph;
  phaser_create(&ph, SIGNAL_WAIT, degree);

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Verifying PHASER BARRIER (EXPECTED RESULT: %d)\n", nthreads * innerreps); 

  for(i = 1; i < nthreads; ++i)
  {
    indices[i] = i;
    async(&barrier_test, (void*) (indices+i), NULL, NULL, PHASER_TRANSMIT_ALL); 
  }
  indices[0] = 0;
  barrier_test((void *) indices);
  end_finish();
  phaser_drop(ph);
}

void no_barrier_test(void* arg) {
  int i,j,k; 
  int tid = *((int *)arg);

  phaser_next(); 

  for (k=0; k<=OUTERREPS; k++){
    if (tid == 0) {
      shared_val = 0;
    }
    phaser_next(); 

    for (j=0; j<nthreads; j++){
      if ((long)j == tid) {
        for (i=0; i<innerreps; i++) { 
          shared_val++;
        }
      }
      syncdelay(delaylength); 
    }     

    phaser_next(); 
    if (tid == 0) {
      printf("Result: %ld\n", shared_val);
    }
    phaser_next(); 
  }
}

void testnobar()
{
  long i;
  int indices[nthreads];

  phaser_t ph;
  phaser_create(&ph, SIGNAL_WAIT, degree);
  start_finish();

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Verifying PHASER NO BARRIER (EXPECTED RESULT: RACY)\n"); 

  for(i = 1; i < nthreads; ++i)
  {
    indices[i] = i;
    async(&no_barrier_test, (void*) (indices+i), NULL, NULL, PHASER_TRANSMIT_ALL); 
  }
  indices[0] = 0;
  no_barrier_test((void *) indices);
  end_finish();
  phaser_drop(ph);
}

void barrier_check(void* arg) {
  int i,j,k; 
  int tid = *((int *)arg);
  for (k=0; k<=innerreps; k++){

    for (i=0; i<OUTERREPS; i++) { 
      gData[tid].val++;
    }

    phaser_next(); 

    for (j=0; j<nthreads; j++){
      if (gData[j].val != gData[tid].val) {
        printf("BARRIER CHECK VERIFICATION FAILED\n"); 
        exit(1);
      }
    }     

    phaser_next(); 
  }
}

void checkbar()
{
  long i;
  //int indices[nthreads];
  int * indices = (int *) malloc(nthreads * sizeof(int));
  printf("nthreads %d\n", nthreads);
  phaser_t ph;
  phaser_create(&ph, SIGNAL_WAIT, degree);
  start_finish();
  gData = (data_t*) malloc(sizeof(data_t) * nthreads);
  for (i = 0; i < nthreads; i++) gData[i].val = 0;

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Verifying PHASER BARRIER (NON VISUAL)\n"); 
  for(i = 1; i < nthreads; ++i)
  {
    indices[i] = i;
    async(&barrier_check, (void*) (indices+i), NULL, NULL, PHASER_TRANSMIT_ALL); 
  }
  indices[0] = 0;
  barrier_check((void *) indices);
  end_finish();
  phaser_drop(ph);
  printf("Verification PASSED\n"); 
}

void syncdelay(int delaylength)
{

   int  i; 
   float a=0.; 

   for (i=0; i<delaylength; i++) a+=i; 
   if (a < 0) printf("%f \n",a); 

} 

