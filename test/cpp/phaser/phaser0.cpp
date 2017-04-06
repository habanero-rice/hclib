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
#include <stdlib.h> 
#include <math.h> 
#include <sys/time.h>
#include "hclib.hpp"
#include "phaser-api.h"

#define OUTERREPS 10
#define CONF95 1.96 


volatile int degree;
int nthreads, delaylength, innerreps; 
double times[OUTERREPS+1], reftime, refsd; 

   time_t starttime = 0; 
   int firstcall = 1; 

   void syncdelay(int);
   void refer(void); 

   void testbar(); 

   void stats(double*, double*); 

   double get_time_of_day_(void);
   void init_time_of_day_(void);
   double getclock(void);

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

  printf(" Running phaser barrier benchmark on %d thread(s) and phaser tree degree %d\n", nthreads, degree); 

  /* GENERATE REFERENCE TIME */ 
  refer();   

  /* TEST BARRIER */
  hclib_init(&argc, argv);
  testbar();
  hclib_finalize();

  return 0;
} 

void refer()
{
  int j,k; 
  double meantime, sd; 

  double getclock(void); 

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Computing reference time 1\n"); 

  for (k=0; k<=OUTERREPS; k++){
    const double start  = getclock(); 
    for (j=0; j<innerreps; j++){
      syncdelay(delaylength); 
    }
    times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
  }

  stats (&meantime, &sd);

  printf("Reference_time_1 =                        %f microseconds +/- %f\n", meantime, CONF95*sd);

  reftime = meantime;
  refsd = sd;  
}

void barrier_test(void * arg) {
  int j,k; 
  double start = 0.0; 
  int tid = *((int *)arg);
  phaser_next(); 

  for (k=0; k<=OUTERREPS; k++){
    if (tid == 0) {
      start  = getclock();
    }

    for (j=0; j<innerreps; j++){
      syncdelay(delaylength); 
      phaser_next(); 
    }     

    if (tid == 0) {
      times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
    }

    phaser_next(); 
  } 
}

void testbar()
{
  long i;
  double meantime, sd; 
  int indices[nthreads];

  phaser_t ph;
  phaser_create(&ph, SIGNAL_WAIT, degree);
  start_finish();

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Computing PHASER BARRIER time\n"); 

  for(i = 1; i < nthreads; ++i)
  {
    indices[i] = i;
    async(&barrier_test, (void *) (indices+i), NULL, NULL, PHASER_TRANSMIT_ALL); 
  }
  indices[0] = 0;
  barrier_test((void *) indices);
  end_finish();
  phaser_drop(ph); 

  stats (&meantime, &sd);

  printf("BARRIER time =                           %f microseconds +/- %f\n", meantime, CONF95*sd);
  
  printf("BARRIER overhead =                       %f microseconds +/- %f\n", meantime-reftime, CONF95*(sd+refsd));

}


void stats (double *mtp, double *sdp) 
{

  double meantime, totaltime, sumsq, mintime, maxtime, sd, cutoff; 

  int i, nr; 

  mintime = 1.0e10;
  maxtime = 0.;
  totaltime = 0.;

  for (i=1; i<=OUTERREPS; i++){
    mintime = (mintime < times[i]) ? mintime : times[i];
    maxtime = (maxtime > times[i]) ? maxtime : times[i];
    totaltime +=times[i]; 
  } 

  meantime  = totaltime / OUTERREPS;
  sumsq = 0; 

  for (i=1; i<=OUTERREPS; i++){
    sumsq += (times[i]-meantime)* (times[i]-meantime); 
  } 
  sd = sqrt(sumsq/(OUTERREPS-1));

  cutoff = 3.0 * sd; 

  nr = 0; 
  
  for (i=1; i<=OUTERREPS; i++){
    if ( fabs(times[i]-meantime) > cutoff ) nr ++; 
  }
  
  printf("\n"); 
  printf("Sample_size       Average     Min         Max          S.D.          Outliers\n");
  printf(" %d                %f   %f   %f    %f      %d\n",OUTERREPS, meantime, mintime, maxtime, sd, nr); 
  printf("\n");

  *mtp = meantime; 
  *sdp = sd; 

} 

double get_time_of_day_()
{

  struct timeval ts; 

  double t;

  gettimeofday(&ts, NULL); 

  t = (double) (ts.tv_sec - starttime)  + (double) ts.tv_usec * 1.0e-6; 
 
  return t; 

}

void init_time_of_day_()
{
  struct  timeval  ts;
  gettimeofday(&ts, NULL);
  starttime = ts.tv_sec; 
}


double getclock()
{
      double time;

      double get_time_of_day_(void);  
      void init_time_of_day_(void);      

      if (firstcall) {
         init_time_of_day_(); 
         firstcall = 0;
      }
      time = get_time_of_day_(); 
      return time;

} 

void syncdelay(int delaylength)
{

   int  i; 
   float a=0.; 

   for (i=0; i<delaylength; i++) a+=i; 
   if (a < 0) printf("%f \n",a); 

} 

