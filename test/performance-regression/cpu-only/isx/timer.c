/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
    * Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

bool first_call = true;

char const * const timer_names[] = {
  [TIMER_TOTAL]                         = "TOTAL",
  [TIMER_ATA_KEYS]                      = "ATA_KEYS",
  [TIMER_INPUT]                         = "MAKE_INPUT",
  [TIMER_BCOUNT]                        = "COUNT_BUCKET_SIZES",
  [TIMER_BUCKETIZE]                     = "BUCKETIZE",
  [TIMER_BOFFSET]                       = "COMPUTE_OFFSETS",
  [TIMER_SORT]                          = "LOCAL_SORT",
  [TIMER_BARRIER_START]                 = "BARRIER_START",
  [TIMER_BARRIER_EXCHANGE]              = "BARRIER_EXCHANGE",
  [TIMER_BARRIER_END]                   = "BARRIER_END"
};

_timer_t timers[TIMER_NTIMERS];

void timer_reset(_timer_t * const timer, const unsigned int num_iters)
{
  for(unsigned int i = 0; i < num_iters; ++i) {
    timer->seconds[i] = 0.0;
    timer->count[i] = 0;
  }
  free(timer->all_times);
  free(timer->all_counts);
  timer->num_iters = num_iters;
  timer->seconds_iter = 0;
  timer->count_iter = 0;
  timer->start.tv_sec = 0;
  timer->start.tv_nsec = 0;
  timer->stop.tv_sec = 0;
  timer->stop.tv_nsec = 0;
}

void init_timers(const unsigned int num_iters)
{
  for(int t = 0; t < TIMER_NTIMERS; ++t){
    if(first_call == true) {
      timers[t].seconds = (double *) malloc(num_iters * sizeof(double));
      assert(timers[t].seconds);
      timers[t].count = (unsigned int *) malloc(num_iters * sizeof(unsigned int));
      assert(timers[t].count);
      timers[t].all_times = NULL;
      timers[t].all_counts = NULL;
    }
    timer_reset(&timers[t], num_iters);
  }
  first_call = false;
}

void report_times(void)
{
  printf("\n");
  printf("Average Timing Info -----------------------------------\n");
  for(int t = 0; t < TIMER_NTIMERS; ++t){
    if(timers[t].seconds_iter > 0){
      printf("%s\t", timer_names[t]);
      for(unsigned int i = 0; i < timers[t].seconds_iter; ++i){
        printf("%3.5f\t", timers[t].seconds[i]);
      }
      printf("\n");
    }

    if(timers[t].count_iter > 0){
      printf("%s_COUNTS\t", timer_names[t]);
      for(unsigned int i = 0; i < timers[t].count_iter; ++i){
        printf("%d\t", timers[t].count[i]);
      }
      printf("\n");
    }
  }
}


void timer_start(_timer_t * const timer)
{
#ifdef __MACH__
  // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  timer->start.tv_sec = mts.tv_sec;
  timer->start.tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_MONOTONIC, &(timer->start));
#endif
}

void just_timer_stop(_timer_t * const timer)
{
#ifdef __MACH__
  // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  timer->stop.tv_sec = mts.tv_sec;
  timer->stop.tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_MONOTONIC, &(timer->stop));
#endif
}

void timer_stop(_timer_t * const timer)
{
  just_timer_stop(timer);
  timer->seconds[timer->seconds_iter] = (double) (timer->stop.tv_sec - timer->start.tv_sec);
  timer->seconds[timer->seconds_iter] += (double) (timer->stop.tv_nsec - timer->start.tv_nsec)*1e-9;
  timer->seconds_iter++;
}

void timer_count(_timer_t * const timer, const unsigned int val)
{
  timer->count[timer->count_iter] = val;
  timer->count_iter++;
}

