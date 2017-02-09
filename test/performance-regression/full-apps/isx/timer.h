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
#ifndef TIMER_H
#define TIMER_H

#define _GNU_SOURCE
#include <time.h>
#include <stdbool.h>

typedef struct
{
  double * seconds;
  double * all_times;
  unsigned int * count;
  unsigned int * all_counts;
  unsigned int num_iters;
  unsigned int seconds_iter;
  unsigned int count_iter;
  struct timespec start;
  struct timespec stop;
} _timer_t;

typedef enum
{
  TIMER_TOTAL,
  TIMER_ATA_KEYS,
  TIMER_INPUT,
  TIMER_BCOUNT,
  TIMER_BUCKETIZE,
  TIMER_BOFFSET,
  TIMER_SORT,
  TIMER_BARRIER_START,
  TIMER_BARRIER_EXCHANGE,
  TIMER_BARRIER_END,
  //
  // Place new timers above and update timer_names[]
  TIMER_NTIMERS
}timer_id;

extern _timer_t timers[];
extern char const * const timer_names[];

void init_timers(const unsigned int num_iters);

void report_times(void);

void timer_start(_timer_t * const timer);

void just_timer_stop(_timer_t * const timer);

void timer_stop(_timer_t * const timer);

void timer_count(_timer_t * const timer, const unsigned int val);

void timer_reset(_timer_t * const timer, const unsigned int num_iters);

#endif
