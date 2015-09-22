/* Copyright (c) 2015, Rice University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
3.  Neither the name of Rice University
     nor the names of its contributors may be used to endorse or
     promote products derived from this software without specific
     prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

/*
 * hcpp-internal.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_INTERNAL_H_
#define HCPP_INTERNAL_H_

#include "hcpp.h"
#include <stdarg.h>
#include "hcpp-deque.h"

namespace hcpp {

#define LOG_LEVEL_FATAL         1
#define LOG_LEVEL_WARN          2
#define LOG_LEVEL_INFO          3
#define LOG_LEVEL_DEBUG         4
#define LOG_LEVEL_TRACE         5

/* set the current log level */
#define LOG_LEVEL LOG_LEVEL_FATAL

#define WHEREARG __FILE__,__LINE__

#define LOG_(level, ...) if (level<=LOG_LEVEL) log_(WHEREARG, current_ws_internal(), __VA_ARGS__);

/* We more or less mimic log4c without the ERROR level */
#define LOG_FATAL(...)  LOG_(LOG_LEVEL_FATAL, __VA_ARGS__)
#define LOG_WARN(...)   LOG_(LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_INFO(...)   LOG_(LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_DEBUG(...)  LOG_(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_TRACE(...)  LOG_(LOG_LEVEL_TRACE, __VA_ARGS__)

/* log the msg using the fatal logger and abort the program */
#define log_die(... ) { LOG_FATAL(__VA_ARGS__); abort(); }
#define check_log_die(cond, ... ) if(cond) { log_die(__VA_ARGS__) }

typedef struct hc_options {
	char * hpt; /* the file name for hpt specification */
	int nproc; /* number of physical processors */
	int nworkers; /* number of workers, one per hardware core, plus workers for device (GPU) (one per device) */
#ifdef __USE_HC_MM__
	int alloc_batch_size; /* the number of memory segment to be move between global buckets and worker buckets if needed in malloc/free */
#endif
} hc_options;

typedef struct hc_context {
	/* for hc memory management */
#ifdef __USE_HC_MM__
	hc_mm_context mm_context;
#endif
	struct hc_workerState** workers;
	hc_options * options;
	place_t ** places; /* all the places */
	place_t * hpt;
	int nworkers;
	int nplaces;
	int nproc; /* the number of hardware core of the runtime */
	volatile int workers_wait_cond; /* a simple implementation of wait/wakeup condition */
	volatile int done;
} hc_context;

typedef struct finish_t {
	struct finish_t* parent;
	volatile int counter;
} finish_t;

typedef struct asyncAnyInfo {
	int asyncAny_pushed;
	volatile int asyncAny_stolen;
} asyncAnyInfo;

typedef struct hc_deque_t {
	/* The actual deque, WARNING: do not move declaration !
	 * Other parts of the runtime rely on it being the first one. */
	deque_t deque;
	struct hc_workerState * ws;
	struct hc_deque_t * nnext;
	struct hc_deque_t * prev; /* the deque list of the worker */
	struct place_t * pl;
} hc_deque_t;

void log_(const char * file, int line, hc_workerState * ws, const char * format, ...);
// thread binding
void bind_thread(int worker_id, int *bind_map, int bind_map_size);

//ddf
int iterate_ddt_frontier(ddt_t * ddt);
ddt_t * rt_async_task_to_ddt(task_t * async_task);
void try_schedule_async(task_t * async_task, int comm_task);
}

#ifdef __USE_HC_MM__
#define HC_MM_ALLOC_BATCH_SIZE	100
#include "mm_impl.h"
#endif




#endif /* HCPP_INTERNAL_H_ */
