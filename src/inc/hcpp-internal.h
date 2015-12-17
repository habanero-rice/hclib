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

#include <stdarg.h>
#include <stdint.h>
#include "hcpp-deque.h"
#include "hcpp.h"
#include "litectx.h"

#define LOG_LEVEL_FATAL         1
#define LOG_LEVEL_WARN          2
#define LOG_LEVEL_INFO          3
#define LOG_LEVEL_DEBUG         4
#define LOG_LEVEL_TRACE         5

/* set the current log level */
#define LOG_LEVEL LOG_LEVEL_FATAL

#define WHEREARG __FILE__,__LINE__

#define LOG_(level, ...) if (level<=LOG_LEVEL) log_(WHEREARG, CURRENT_WS_INTERNAL, __VA_ARGS__);

/* We more or less mimic log4c without the ERROR level */
#define LOG_FATAL(...)  LOG_(LOG_LEVEL_FATAL, __VA_ARGS__)
#define LOG_WARN(...)   LOG_(LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_INFO(...)   LOG_(LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_DEBUG(...)  LOG_(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_TRACE(...)  LOG_(LOG_LEVEL_TRACE, __VA_ARGS__)

/* log the msg using the fatal logger and abort the program */
#define log_die(... ) { LOG_FATAL(__VA_ARGS__); abort(); }
#define check_log_die(cond, ... ) if(cond) { log_die(__VA_ARGS__) }

#define CACHE_LINE_L1 8

// Default value of a DDF datum
#define UNINITIALIZED_DDF_DATA_PTR ((void *)-1)

typedef struct {
    volatile uint64_t flag;
    void * pad[CACHE_LINE_L1-1];
} worker_done_t;

typedef struct hc_options {
    int nproc; /* number of physical processors */
    /* number of workers, one per hardware core, plus workers for device (GPU)
     * (one per device) */
    int nworkers; 
} hc_options;

/*
 * Global context information for the HC runtime, shared by all worker threads.
 */
typedef struct hc_context {
    struct hc_workerState** workers; /* all workers */
    place_t ** places; /* all the places */
    place_t * hpt; /* root of the HPT? */
    int nworkers; /* # of worker threads created */
    int nplaces; /* # of places */
    int nproc; /* the number of hardware cores in the runtime */
    /* a simple implementation of wait/wakeup condition */
    volatile int workers_wait_cond;
    worker_done_t *done_flags;
} hc_context;

#include "hcpp-finish.h"

typedef struct hc_deque_t {
    /* The actual deque, WARNING: do not move declaration !
     * Other parts of the runtime rely on it being the first one. */
    deque_t deque;
    struct hc_workerState * ws;
    struct hc_deque_t * nnext;
    struct hc_deque_t * prev; /* the deque list of the worker */
    struct place_t * pl;
} hc_deque_t;

void log_(const char * file, int line, hc_workerState * ws, const char * format,
        ...);

// thread binding
void bind_thread(int worker_id, int *bind_map, int bind_map_size);

int get_current_worker();

// ddf
int iterate_ddt_frontier(ddt_t * ddt);
ddt_t * rt_async_task_to_ddt(task_t * async_task);
void try_schedule_async(task_t * async_task, int comm_task);

#endif /* HCPP_INTERNAL_H_ */
