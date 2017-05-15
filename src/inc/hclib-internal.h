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

#ifndef HCLIB_INTERNAL_H_
#define HCLIB_INTERNAL_H_

#include <stdarg.h>
#include <stdint.h>
#include "hclib_config.h"
#include "hclib-worker-config.h"
#include "hclib-tree.h"
#include "hclib-deque.h"
#include "hclib.h"
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

// Default value of a promise datum
#define UNINITIALIZED_PROMISE_DATA_PTR NULL

// For waiting frontier (last element of the list)
#define SATISFIED_FUTURE_WAITLIST_PTR NULL
#define SENTINEL_FUTURE_WAITLIST_PTR ((void*) -1)

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

typedef struct hclib_worker_state {
        pthread_t t; // the pthread associated
        struct finish_t* current_finish;
        struct place_t * pl; // the directly attached place
        // Path from root to worker's leaf place. Array of places.
        struct place_t ** hpt_path;
        struct hc_context * context;
        // the link of other ws in the same place
        struct hclib_worker_state * next_worker;
        struct hc_deque_t * current; // the current deque/place worker is on
        struct hc_deque_t * deques;
        int id; // The id, identify a worker
        int did; // the mapping device id
        LiteCtx *curr_ctx;
        LiteCtx *root_ctx;
} hclib_worker_state;

/*
 * Global context information for the HC runtime, shared by all worker threads.
 */
typedef struct hc_context {
    struct hclib_worker_state** workers; /* all workers */
    place_t ** places; /* all the places */
    place_t * hpt; /* root of the HPT? */
    int nworkers; /* # of worker threads created */
    int nplaces; /* # of places */
    int nproc; /* the number of hardware cores in the runtime */
    /* a simple implementation of wait/wakeup condition */
    volatile int workers_wait_cond;
    worker_done_t *done_flags;
} hc_context;

#include "hclib-finish.h"

typedef struct hc_deque_t {
    /* The actual deque, WARNING: do not move declaration !
     * Other parts of the runtime rely on it being the first one. */
    deque_t deque;
    struct hclib_worker_state * ws;
    struct hc_deque_t * nnext;
    struct hc_deque_t * prev; /* the deque list of the worker */
    struct place_t * pl;
} hc_deque_t;

void log_(const char * file, int line, hclib_worker_state * ws, const char * format,
        ...);

// thread binding
void bind_thread(int worker_id, int *bind_map, int bind_map_size);

int get_current_worker();

// promise
int register_on_all_promise_dependencies(hclib_task_t *task);
void try_schedule_async(hclib_task_t * async_task, hclib_worker_state *ws);

int static inline _hclib_promise_is_satisfied(hclib_promise_t *p) {
    return p->wait_list_head == SATISFIED_FUTURE_WAITLIST_PTR;
}

#endif /* HCLIB_INTERNAL_H_ */
