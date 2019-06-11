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
 * hclib-rt.h
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <assert.h>
#include "litectx.h"

#ifndef HCLIB_RT_H_
#define HCLIB_RT_H_

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
extern pthread_key_t ws_key;
struct hc_context;
struct hclib_options;
struct place_t;
struct hclib_deque_t;
struct finish_t;
struct _hclib_worker_paths;

#ifdef USE_HWLOC
// Various thread affinities supported by HClib
typedef enum {
    /*
     * In HCLIB_AFFINITY_STRIDED mode the HClib runtime will take the set of
     * CPUs it has been given by the calling driver (e.g. taskset, srun, aprun)
     * and set thread affinities such that each thread has the Nth bit,
     * (N+nthreads)th bit, (N+2*nthreads)th bit, etc. set. It is an error to set
     * this affinity with fewer CPUs than there are runtime threads.
     */
    HCLIB_AFFINITY_STRIDED,
    /*
     * In HCLIB_AFFINITY_CHUNKED mode, the HClib runtime will chunk together the
     * set bits in a thread's CPU mask.
     */
    HCLIB_AFFINITY_CHUNKED
} hclib_affinity_t;
#endif

typedef struct _hclib_worker_state {
    // Global context for this instance of the runtime.
    struct hclib_context *context;
    // Pop and steal paths for this worker to traverse when looking for work.
    struct _hclib_worker_paths *paths;
    // The pthread associated with this worker context.
    pthread_t t;
    // Finish scope for the currently executing task.
    struct finish_t* current_finish;
    // Stack frame we are currently executing on.
    LiteCtx *curr_ctx;
    // Root context of the whole runtime instance.
    LiteCtx *root_ctx;
    // The id, identify a worker.
    int id;
    // Total number of workers in this instance of the HClib runtime.
    int nworkers;
    // Place to keep module-specific per-worker state.
    char *module_state;
    /*
     * Variables holding worker IDs for a contiguous chunk of workers that share
     * a NUMA node with this worker, and who we should therefore try to steal
     * from first.
     */
    int base_intra_socket_workers;
    int limit_intra_socket_workers;

    /*
     * Information on currently executing task.
     */
    void *curr_task;
    /* Indicate whether this work is initiating hclib_finalize */ 
    int is_terminator;

    /* This value is used to get ouf of the core_work_loop and find_and_run_task under the root finish */
    int root_counter_val;
} __attribute__ ((aligned (128))) hclib_worker_state;

#define HCLIB_MACRO_CONCAT(x, y) _HCLIB_MACRO_CONCAT_IMPL(x, y)
#define _HCLIB_MACRO_CONCAT_IMPL(x, y) x ## y

#ifdef HC_ASSERTION_CHECK
#define HASSERT(cond) { \
    if (!(cond)) { \
        if (pthread_getspecific(ws_key)) { \
            fprintf(stderr, "W%d: assertion failure\n", hclib_get_current_worker()); \
        } \
        assert(cond); \
    } \
}
#else
#define HASSERT(cond)       // Do Nothing
#endif

#if defined(static_assert) || __cplusplus >= 201103L // defined in C11, C++11
#define HASSERT_STATIC static_assert
#elif __STDC_VERSION__ >= 201112L // C11
#define HASSERT_STATIC _Static_assert
#elif defined(__COUNTER__)
#define HASSERT_STATIC(COND, MSG) \
typedef int HCLIB_MACRO_CONCAT(_hc_static_assert, __COUNTER__)[(COND) ? 1 : -1]
#elif defined(HC_ASSERTION_CHECK)
#warning "Static assertions are not available"
#endif

#define CURRENT_WS_INTERNAL ((hclib_worker_state *) pthread_getspecific(ws_key))

int hclib_get_current_worker();
hclib_worker_state* current_ws();

typedef void (*generic_frame_ptr)(void*);

#include "hclib-timer.h"
#include "hclib-promise.h"

int  hclib_get_num_workers();
void hclib_start_finish();
void hclib_end_finish();
void hclib_user_harness_timer(double dur);

#ifdef __cplusplus
}
#endif

#endif
