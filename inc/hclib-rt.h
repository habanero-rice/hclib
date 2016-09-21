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

#define COMMUNICATION_WORKER_ID 1
#define GPU_WORKER_ID 2

// forward declaration
extern pthread_key_t ws_key;
struct hc_context;
struct hclib_options;
struct place_t;
struct deque_t;
struct hc_deque_t;
struct finish_t;
struct _hclib_worker_paths;

typedef struct _hclib_worker_state {
    struct hclib_context *context;
    struct _hclib_worker_paths *paths;
    pthread_t t; // the pthread associated
    struct finish_t* current_finish;
    LiteCtx *curr_ctx;
    LiteCtx *root_ctx;
    // The id, identify a worker
    int id;
    // Total number of workers in this instance of the HClib runtime
    int nworkers;
} hclib_worker_state;

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

#define CURRENT_WS_INTERNAL ((hclib_worker_state *) pthread_getspecific(ws_key))

int hclib_get_current_worker();
hclib_worker_state* current_ws();

#define HC_MALLOC(msize)	malloc(msize)
#define HC_FREE(p)			free(p)
typedef void (*generic_frame_ptr)(void*);

#include "hclib-timer.h"
#include "hclib-promise.h"

int  hclib_num_workers();
void hclib_start_finish();
void hclib_end_finish();
void hclib_user_harness_timer(double dur);

#ifdef __cplusplus
}
#endif

#endif
