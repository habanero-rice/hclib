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
struct hclib_worker_state;
struct place_t;
struct deque_t;
struct hc_deque_t;
struct finish_t;

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

#define HCLIB_MACRO_CONCAT(x, y) _HCLIB_MACRO_CONCAT_IMPL(x, y)
#define _HCLIB_MACRO_CONCAT_IMPL(x, y) x ## y

#ifdef HC_ASSERTION_CHECK
#define HASSERT(cond) { \
    if (!(cond)) { \
        fprintf(stderr, "W%d: assertion failure\n", get_current_worker()); \
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

int get_current_worker();
hclib_worker_state* current_ws();

typedef void (*generic_frame_ptr)(void*);

#include "hclib-timer.h"
#include "hclib-promise.h"
#include "hclib-place.h"

int  hclib_num_workers();
void hclib_start_finish();
void hclib_end_finish();
void hclib_user_harness_timer(double dur);
void hclib_launch(generic_frame_ptr fct_ptr, void * arg);

#ifdef __cplusplus
}
#endif

#endif
