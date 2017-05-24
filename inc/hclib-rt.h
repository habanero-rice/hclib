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
#define HC_ASSERTION_CHECK_ENABLED 1
#else
#define HC_ASSERTION_CHECK_ENABLED 0
#endif

#define HASSERT(cond) do { \
    if (HC_ASSERTION_CHECK_ENABLED) { \
        if (!(cond)) { \
            fprintf(stderr, "W%d: assertion failure\n", get_current_worker()); \
            assert(0 && (cond)); \
        } \
    } \
} while (0)

#if __cplusplus // C++11 static assert
#define HASSERT_STATIC static_assert
#else // C11 static assert
#define HASSERT_STATIC _Static_assert
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
