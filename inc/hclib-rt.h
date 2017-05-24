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

#ifndef HCLIB_RT_H_
#define HCLIB_RT_H_

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>

#include "hclib_common.h"

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

typedef struct hclib_worker_state hclib_worker_state;

#define HCLIB_MACRO_CONCAT(x, y) _HCLIB_MACRO_CONCAT_IMPL(x, y)
#define _HCLIB_MACRO_CONCAT_IMPL(x, y) x ## y

#ifdef HCLIB_DEBUG
#define HC_DEBUG_ENABLED 1
#else
#define HC_DEBUG_ENABLED 0
#endif

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

#define HCHECK(expr) do { \
    long _hclib_check_rt_result = expr; \
    if (_hclib_check_rt_result != 0) { \
        fprintf(stderr, "W%d: Non-zero return value (%ld) from:\n\t%s\n", \
                get_current_worker(), _hclib_check_rt_result, #expr); \
        abort(); \
    } \
} while (0);

#define HC_UNREACHABLE abort()

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
