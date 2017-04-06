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

#ifndef HCLIB_PROMISE_H_
#define HCLIB_PROMISE_H_

#include <stdlib.h>

/**
 * @file User Interface to HCLIB's futures and promises.
 */

/**
 * @brief Opaque type for promises.
 */
struct hclib_promise_st;

typedef enum promise_kind {
	PROMISE_KIND_UNKNOWN=0,
	PROMISE_KIND_SHARED,
	PROMISE_KIND_DISTRIBUTED_OWNER,
	PROMISE_KIND_DISTRIBUTED_REMOTE,
} promise_kind_t;

typedef struct _hclib_future_t {
    struct hclib_promise_st *owner;
} hclib_future_t;

struct hclib_task_t;

// We define a typedef in this unit for convenience
typedef struct hclib_promise_st {
    hclib_future_t future;
    void *volatile datum;
    // List of tasks that are awaiting the satisfaction of this promise
    struct hclib_task_t *volatile wait_list_head;
    promise_kind_t kind;
} hclib_promise_t;

/**
 * @brief Allocate and initialize a promise.
 * @return A promise.
 */
hclib_promise_t *hclib_promise_create();

/**
 * Initialize a pre-Allocated promise.
 */
void hclib_promise_init(hclib_promise_t *promise);

/**
 * Fetch the future associated with the provided promise. Tasks can then express
 * their dependencies on the satisfaction of the promise by awaiting on this
 * future.
 */
hclib_future_t *hclib_get_future_for_promise(hclib_promise_t *promise);

/**
 * @brief Allocate and initialize an array of promises.
 * @param[in] nb_promises 				Size of the promise array
 * @param[in] null_terminated 		If true, create nb_promises-1 and set the last element to NULL.
 * @return A contiguous array of promises
 */
hclib_promise_t **hclib_promise_create_n(size_t nb_promises,
        int null_terminated);

/**
 * @brief Destruct a promise.
 * @param[in] nb_promises 			Size of the promise array
 * @param[in] null_terminated 		If true, create nb_promises-1 and set the last element to NULL.
 * @param[in] promise 				The promise to destruct
 */
void hclib_promise_free_n(hclib_promise_t **promise, size_t nb_promises,
        int null_terminated);

/**
 * @brief Destruct a promise.
 * @param[in] promise 				The promise to destruct
 */
void hclib_promise_free(hclib_promise_t *promise);

/**
 * @brief Get the value of a promise.
 * @param[in] promise 				The promise to get a value from
 */
void *hclib_future_get(hclib_future_t *future);

/**
 * @brief Put a value in a promise.
 * @param[in] promise 				The promise to get a value from
 * @param[in] datum 			The datum to be put in the promise
 */
void hclib_promise_put(hclib_promise_t *promise, void *datum);

/*
 * Block the currently executing task on the provided promise. Returns the datum
 * that was put on promise.
 */
void *hclib_future_wait(hclib_future_t *future);

#endif /* HCLIB_PROMISE_H_ */
