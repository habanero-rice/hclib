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
 * hclib-promise.h
 *
 * NOTE: Terminology
 *   promise = an object that has data put on it, and by being put on can
 *             trigger downstream dependent tasks to execute. Promises are
 *             write-only.
 *   future  = a read-only handle to a promise's result that can be used to
 *             block on the satisfaction of that handle, or to express a task
 *             dependency on that promise.
 *   triggered task = a normal task whose execution is predicated on the
 *             satisfaction of some promise. This dependency is expressed using
 *             the future associated with that promise.
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu), Max Grossman (jmg3@rice.edu)
 *      Ported from hclib
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCLIB_PROMISE_H_
#define HCLIB_PROMISE_H_

#include <stdlib.h>

/**
 * @file User Interface to HCLIB's futures and promises.
 */

/*
 * The maximum number of future objects a task can wait on.
 */
#define MAX_NUM_WAITS 4

/**
 * @brief Opaque type for promises.
 */
struct hclib_promise_st;

typedef struct _hclib_future_t {
    struct hclib_promise_st *owner;
} hclib_future_t;

struct hclib_task_t;

// We define a typedef in this unit for convenience
typedef struct hclib_promise_st {
    hclib_future_t future;
    volatile int satisfied;
    void *volatile datum;
    /*
     * List of tasks that are awaiting the satisfaction of this promise.
     * wait_list_head is initialized to SENTINEL_FUTURE_WAITLIST_PTR when
     * the promise has no dependent tasks and swapped out to point to different
     * tasks as a chained list of tasks is created hanging off of this promise.
     * When the promise is satisfied, this is set to
     * SATISFIED_FUTURE_WAITLIST_PTR to indicate that no chaining needs to occur
     * anymore.
     */
    struct hclib_task_t *volatile wait_list_head;
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

void *hclib_future_wait_and_get(hclib_future_t *future);

/*
 * Check if a value has been put on the corresponding promise.
 */
int hclib_future_is_satisfied(hclib_future_t *future);

#endif /* HCLIB_PROMISE_H_ */
