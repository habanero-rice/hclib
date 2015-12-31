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
 *   promise = data-driven future
 *   DDT = data-driven task (a task that waits on promise objects)
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Ported from hclib
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCLIB_PROMISE_H_
#define HCLIB_PROMISE_H_

#include <stdlib.h>

/**
 * @file User Interface to HCLIB's Data-Driven Futures
 */

/**
 * @defgroup promise Data-Driven Future
 * @brief Data-Driven Future API for data-flow like programming.
 *
 * @{
 **/

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

/**
 * DDT data-structure to associate DDTs and promises.
 * This is exposed so that the runtime knows the size of the struct.
 */
typedef struct hclib_ddt_st {
    // NULL-terminated list of promises the DDT is registered on
    struct hclib_promise_st ** waitingFrontier;
    /*
     * This allows us to chain all DDTs waiting on a same promise. Whenever a DDT
     * wants to register on a promise, and that promise is not ready, we chain the
     * current DDT and the promise's headDDTWaitList and try to cas on the promise's
     * headDDTWaitList, with the current DDT.
     */
    struct hclib_ddt_st * nextDDTWaitingOnSamePromise;
} hclib_ddt_t;

// We define a typedef in this unit for convenience
typedef struct hclib_promise_st {
	int kind;
    volatile void * datum;
    volatile hclib_ddt_t * headDDTWaitList;
} hclib_promise_t;

/**
 * @brief Allocate and initialize a promise.
 * @return A promise.
 */
hclib_promise_t *hclib_promise_create();

/**
 * Initialize a pre-Allocated promise.
 */
void hclib_promise_init(hclib_promise_t* promise);

/**
 * @brief Allocate and initialize an array of promises.
 * @param[in] nb_promises 				Size of the promise array
 * @param[in] null_terminated 		If true, create nb_promises-1 and set the last element to NULL.
 * @return A contiguous array of promises
 */
hclib_promise_t **hclib_promise_create_n(size_t nb_promises, int null_terminated);

/**
 * @brief Destruct a promise.
 * @param[in] nb_promises 				Size of the promise array
 * @param[in] null_terminated 		If true, create nb_promises-1 and set the last element to NULL.
 * @param[in] promise 				The promise to destruct
 */
void hclib_promise_free_n(hclib_promise_t ** promise,  size_t nb_promises, int null_terminated);

/**
 * @brief Destruct a promise.
 * @param[in] promise 				The promise to destruct
 */
void hclib_promise_free(hclib_promise_t * promise);

/**
 * @brief Get the value of a promise.
 * @param[in] promise 				The promise to get a value from
 */
void * hclib_promise_get(hclib_promise_t * promise);

/**
 * @brief Put a value in a promise.
 * @param[in] promise 				The promise to get a value from
 * @param[in] datum 			The datum to be put in the promise
 */
void hclib_promise_put(hclib_promise_t * promise, void * datum);

/*
 * Block the currently executing task on the provided promise. Returns the datum
 * that was put on promise.
 */
void *hclib_promise_wait(hclib_promise_t *promise);

/*
 * Some extras
 */
void hclib_ddt_init(hclib_ddt_t * ddt, hclib_promise_t ** promise_list);

#endif /* HCLIB_PROMISE_H_ */
