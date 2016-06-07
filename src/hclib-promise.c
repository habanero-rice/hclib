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
 * hclib-runtime.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include <stdio.h>

#include "hclib-internal.h"
#include "hclib-task.h"

// Control debug statements
#define DEBUG_PROMISE 0

#define EMPTY_DATUM_ERROR_MSG "can not put sentinel value for \"uninitialized\" as a value into promise"

// For 'wait_list_head' when a promise has been satisfied
#define PROMISE_SATISFIED NULL

// For waiting frontier (last element of the list)
#define UNINITIALIZED_PROMISE_WAITLIST_PTR ((hclib_triggered_task_t *) -1)
#define EMPTY_FUTURE_WAITLIST_PTR NULL

/**
 * Associate a triggered task to a promise list.
 */
void hclib_triggered_task_init(hclib_triggered_task_t *task,
        hclib_future_t *waiting_future_0, hclib_future_t *waiting_future_1) {
    task->waiting_on[0] = waiting_future_0;
    task->waiting_on[1] = waiting_future_1;
    task->waiting_on_index = -1;
    task->next_waiting_on_same_future = NULL;
}

/**
 * Initialize a pre-Allocated promise.
 */
void hclib_promise_init(hclib_promise_t *promise) {
    promise->kind = PROMISE_KIND_SHARED;
    promise->datum = UNINITIALIZED_PROMISE_DATA_PTR;
    promise->wait_list_head = UNINITIALIZED_PROMISE_WAITLIST_PTR;
    promise->future.owner = promise;
}

/**
 * Allocate a promise and initializes it.
 */
hclib_promise_t *hclib_promise_create() {
    hclib_promise_t *promise = (hclib_promise_t *) malloc(sizeof(hclib_promise_t));
    HASSERT(promise);
    hclib_promise_init(promise);
    return promise;
}

hclib_future_t *hclib_get_future_for_promise(hclib_promise_t *promise) {
    return &promise->future;
}

/**
 * Allocate 'nb_promises' promises in contiguous memory.
 */
hclib_promise_t **hclib_promise_create_n(size_t nb_promises,
        int null_terminated) {
    hclib_promise_t **promises = (hclib_promise_t **) malloc((sizeof(
                                     hclib_promise_t *) * nb_promises));
    int i = 0;
    int lg = (null_terminated) ? nb_promises - 1 : nb_promises;
    while (i < lg) {
        promises[i] = hclib_promise_create();
        i++;
    }
    if (null_terminated) {
        promises[lg] = NULL;
    }
    return promises;
}

/**
 * Get datum from a promise.
 * Note: this is concurrent with the 'put' operation.
 */
void *hclib_future_get(hclib_future_t *future) {
    if (future->owner->datum == UNINITIALIZED_PROMISE_DATA_PTR) {
        return NULL;
    }
    return (void *)future->owner->datum;
}

/**
 * @brief Destruct a promise.
 * @param[in] nb_promises                           Size of the promise array
 * @param[in] null_terminated           If true, create nb_promises-1 and set the last element to NULL.
 * @param[in] promise                               The promise to destruct
 */
void hclib_promise_free_n(hclib_promise_t **promises, size_t nb_promises,
                          int null_terminated) {
    int i = 0;
    int lg = (null_terminated) ? nb_promises-1 : nb_promises;
    while(i < lg) {
        hclib_promise_free(promises[i]);
        i++;
    }
    free(promises);
}

/**
 * Deallocate a promise pointer
 */
void hclib_promise_free(hclib_promise_t *promise) {
    free(promise);
}

__inline__ int __register_if_promise_not_ready(hclib_triggered_task_t *wrapper_task,
        hclib_future_t *future_to_check) {
    int success = 0;
    hclib_triggered_task_t *wait_list_of_future =
        (hclib_triggered_task_t *)future_to_check->owner->wait_list_head;

    if (wait_list_of_future != EMPTY_FUTURE_WAITLIST_PTR) {

        while (wait_list_of_future != EMPTY_FUTURE_WAITLIST_PTR && !success) {
            // wait_list_of_future can not be EMPTY_FUTURE_WAITLIST_PTR in here
            wrapper_task->next_waiting_on_same_future = wait_list_of_future;

            success = __sync_bool_compare_and_swap(
                          &(future_to_check->owner->wait_list_head), wait_list_of_future,
                          wrapper_task);

            /*
             * may have failed because either some other task tried to be the
             * head or a put occurred.
             */
            if (!success) {
                wait_list_of_future =
                    (hclib_triggered_task_t *)future_to_check->owner->wait_list_head;
                /*
                 * if wait_list_of_future was set to EMPTY_FUTURE_WAITLIST_PTR,
                 * the loop condition will handle that if another task was
                 * added, now try to add in front of that
                 */
            }
        }
    }

    return success;
}

/**
 * Returns '1' if all promise dependencies have been satisfied.
 */
int register_on_all_promise_dependencies(hclib_triggered_task_t *wrapper_task) {

    while (wrapper_task->waiting_on_index < MAX_NUM_WAITS - 1) {
        wrapper_task->waiting_on_index++;
        hclib_future_t *curr = wrapper_task->waiting_on[wrapper_task->waiting_on_index];
        if (curr) {
            if (__register_if_promise_not_ready(wrapper_task, curr)) {
                return 0;
            }
        }
    }

    return 1;
}

//
// Task conversion Implementation
//

hclib_task_t *rt_triggered_task_to_async_task(hclib_triggered_task_t *task) {
    hclib_task_t *t = &(((hclib_task_t *)task)[-1]);
    return t;
}

hclib_triggered_task_t *rt_async_task_to_triggered_task(
    hclib_task_t *async_task) {
    return &(((hclib_dependent_task_t *) async_task)->deps);
}

/**
 * Put datum in the promise.
 * Close down registration of triggered tasks on this promise and iterate over
 * the promise's frontier to try to advance tasks that were waiting on this
 * promise.
 */
void hclib_promise_put(hclib_promise_t *promise_to_be_put, void *datum_to_be_put) {
    HASSERT (datum_to_be_put != UNINITIALIZED_PROMISE_DATA_PTR &&
             EMPTY_DATUM_ERROR_MSG);
    HASSERT (promise_to_be_put != NULL && "can not put into NULL promise");
    HASSERT (promise_to_be_put-> datum == UNINITIALIZED_PROMISE_DATA_PTR &&
             "violated single assignment property for promises");

    volatile hclib_triggered_task_t *wait_list_of_promise =
        promise_to_be_put->wait_list_head;
    hclib_triggered_task_t *curr_task = NULL;
    hclib_triggered_task_t *next_task = NULL;

    promise_to_be_put->datum = datum_to_be_put;
    /*
     * Loop while this CAS fails, trying to atomically grab the list of tasks
     * dependent on the future of this promise. Anyone else who comes along will
     * see that the datum was set and will not add themselves to this list.
     * Seems like we can't avoid a CAS here.
     */
    while (!__sync_bool_compare_and_swap(&(promise_to_be_put->wait_list_head),
                                         wait_list_of_promise,
                                         EMPTY_FUTURE_WAITLIST_PTR)) {
        wait_list_of_promise = promise_to_be_put->wait_list_head;
    }

    curr_task = (hclib_triggered_task_t *)wait_list_of_promise;

    while (curr_task != UNINITIALIZED_PROMISE_WAITLIST_PTR) {

        next_task = curr_task->next_waiting_on_same_future;
        /*
         * For each task that was registered on this promise, we register on the
         * next promise in its list. If there are no remaining unsatisfied
         * promises in its list, the dependent task is made eligible for
         * scheduling.
         */
        if (register_on_all_promise_dependencies(curr_task)) {
            hclib_task_t *async_task = rt_triggered_task_to_async_task(
                                           curr_task);
            if (DEBUG_PROMISE) {
                printf("promise: async_task %p\n", async_task);
            }
            try_schedule_async(async_task, HIGH_PRIORITY, CURRENT_WS_INTERNAL);
        }

        curr_task = next_task;
    }
}


