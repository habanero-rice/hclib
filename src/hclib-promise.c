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

#include <stdio.h>

#include "hclib-internal.h"
#include "hclib-task.h"

// Control debug statements
#define DEBUG_PROMISE 0

// Index value indicating that all dependencies are ready
#define FUTURE_FRONTIER_EMPTY (-1)

static inline hclib_task_t **_next_waiting_task(hclib_task_t *t) {
    HASSERT(t && t->future_list);
    return &t->next_waiter;
}

/**
 * Initialize a pre-Allocated promise.
 */
void hclib_promise_init(hclib_promise_t *promise) {
    promise->kind = PROMISE_KIND_SHARED;
    promise->datum = UNINITIALIZED_PROMISE_DATA_PTR;
    promise->wait_list_head = SENTINEL_FUTURE_WAITLIST_PTR;
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
    HASSERT(promises);

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
    HASSERT(_hclib_promise_is_satisfied(future->owner));
    return future->owner->datum;
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

/** Returns '1' if the task was registered and is now waiting */
static inline int _register_if_promise_not_ready(
    hclib_task_t *task,
    hclib_future_t *future_to_check) {
    HASSERT(task != SENTINEL_FUTURE_WAITLIST_PTR);
    int success = 0;
    hclib_promise_t *p = future_to_check->owner;
    hclib_task_t *current_head = p->wait_list_head;

    if (current_head != SATISFIED_FUTURE_WAITLIST_PTR) {

        while (current_head != SATISFIED_FUTURE_WAITLIST_PTR && !success) {
            // current_head can not be SATISFIED_FUTURE_WAITLIST_PTR in here
            *_next_waiting_task(task) = current_head;

            success = __sync_bool_compare_and_swap(
                    &p->wait_list_head, current_head, task);

            /*
             * may have failed because either some other task tried to be the
             * head or a put occurred.
             */
            if (!success) {
                current_head = p->wait_list_head;
                /*
                 * if current_head was set to SATISFIED_FUTURE_WAITLIST_PTR,
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
int register_on_all_promise_dependencies(hclib_task_t *task) {
    hclib_future_t *next_future;
    int i = task->future_frontier;

    if (i == FUTURE_FRONTIER_EMPTY) { return true; }

    while ((next_future = task->future_list[i++])) { // this is an assignment
        if (_register_if_promise_not_ready(task, next_future)) {
            task->future_frontier = i;
            return false;
        }
    }

    HASSERT(next_future == NULL);
    task->future_frontier = FUTURE_FRONTIER_EMPTY;
    return true;
}

/**
 * Put datum in the promise.
 * Close down registration of triggered tasks on this promise and iterate over
 * the promise's frontier to try to advance tasks that were waiting on this
 * promise.
 */
void hclib_promise_put(hclib_promise_t *promiseToBePut, void *datumToBePut) {
    HASSERT (promiseToBePut != NULL && "can not put into NULL promise");
    HASSERT (promiseToBePut->datum == UNINITIALIZED_PROMISE_DATA_PTR &&
             !_hclib_promise_is_satisfied(promiseToBePut) &&
             "violated single assignment property for promises");

    hclib_task_t *current_list_head;

    promiseToBePut->datum = datumToBePut;

    do {
        current_list_head = promiseToBePut->wait_list_head;
    /*seems like I can not avoid a CAS here*/
    // FIXME - should be able ot use __atomic_exchange builtin here
    } while (!__sync_bool_compare_and_swap(&(promiseToBePut->wait_list_head),
                current_list_head, SATISFIED_FUTURE_WAITLIST_PTR));

    hclib_task_t *curr_task = current_list_head;
    hclib_task_t *next_task = NULL;
    int iter_count = 0;
    while (curr_task != SENTINEL_FUTURE_WAITLIST_PTR) {

        next_task = *_next_waiting_task(curr_task);
        if (register_on_all_promise_dependencies(curr_task)) {
            /*deque_push_default(currFrame);*/
            // task eligible to scheduling
            if (DEBUG_PROMISE) {
                printf("promise: async_task %p at %d\n", curr_task, iter_count);
            }
            try_schedule_async(curr_task, CURRENT_WS_INTERNAL);
        }
        curr_task = next_task;
        iter_count++;
    }
}


