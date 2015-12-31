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
#define DEBUG_DDF 0

#define EMPTY_DATUM_ERROR_MSG "can not put sentinel value for \"uninitialized\" as a value into DDF"

// For 'headDDTWaitList' when a DDF has been satisfied
#define DDF_SATISFIED NULL

// For waiting frontier (last element of the list)
#define UNINITIALIZED_DDF_WAITLIST_PTR ((hclib_ddt_t *) -1)
#define EMPTY_DDF_WAITLIST_PTR NULL

/**
 * Associate a DDT to a DDF list.
 */
void hclib_ddt_init(hclib_ddt_t * ddt, hclib_promise_t ** promise_list) {
	ddt->waitingFrontier = promise_list;
	ddt->nextDDTWaitingOnSameDDF = NULL;
}

/**
 * Initialize a pre-Allocated DDF.
 */
void hclib_promise_init(hclib_promise_t* promise) {
    promise->kind = DDF_KIND_SHARED;
    promise->datum = UNINITIALIZED_DDF_DATA_PTR;
    promise->headDDTWaitList = UNINITIALIZED_DDF_WAITLIST_PTR;
}

/**
 * Allocate a DDF and initializes it.
 */
hclib_promise_t * hclib_promise_create() {
    hclib_promise_t * promise = (hclib_promise_t *) malloc(sizeof(hclib_promise_t));
    HASSERT(promise);
    hclib_promise_init(promise);
    return promise;
}

/**
 * Allocate 'nb_promises' DDFs in contiguous memory.
 */
hclib_promise_t **hclib_promise_create_n(size_t nb_promises, int null_terminated) {
	hclib_promise_t ** promises = (hclib_promise_t **) malloc((sizeof(hclib_promise_t*) * nb_promises));
	int i = 0;
	int lg = (null_terminated) ? nb_promises-1 : nb_promises;
	while(i < lg) {
		promises[i] = hclib_promise_create();
		i++;
	}
	if (null_terminated) {
		promises[lg] = NULL;
	}
	return promises;
}

/**
 * Get datum from a DDF.
 * Note: this is concurrent with the 'put' operation.
 */
void * hclib_promise_get(hclib_promise_t * promise) {
	if (promise->datum == UNINITIALIZED_DDF_DATA_PTR) {
		return NULL;
	}
	return (void *)promise->datum;
}

/**
 * @brief Destruct a DDF.
 * @param[in] nb_promises                           Size of the DDF array
 * @param[in] null_terminated           If true, create nb_promises-1 and set the last element to NULL.
 * @param[in] promise                               The DDF to destruct
 */
void hclib_promise_free_n(hclib_promise_t ** promises, size_t nb_promises, int null_terminated) {
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
void hclib_promise_free(hclib_promise_t * promise) {
	free(promise);
}

__inline__ int __registerIfDDFnotReady_AND(hclib_ddt_t* wrapperTask,
        hclib_promise_t* promiseToCheck) {
    int success = 0;
    hclib_ddt_t* waitListOfDDF = (hclib_ddt_t*)promiseToCheck->headDDTWaitList;

    if (waitListOfDDF != EMPTY_DDF_WAITLIST_PTR) {

        while (waitListOfDDF != EMPTY_DDF_WAITLIST_PTR && !success) {
            // waitListOfDDF can not be EMPTY_DDF_WAITLIST_PTR in here
            wrapperTask->nextDDTWaitingOnSameDDF = waitListOfDDF;

            success = __sync_bool_compare_and_swap(
                    &(promiseToCheck -> headDDTWaitList), waitListOfDDF,
                    wrapperTask);
            // printf("task:%p registered to DDF:%p\n", pollingTask,promiseToCheck);

            /*
             * may have failed because either some other task tried to be the
             * head or a put occurred.
             */
            if (!success) {
                waitListOfDDF = (hclib_ddt_t*)promiseToCheck->headDDTWaitList;
                /*
                 * if waitListOfDDF was set to EMPTY_DDF_WAITLIST_PTR, the loop
                 * condition will handle that if another task was added, now try
                 * to add in front of that
                 */
            }
        }
    }

    return success;
}

/**
 * Runtime interface to DDTs.
 * Returns '1' if all promise dependencies have been satisfied.
 */
int iterate_ddt_frontier(hclib_ddt_t* wrapperTask) {
	hclib_promise_t** currDDFnodeToWaitOn = wrapperTask->waitingFrontier;

    while (*currDDFnodeToWaitOn && !__registerIfDDFnotReady_AND(wrapperTask,
                *currDDFnodeToWaitOn) ) {
        ++currDDFnodeToWaitOn;
    }
	wrapperTask->waitingFrontier = currDDFnodeToWaitOn;
    return *currDDFnodeToWaitOn == NULL;
}

//
// Task conversion Implementation
//

hclib_task_t * rt_ddt_to_async_task(hclib_ddt_t * ddt) {
	hclib_task_t* t = &(((hclib_task_t *)ddt)[-1]);
	return t;
}

hclib_ddt_t * rt_async_task_to_ddt(hclib_task_t * async_task) {
	return &(((hclib_dependent_task_t*) async_task)->ddt);
}

/**
 * Put datum in the DDF.
 * Close down registration of DDTs on this DDF and iterate over the
 * DDF's frontier to try to advance DDTs that were waiting on this DDF.
 */
void hclib_promise_put(hclib_promise_t* promiseToBePut, void *datumToBePut) {
    HASSERT (datumToBePut != UNINITIALIZED_DDF_DATA_PTR &&
            EMPTY_DATUM_ERROR_MSG);
    HASSERT (promiseToBePut != NULL && "can not put into NULL DDF");
    HASSERT (promiseToBePut-> datum == UNINITIALIZED_DDF_DATA_PTR &&
            "violated single assignment property for DDFs");

    volatile hclib_ddt_t* waitListOfDDF = NULL;
    hclib_ddt_t* currDDT = NULL;
    hclib_ddt_t* nextDDT = NULL;

    promiseToBePut-> datum = datumToBePut;
    waitListOfDDF = promiseToBePut->headDDTWaitList;
    /*seems like I can not avoid a CAS here*/
    while (!__sync_bool_compare_and_swap( &(promiseToBePut -> headDDTWaitList),
                waitListOfDDF, EMPTY_DDF_WAITLIST_PTR)) {
        waitListOfDDF = promiseToBePut -> headDDTWaitList;
    }

    currDDT = (hclib_ddt_t*)waitListOfDDF;

    int iter_count = 0;
    while (currDDT != UNINITIALIZED_DDF_WAITLIST_PTR) {

        nextDDT = currDDT->nextDDTWaitingOnSameDDF;
        if (iterate_ddt_frontier(currDDT)) {
            /* printf("pushed:%p\n", currDDT); */
            /*deque_push_default(currFrame);*/
            // DDT eligible to scheduling
            hclib_task_t *async_task = rt_ddt_to_async_task(currDDT);
            if (DEBUG_DDF) { printf("promise: async_task %p\n", async_task); }
            try_schedule_async(async_task, 0, 0, CURRENT_WS_INTERNAL);
        }
        currDDT = nextDDT;
        iter_count++;
    }
}
