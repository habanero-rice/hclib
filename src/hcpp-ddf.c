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
 * hcpp-runtime.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include <stdio.h>

#include "hcpp-internal.h"
#include "hcpp-task.h"

// Control debug statements
#define DEBUG_DDF 0

#define EMPTY_DATUM_ERROR_MSG "can not put sentinel value for \"uninitialized\" as a value into DDF"

// For 'headDDTWaitList' when a DDF has been satisfied
#define DDF_SATISFIED NULL

// Default value of a DDF datum
// #define UNINITIALIZED_DDF_DATA_PTR NULL
#define UNINITIALIZED_DDF_DATA_PTR ((void *) -1)

// For waiting frontier (last element of the list)
#define UNINITIALIZED_DDF_WAITLIST_PTR ((ddt_t *) -1)
#define EMPTY_DDF_WAITLIST_PTR NULL

/**
 * Associate a DDT to a DDF list.
 */
void ddt_init(ddt_t * ddt, hclib_ddf_t ** ddf_list) {
	ddt->waitingFrontier = ddf_list;
	ddt->nextDDTWaitingOnSameDDF = NULL;
}

/**
 * Initialize a pre-Allocated DDF.
 */
void hclib_ddf_init(hclib_ddf_t* ddf) {
    ddf->kind = DDF_KIND_SHARED;
    ddf->datum = UNINITIALIZED_DDF_DATA_PTR;
    ddf->headDDTWaitList = UNINITIALIZED_DDF_WAITLIST_PTR;
}

/**
 * Allocate a DDF and initializes it.
 */
hclib_ddf_t * hclib_ddf_create() {
    hclib_ddf_t * ddf = (hclib_ddf_t *) malloc(sizeof(hclib_ddf_t));
    assert(ddf);
    hclib_ddf_init(ddf);
    return ddf;
}

/**
 * Allocate 'nb_ddfs' DDFs in contiguous memory.
 */
hclib_ddf_t **hclib_ddf_create_n(size_t nb_ddfs, int null_terminated) {
	hclib_ddf_t ** ddfs = (hclib_ddf_t **) malloc((sizeof(hclib_ddf_t*) * nb_ddfs));
	int i = 0;
	int lg = (null_terminated) ? nb_ddfs-1 : nb_ddfs;
	while(i < lg) {
		ddfs[i] = hclib_ddf_create();
		i++;
	}
	if (null_terminated) {
		ddfs[lg] = NULL;
	}
	return ddfs;
}

/**
 * Get datum from a DDF.
 * Note: this is concurrent with the 'put' operation.
 */
void * hclib_ddf_get(hclib_ddf_t * ddf) {
	if (ddf->datum == UNINITIALIZED_DDF_DATA_PTR) {
		return NULL;
	}
	return (void *)ddf->datum;
}

/**
 * @brief Destruct a DDF.
 * @param[in] nb_ddfs                           Size of the DDF array
 * @param[in] null_terminated           If true, create nb_ddfs-1 and set the last element to NULL.
 * @param[in] ddf                               The DDF to destruct
 */
void hclib_ddf_free_n(hclib_ddf_t ** ddfs, size_t nb_ddfs, int null_terminated) {
	int i = 0;
	int lg = (null_terminated) ? nb_ddfs-1 : nb_ddfs;
	while(i < lg) {
		hclib_ddf_free(ddfs[i]);
		i++;
	}
	free(ddfs);
}

/**
 * Deallocate a ddf pointer
 */
void hclib_ddf_free(hclib_ddf_t * ddf) {
	free(ddf);
}

__inline__ int __registerIfDDFnotReady_AND(ddt_t* wrapperTask,
        hclib_ddf_t* ddfToCheck ) {
    int success = 0;
    ddt_t* waitListOfDDF = (ddt_t*)ddfToCheck->headDDTWaitList;

    if (waitListOfDDF != EMPTY_DDF_WAITLIST_PTR) {

        while (waitListOfDDF != EMPTY_DDF_WAITLIST_PTR && !success) {
            /* waitListOfDDF can not be EMPTY_DDF_WAITLIST_PTR in here*/
            wrapperTask->nextDDTWaitingOnSameDDF = waitListOfDDF;

            success = __sync_bool_compare_and_swap(&(ddfToCheck -> headDDTWaitList), waitListOfDDF, wrapperTask);
            /* printf("task:%p registered to DDF:%p\n", pollingTask,ddfToCheck); */

            /* may have failed because either some other task tried to be the head or a put occurred. */
            if ( !success ) {
                waitListOfDDF = ( ddt_t* ) ddfToCheck -> headDDTWaitList;
                /* if waitListOfDDF was set to EMPTY_DDF_WAITLIST_PTR, the loop condition will handle that
                 * if another task was added, now try to add in front of that
                 * */
            }
        }

    }

    return success;
}

/**
 * Runtime interface to DDTs.
 * Returns '1' if all ddf dependencies have been satisfied.
 */
int iterate_ddt_frontier(ddt_t* wrapperTask) {
	hclib_ddf_t** currDDFnodeToWaitOn = wrapperTask->waitingFrontier;

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

task_t * rt_ddt_to_async_task(ddt_t * ddt) {
	task_t* t = &(((task_t *)ddt)[-1]);
	return t;
}

ddt_t * rt_async_task_to_ddt(task_t * async_task) {
	return &(((hcpp_task_t*) async_task)->ddt);
}

/**
 * Put datum in the DDF.
 * Close down registration of DDTs on this DDF and iterate over the
 * DDF's frontier to try to advance DDTs that were waiting on this DDF.
 */
void hclib_ddf_put(hclib_ddf_t* ddfToBePut, void * datumToBePut) {
	HASSERT (datumToBePut != UNINITIALIZED_DDF_DATA_PTR && EMPTY_DATUM_ERROR_MSG);
	HASSERT (ddfToBePut != NULL && "can not put into NULL DDF");
	HASSERT (ddfToBePut-> datum == UNINITIALIZED_DDF_DATA_PTR && "violated single assignment property for DDFs");

	volatile ddt_t* waitListOfDDF = NULL;
	ddt_t* currDDT = NULL;
	ddt_t* nextDDT = NULL;

	ddfToBePut-> datum = datumToBePut;
	waitListOfDDF = ddfToBePut->headDDTWaitList;
	/*seems like I can not avoid a CAS here*/
	while ( !__sync_bool_compare_and_swap( &(ddfToBePut -> headDDTWaitList), waitListOfDDF, EMPTY_DDF_WAITLIST_PTR)) {
		waitListOfDDF = ddfToBePut -> headDDTWaitList;
	}

	currDDT = (ddt_t*)waitListOfDDF;

    int iter_count = 0;
	/* printf("DDF:%p was put:%p with value:%d\n", ddfToBePut, datumToBePut,*((int*)datumToBePut)); */
	while (currDDT != UNINITIALIZED_DDF_WAITLIST_PTR) {

		nextDDT = currDDT->nextDDTWaitingOnSameDDF;
		if (iterate_ddt_frontier(currDDT) ) {
			/* printf("pushed:%p\n", currDDT); */
			/*deque_push_default(currFrame);*/
			// DDT eligible to scheduling
			task_t * async_task = rt_ddt_to_async_task(currDDT);
			if (DEBUG_DDF) { printf("ddf: async_task %p\n", async_task); }
			try_schedule_async(async_task, 0);
		}
		currDDT = nextDDT;
        iter_count++;
	}
}
