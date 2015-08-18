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

#include "hcpp-internal.h"
#include <stdio.h>

namespace hcpp {

// Control debug statements
#define DEBUG_DDF 0

#define EMPTY_DATUM_ERROR_MSG "can not put sentinel value for \"uninitialized\" as a value into DDF"

// For 'headDDTWaitList' when a DDF has been satisfied
#define DDF_SATISFIED NULL

// Default value of a DDF datum
#define UNINITIALIZED_DDF_DATA_PTR NULL

// For waiting frontier (last element of the list)
#define UNINITIALIZED_DDF_WAITLIST_PTR ((struct ddt_st *) -1)

/**
 * Associate a DDT to a DDF list.
 */
void ddt_init(ddt_t * ddt, ddf_t ** ddf_list) {
	ddt->waitingFrontier = ddf_list;
	ddt->nextDDTWaitingOnSameDDF = NULL;
}

/**
 * Allocate a DDF and initializes it.
 */
ddf_t * ddf_create() {
	ddf_t * ddf = (ddf_t *) malloc(sizeof(ddf_t));
	ddf->datum = NULL;
	ddf->headDDTWaitList = UNINITIALIZED_DDF_WAITLIST_PTR;
	return ddf;
}

/**
 * Allocate 'nb_ddfs' DDFs in contiguous memory.
 */
ddf_t ** ddf_create_n(size_t nb_ddfs, int null_terminated) {
	ddf_t ** ddfs = (ddf_t **) malloc((sizeof(ddf_t*) * nb_ddfs));
	int i = 0;
	int lg = (null_terminated) ? nb_ddfs-1 : nb_ddfs;
	while(i < lg) {
		ddfs[i] = ddf_create();
		i++;
	}
	if (null_terminated) {
		ddfs[lg] = NULL;
	}
	return ddfs;
}

/**
 * Deallocate a ddf pointer
 */
void ddf_free(ddf_t * ddf) {
	free(ddf);
}

static int register_if_ddf_not_ready(ddt_t * ddt, ddf_t * ddf) {
	int registered = 0;
	ddt_t * headDDTWaitList = (ddt_t *) ddf->headDDTWaitList;
	if (DEBUG_DDF) {
		printf("ddf: register_if_ddf_not_ready ddt = %p, ddf = %p \n", ddt, ddf);
	}
	while (headDDTWaitList != DDF_SATISFIED && !registered) {
		/* headDDTWaitList can not be DDF_SATISFIED in here */
		ddt->nextDDTWaitingOnSameDDF = headDDTWaitList;

		// Try to add the ddt to the ddf's list of ddts waiting.
		registered = __sync_bool_compare_and_swap(&(ddf->headDDTWaitList),
				headDDTWaitList, ddt);

		/* may have failed because either some other task tried to be the head or a put occurred. */
		if (!registered) {
			headDDTWaitList = (ddt_t *) ddf->headDDTWaitList;
			/* if waitListOfDDF was set to DDF_SATISFIED, the loop condition will handle that
			 * if another task was added, now try to add in front of that
			 */
		}
	}
	if (DEBUG_DDF && registered) {
		printf("ddf: %p registered %p as headDDT of ddf %p\n", ddt, ddf->headDDTWaitList, ddf);
	}
	return registered;
}

/**
 * Runtime interface to DDTs.
 * Returns '1' if all ddf dependencies have been satisfied.
 */
int iterate_ddt_frontier(ddt_t * ddt) {
	ddf_t ** currentDDF = ddt->waitingFrontier;

	if (DEBUG_DDF) {
		printf("ddf: Trying to iterate ddt %p, frontier ddf is %p \n", ddt, currentDDF);
	}
	// Try and register on ddf until one is not ready.
	while ((*currentDDF) && !register_if_ddf_not_ready(ddt, *currentDDF)) {
		++currentDDF;
	}
	// Here we either:
	// - Have hit a ddf not ready (i.e. no 'put' on this one yet)
	// - iterated over the whole list (i.e. all 'put' must have happened)
	if (DEBUG_DDF) { printf("ddf: iterate result %d\n", (*currentDDF == NULL)); }
	ddt->waitingFrontier = currentDDF;
	return (*currentDDF == NULL);
}

//
// Task conversion Implementation
//

int iterate_ddt_frontier(ddt_t * ddt);

task_t * rt_ddt_to_async_task(ddt_t * ddt) {
    task_t* t = &(((task_t *)ddt)[-1]);
    return t;
}

ddt_t * rt_async_task_to_ddt(task_t * async_task) {
    return &(((hcpp_task_t*) async_task)->ddt);
}

/**
 * Get datum from a DDF.
 * Note: this is concurrent with the 'put' operation.
 */
void * ddf_get(ddf_t * ddf) {
	if (ddf->datum == UNINITIALIZED_DDF_DATA_PTR) {
		return NULL;
	}
	return (void *) ddf->datum;
}

/**
 * Put datum in the DDF.
 * Close down registration of DDTs on this DDF and iterate over the
 * DDF's frontier to try to advance DDTs that were waiting on this DDF.
 */
void ddf_put(ddf_t * ddf, void * datum) {
	HASSERT(datum != UNINITIALIZED_DDF_DATA_PTR && EMPTY_DATUM_ERROR_MSG);
	HASSERT(ddf != NULL && "error: can't DDF is a pointer to NULL");
	//TODO Limitation: not enough to guarantee single assignment
	if (DEBUG_DDF) { printf("ddf: put datum %p\n", ddf->datum); }
	HASSERT(ddf->datum == NULL && "error: violated single assignment property for DDFs");

	volatile ddt_t * headDDTWaitList = NULL;
	ddf->datum = datum;
	headDDTWaitList = ddf->headDDTWaitList;
	if (DEBUG_DDF) { printf("ddf: Retrieve ddf %p head's ddt %p\n", ddf, headDDTWaitList); }
	// Try to cas the wait list to prevent new DDTs from registering on this ddf.
	// When cas successed, new DDTs see the DDF has been satisfied and proceed.
	while (!__sync_bool_compare_and_swap(&(ddf->headDDTWaitList),
			(struct ddt_st *) headDDTWaitList, DDF_SATISFIED )) {
		headDDTWaitList = ddf->headDDTWaitList;
	}
	if (DEBUG_DDF) {
		printf("ddf: Put successful on ddf %p set head's ddf to %p\n", ddf, ddf->headDDTWaitList);
	}
	// Now iterate over the list of DDTs, and try to advance their frontier.
	ddt_t * currDDT = (ddt_t *) headDDTWaitList;
	while (currDDT != UNINITIALIZED_DDF_WAITLIST_PTR) {
		if (DEBUG_DDF) { printf("ddf: Trying to iterate frontier of ddt %p\n", currDDT); }
		if (iterate_ddt_frontier(currDDT)) {
			// DDT eligible to scheduling
			task_t * async_task = rt_ddt_to_async_task(currDDT);
			if (DEBUG_DDF) { printf("ddf: async_task %p\n", async_task); }
			try_schedule_async(async_task, 0);
		}
		currDDT = currDDT->nextDDTWaitingOnSameDDF;
	}
}

}
