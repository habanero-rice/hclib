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
 * hcpp.h
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_H_
#define HCPP_H_

#include <iostream>
// Required for lambda as std::function
#include <list>
#include <functional>
#include <stdlib.h>
#include <cstring>
#include <assert.h>

namespace hcpp {

// -------------------- SOME BUILD TIME CONTROL FLAGS ------------------------> START //

//#define __USE_HC_MM__

/*
 * Use this flag to create one dedicated communication worker
 */
#define HCPP_COMM_WORKER

// END <-------------------- SOME BUILD TIME CONTROL FLAGS ------------------------ //

// forward declaration
extern pthread_key_t wskey;
struct hc_context;
struct hc_options;
struct hc_workerState;
struct place_t;
struct hc_deque_t;
struct finish_t;

#ifndef HC_ASSERTION_CHECK
#define HASSERT(cond) if(!(cond)){ printf("W%d: assertion failure\n", hcpp::get_hc_wid()); assert(cond); }
#else
#define HASSERT(cond)       //Do Nothing
#endif
#define current_ws() ((hc_workerState *) pthread_getspecific(wskey))
int get_hc_wid();
}

#ifdef __USE_HC_MM__
#include "mm.h"
#endif

namespace hcpp {
#ifdef __USE_HC_MM__
#define HC_MALLOC(msize) hc_mm_malloc(current_ws(), msize)
#define HC_FREE(p) hc_mm_free(current_ws(), p)
void *hc_mm_malloc(struct hc_workerState * const ws, size_t msize);
void hc_mm_free(struct hc_workerState * const ws, void *ptr);
#else
#define HC_MALLOC(msize)	malloc(msize)
#define HC_FREE(p)			free(p)
#endif
typedef void (*generic_framePtr)(void*);
}

#include "hcpp-ddf.h"
#include "hcpp-async.h"
#include "hcpp-asyncAwait.h"
#include "hcpp-forasync.h"
#include "hcpp-hpt.h"

namespace hcpp {

#define	 DDT_t	ddt_t
void finish(std::function<void()> lambda);
int numWorkers();
void init(int * argc, char ** argv);
void finalize();
void start_finish();
void end_finish();

#ifdef HUPCPP
void gather_commWorker_Stats(int* push_outd, int* push_ind, int* steal_ind);
int totalPendingLocalAsyncs();
int totalAsyncAnyAvailable();
void display_runtime();
volatile int* start_finish_special();
void registerHCUPC_callback(volatile int*);
bool steal_fromComputeWorkers_forDistWS(remoteAsyncAny_task* remAsyncAnybuff);
#endif

}

#endif /* HCPP_H_ */
