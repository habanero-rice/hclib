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
 * hcshmem-support.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

#include "hclib-internal.h"
#include "hclib-atomics.h"
#include "hcshmem-support.h"

#ifdef HCSHMEM

int totalPendingLocalAsyncs() {
    /*
     * snapshot of all pending tasks at all workers
     */
#if 1
    return CURRENT_WS_INTERNAL->current_finish->counter;
#else
    int pending_tasks = 0;
    for(int i=0; i<hclib_context->nworkers; i++) {
        hclib_worker_state *ws = hclib_context->workers[i];
        const finish_t *ws_curr_f_i = ws->current_finish;
        if(ws_curr_f_i) {
            bool found = false;
            for(int j=0; j<i; j++) {
                const finish_t *ws_curr_f_j = hclib_context->workers[j]->current_finish;
                if(ws_curr_f_j && ws_curr_f_j == ws_curr_f_i) {
                    found = true;
                    break;
                }
            }
            if(!found) pending_tasks += ws_curr_f_i->counter;
        }
    }

    return pending_tasks;
#endif
}

volatile int *hclib_start_finish_special() {
    hclib_start_finish();
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    return &(ws->current_finish->counter);
}

#endif
