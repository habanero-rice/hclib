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
