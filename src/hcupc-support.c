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
#include "hcupc-support.hpp"

#if defined(HUPCPP) && defined(DIST_WS)

#define WORKER_IS_FREE	false
#define WORKER_IS_BUSY	true

asyncAnyInfo  *asyncAnyInfo_forWorker;
commWorkerAsyncAny_infoStruct_t *commWorkerAsyncAny_infoStruct;

/*
 * snapshot of total asyncAny tasks currently available with all computation workers
 */
int totalAsyncAnyAvailable() {
    const int nworkers = numWorkers();
    int total_asyncany = 0;
    for(int i=1; i<nworkers; i++) {
        total_asyncany += (asyncAnyInfo_forWorker[i].asyncAny_pushed -
                           asyncAnyInfo_forWorker[i].asyncAny_stolen);
    }
    return total_asyncany;
}

void spawn_asyncAnyTask(hclib_task_t *task) {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    spawn(task);
    asyncAnyInfo_forWorker[ws->id].asyncAny_pushed++;
}

/*
 * HPT related code
 */

static bool hcupc_callback_registered = false;
volatile int *idle_workers;
static bool *worker_state;

void registerHCUPC_callback(volatile int *idle_wrkrs) {
    idle_workers = idle_wrkrs;
    int workers = CURRENT_WS_INTERNAL->context->nworkers;
    worker_state = new bool[workers];
    for(int i=0; i<workers; i++) worker_state[i] = WORKER_IS_BUSY;
    hcupc_callback_registered = true;
}

void inform_HCUPC_myStatus(int wid, bool status) {
    if(wid == 0) return;	// communication worker

    if(hcupc_callback_registered) {
        if(status == WORKER_IS_BUSY) {
            if(worker_state[wid] == WORKER_IS_FREE) {
                worker_state[wid] = WORKER_IS_BUSY;
                hc_atomic_dec(idle_workers);
            }
        } else {	// WORKER_IS_FREE
            if(worker_state[wid] == WORKER_IS_BUSY) {
                worker_state[wid] = WORKER_IS_FREE;
                hc_atomic_inc(idle_workers);
            }
        }
    }
}

#ifdef HPT_VERSION
bool steal_fromComputeWorkers_forDistWS(remoteAsyncAny_task *remAsyncAnybuff) {
    hclib_task_t buff;
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    place_t *pl = ws->pl;
    while (pl != NULL) {
        hc_deque_t *deqs = pl->deques;
        int nb_deq = pl->ndeques;
        /* Try to steal from right neighbour */

        /* Try to steal once from every other worker first */
        for (int i=1; i<nb_deq; i++) {
            int victim = ((ws->id+i)%nb_deq);
            if((asyncAnyInfo_forWorker[victim].asyncAny_pushed -
                    asyncAnyInfo_forWorker[victim].asyncAny_stolen) <= 0) continue;
            hc_deque_t *d = &deqs[victim];
            hclib_task_t *t = dequeSteal(&(d->deque));
            if (t) { /* steal succeeded */
                HASSERT(t->promise_list == NULL); //TODO promises not supported inside asyncAny
                memcpy(&buff, t, sizeof(hclib_task_t));
                /*
                 * decrement the finish counter associated with this task
                 * as this task is going to complete at remote place
                 * and hence local finish counter will never be decremented
                 * after this.
                 *
                 * TODO: the nesting of finish will not work in this case.
                 * Nesting means the current finish scope of comm worker
                 * will not be same as that of the finish associated with
                 * this task
                 */
                HASSERT(ws->current_finish == buff.current_finish);	//TODO
                hc_atomic_dec(&(buff.current_finish->counter));

                ws->current = get_deque_place(ws, pl);
                if(buff.is_asyncAnyTask()) {
                    hc_atomic_inc(&(asyncAnyInfo_forWorker[victim].asyncAny_stolen));
                }
                /*
                 * copy the lambda into the remoteAsyncAny_task
                 */
                commWorkerAsyncAny_infoStruct->ptr_to_outgoingAsyncAny = remAsyncAnybuff;
                commWorkerAsyncAny_infoStruct->initiatePackingOfAsyncAny = true;
                // Executing this lambda does not means we are executing the task,
                // rather we are forcing the lambda to be copied into the remoteAsyncAny_task struct
                (*buff._fp)(buff._args);
                commWorkerAsyncAny_infoStruct->initiatePackingOfAsyncAny = false;
                commWorkerAsyncAny_infoStruct->ptr_to_outgoingAsyncAny = NULL;
                return true;
            }
        }
#if TODO
        /* We also steal from places that represents the device (GPU),
         * those continuations that are pushed onto the deque by the
         * device workers
         */
        place_t *child = pl->child;
        while (child != NULL) {
            if (is_device_place(child)) {
                frame = hc_deque_steal(ws, child->deques);
                if (frame != NULL) {
                    ws->current = get_deque_place(ws, pl);
                    return true;
                }
            }
            child = child->nnext;
        }
#endif // TODO
        /* Nothing found in this place, go to the parent */
        pl = pl->parent;
    }
    return false;
}
#else	// HPT_VERSION
/*
 * in this version comm_worker simply try to steal by choosing victims in sequential order
 */
bool steal_fromComputeWorkers_forDistWS(remoteAsyncAny_task *remAsyncAnybuff) {
    hclib_task_t buff;
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    const hc_context *context = ws->context;
    const int nworkers = context->nworkers;
    for(int i=1; i<nworkers; i++) {
        hclib_worker_state *ws_i = context->workers[i];
        int victim = ws_i->id;
        if((asyncAnyInfo_forWorker[victim].asyncAny_pushed -
                asyncAnyInfo_forWorker[victim].asyncAny_stolen) <= 0) continue;
        hclib_task_t *t = dequeSteal(&(ws_i->current->deque));
        if (t) { /* steal succeeded */
            HASSERT(t->promise_list == NULL); //TODO promise not supported inside asyncAny
            memcpy(&buff, t, sizeof(hclib_task_t));
            /*
             * decrement the finish counter associated with this task
             * as this task is going to complete at remote place
             * and hence local finish counter will never be decremented
             * after this.
             *
             * TODO: the nesting of finish will not work in this case.
             * Nesting means the current finish scope of comm worker
             * will not be same as that of the finish associated with
             * this task
             */
            HASSERT(ws->current_finish == buff.current_finish);	//TODO
            hc_atomic_dec(&(buff.current_finish->counter));

            if(buff.is_asyncAnyTask()) {
                hc_atomic_inc(&(asyncAnyInfo_forWorker[victim].asyncAny_stolen));
            }
            /*
             * copy the lambda into the remoteAsyncAny_task
             */
            commWorkerAsyncAny_infoStruct->ptr_to_outgoingAsyncAny = remAsyncAnybuff;
            commWorkerAsyncAny_infoStruct->initiatePackingOfAsyncAny = true;
            // Executing this lambda does not means we are executing the task,
            // rather we are forcing the lambda to be copied into the remoteAsyncAny_task struct
            (*buff._fp)(buff._args);
            commWorkerAsyncAny_infoStruct->initiatePackingOfAsyncAny = false;
            commWorkerAsyncAny_infoStruct->ptr_to_outgoingAsyncAny = NULL;
            return true;
        }
    }
    return false;
}
#endif	// HPT_VERSION
#endif // defined(HUPCPP) && defined(DIST_WS)

void init_hcupc_related_datastructures(int workers) {
#if defined(HUPCPP) && defined(DIST_WS)
    asyncAnyInfo_forWorker = (asyncAnyInfo *) malloc(sizeof(
                                 asyncAnyInfo) * workers);
    for(int i=0; i<workers; i++) {
        asyncAnyInfo_forWorker[i].asyncAny_pushed = 0;
        asyncAnyInfo_forWorker[i].asyncAny_stolen = 0;
    }
    commWorkerAsyncAny_infoStruct = new commWorkerAsyncAny_infoStruct_t;
    commWorkerAsyncAny_infoStruct->ptr_to_outgoingAsyncAny = NULL;
    commWorkerAsyncAny_infoStruct->initiatePackingOfAsyncAny = false;
#endif
}

void free_hcupc_related_datastructures() {
#if defined(HUPCPP) && defined(DIST_WS)
    free(asyncAnyInfo_forWorker);
#endif
}

void hcupc_reset_asyncAnyInfo(int id) {
#ifdef DIST_WS
    asyncAnyInfo_forWorker[id].asyncAny_pushed = 0;
    asyncAnyInfo_forWorker[id].asyncAny_stolen = 0;
#endif
}

void hcupc_check_if_asyncAny_stolen(hclib_task_t *buff, int victim, int id) {
#ifdef DIST_WS
    if(buff->is_asyncAnyTask()) {
        hc_atomic_inc(&(asyncAnyInfo_forWorker[victim].asyncAny_stolen));
    }
    // If I am here then local steal has passed
    inform_HCUPC_myStatus(id, WORKER_IS_BUSY);
#endif
}

void hcupc_inform_failedSteal(int id) {
#ifdef DIST_WS
    // If I am here then local steal has failed
    inform_HCUPC_myStatus(id, WORKER_IS_FREE);
#endif
}

void hcupc_check_if_asyncAny_pop(hclib_task_t *buff, int id) {
#ifdef DIST_WS
    if(buff->is_asyncAnyTask()) asyncAnyInfo_forWorker[id].asyncAny_pushed--;
#endif
}

#ifdef HUPCPP

void (*hclib_distributed_future_register_callback)(hclib_future_t **future_list)
    = NULL;

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

void check_if_hcupc_distributed_futures(hclib_future_t **future_list) {
#ifdef HUPCPP
    HASSERT(hclib_distributed_future_register_callback);
    hclib_distributed_future_register_callback(future_list);
#endif
}


