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

#include <pthread.h>
#include <sys/time.h>

#include "hclib.h"
#include "hcpp-internal.h"
#include "hcpp-atomics.h"
#include "hcpp-finish.h"
#include "hcpp-hpt.h"
#include "hcupc-support.h"
#include "hcpp-cuda.h"

// #define VERBOSE

static const int communication_worker_id = 1;
static const int gpu_worker_id = 2;
static double benchmark_start_time_stats = 0;
static double user_specified_timer = 0;
// TODO use __thread on Linux?
pthread_key_t ws_key;

#ifdef HC_COMM_WORKER
semi_conc_deque_t *comm_worker_out_deque;
#endif
#ifdef HC_CUDA
semi_conc_deque_t *gpu_worker_deque;
pending_cuda_op *pending_cuda_ops_head = NULL;
pending_cuda_op *pending_cuda_ops_tail = NULL;
#endif

hc_context* 		hcpp_context;

static char *hcpp_stats = NULL;
static int bind_threads = -1;

void hclib_start_finish();

void log_(const char * file, int line, hc_workerState * ws, const char * format,
        ...) {
    va_list l;
    FILE * f = stderr;
    if (ws != NULL) {
        fprintf(f, "[worker: %d (%s:%d)] ", ws->id, file, line);
    } else {
        fprintf(f, "[%s:%d] ", file, line);
    }
    va_start(l, format);
    vfprintf(f, format, l);
    fflush(f);
    va_end(l);
}

// Statistics
int total_push_outd;
int* total_push_ind;
int* total_steals;

inline void increment_async_counter(int wid) {
    total_push_ind[wid]++;
}

inline void increment_steals_counter(int wid) {
    total_steals[wid]++;
}

inline void increment_asyncComm_counter() {
    total_push_outd++;
}

void set_current_worker(int wid) {
    if (pthread_setspecific(ws_key, hcpp_context->workers[wid]) != 0) {
		log_die("Cannot set thread-local worker state");
    }

    if (bind_threads) {
        bind_thread(wid, NULL, 0);
    }
}

int get_current_worker() {
    return ((hc_workerState*)pthread_getspecific(ws_key))->id;
}

static void set_curr_lite_ctx(LiteCtx *ctx) {
    CURRENT_WS_INTERNAL->curr_ctx = ctx;
}

static LiteCtx *get_curr_lite_ctx() {
    return CURRENT_WS_INTERNAL->curr_ctx;
}

static __inline__ void ctx_swap(LiteCtx *current, LiteCtx *next,
        const char *lbl) {
    // switching to new context
    set_curr_lite_ctx(next);
    LiteCtx_swap(current, next, lbl);
    // switched back to this context
    set_curr_lite_ctx(current);
}

hc_workerState* current_ws() {
    return CURRENT_WS_INTERNAL;
}

// FWD declaration for pthread_create
static void *worker_routine(void * args);
#ifdef HC_COMM_WORKER
static void *communication_worker_routine(void* finish);
#endif
#ifdef HC_CUDA
static void *gpu_worker_routine(void *finish);
#endif

/*
 * Main initialization function for the hcpp_context object.
 */
void hcpp_global_init() {
    // Build queues
    hcpp_context->hpt = read_hpt(&hcpp_context->places,
            &hcpp_context->nplaces, &hcpp_context->nproc,
            &hcpp_context->workers, &hcpp_context->nworkers);
    for (int i = 0; i < hcpp_context->nworkers; i++) {
        hc_workerState * ws = hcpp_context->workers[i];
        ws->context = hcpp_context;
        ws->current_finish = NULL;
        ws->curr_ctx = NULL;
        ws->root_ctx = NULL;
    }
    hcpp_context->done_flags = (worker_done_t *)malloc(
            hcpp_context->nworkers * sizeof(worker_done_t));
#ifdef HC_CUDA
    hcpp_context->pinned_host_allocs = NULL;
#endif

    total_push_outd = 0;
    total_steals = (int *)malloc(hcpp_context->nworkers * sizeof(int));
    HASSERT(total_steals);
    total_push_ind = (int *)malloc(hcpp_context->nworkers * sizeof(int));
    HASSERT(total_push_ind);
    for (int i = 0; i < hcpp_context->nworkers; i++) {
        total_steals[i] = 0;
        total_push_ind[i] = 0;
        hcpp_context->done_flags[i].flag = 1;
    }

#ifdef HC_COMM_WORKER
    comm_worker_out_deque = (semi_conc_deque_t *)malloc(sizeof(semi_conc_deque_t));
    HASSERT(comm_worker_out_deque);
    semi_conc_deque_init(comm_worker_out_deque, NULL);
#endif
#ifdef HC_CUDA
    gpu_worker_deque = (semi_conc_deque_t *)malloc(sizeof(semi_conc_deque_t));
    HASSERT(gpu_worker_deque);
    semi_conc_deque_init(gpu_worker_deque, NULL);
#endif

    init_hcupc_related_datastructures(hcpp_context->nworkers);

    // Sets up the deques and worker contexts for the parsed HPT
    hc_hpt_init(hcpp_context);
}

/*
 * Launch nworkers - 1 worker threads, retaining the current main thread as the
 * final worker. See worker_routine for a description of worker initialization.
 */
void hcpp_create_worker_threads(int nb_workers) {
}

void display_runtime() {
	printf("---------HCPP_RUNTIME_INFO-----------\n");
	printf(">>> HCPP_WORKERS\t= %s\n", getenv("HCPP_WORKERS"));
	printf(">>> HCPP_HPT_FILE\t= %s\n", getenv("HCPP_HPT_FILE"));
	printf(">>> HCPP_BIND_THREADS\t= %s\n", bind_threads ? "true" : "false");
	if (getenv("HCPP_WORKERS") && bind_threads) {
		printf("WARNING: HCPP_BIND_THREADS assign cores in round robin. E.g., "
                "setting HCPP_WORKERS=12 on 2-socket node, each with 12 cores, "
                "will assign both HCUPC++ places on same socket\n");
	}
	printf(">>> HCPP_STATS\t\t= %s\n", hcpp_stats);
	printf("----------------------------------------\n");
}

void hcpp_entrypoint() {
    if (hcpp_stats) {
        display_runtime();
    }

    srand(0);

    hcpp_context = (hc_context *)malloc(sizeof(hc_context));
    HASSERT(hcpp_context);

    /*
     * Parse the platform description from the HPT configuration file and load
     * it into the hcpp_context.
     */
    hcpp_global_init();

#ifdef HC_COMM_WORKER
    const int have_comm_worker = 1;
#else
    const int have_comm_worker = 0;
#endif
    // init timer stats
    hcpp_initStats(hcpp_context->nworkers, have_comm_worker);

    /* Create key to store per thread worker_state */
    if (pthread_key_create(&ws_key, NULL) != 0) {
        log_die("Cannot create ws_key for worker-specific data");
    }

    /*
     * set pthread's concurrency. Doesn't seem to do much on Linux, only
     * relevant when there are more pthreads than hardware cores to schedule
     * them on. */
    pthread_setconcurrency(hcpp_context->nworkers);

    // Launch the worker threads
    if (hcpp_stats) {
        printf("Using %d worker threads (including main thread)\n",
                hcpp_context->nworkers);
    }

    // Start workers
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        fprintf(stderr, "Error in pthread_attr_init\n");
        exit(3);
    }

    for (int i = 1; i < hcpp_context->nworkers; i++) {
#ifdef HC_COMM_WORKER
        /*
         * If running with a thread dedicated to network communication (e.g. through
         * UPC, MPI, OpenSHMEM) then skip creating that thread as a compute worker.
         */
        assert(communication_worker_id > 0);
        if (i == communication_worker_id) continue;
#endif
#ifdef HC_CUDA
        assert(gpu_worker_id > 0);
        if (i == gpu_worker_id) continue;
#endif

        if (pthread_create(&hcpp_context->workers[i]->t, &attr, worker_routine,
                &hcpp_context->workers[i]->id) != 0) {
            fprintf(stderr, "Error launching thread\n");
            exit(4);
        }
    }
    set_current_worker(0);

    // allocate root finish
    hclib_start_finish();

#ifdef HC_COMM_WORKER
    // Kick off a dedicated communication thread
    if (pthread_create(&hcpp_context->workers[communication_worker_id]->t, &attr,
                communication_worker_routine,
                CURRENT_WS_INTERNAL->current_finish) != 0) {
        fprintf(stderr, "Error launching communication worker\n");
        exit(5);
    }
#endif
#ifdef HC_CUDA
    // Kick off a dedicated thread to manage all GPUs in a node
    if (pthread_create(&hcpp_context->workers[gpu_worker_id]->t, &attr,
                gpu_worker_routine, CURRENT_WS_INTERNAL->current_finish) != 0) {
        fprintf(stderr, "Error launching GPU worker\n");
        exit(5);
    }
#endif
}

void hcpp_signal_join(int nb_workers) {
    int i;
    for (i = 0; i < nb_workers; i++) {
        hcpp_context->done_flags[i].flag = 0;
    }
}

void hcpp_join(int nb_workers) {
    // Join the workers
    for(int i=1;i< nb_workers; i++) {
        pthread_join(hcpp_context->workers[i]->t, NULL);
    }
}

void hcpp_cleanup() {
	hc_hpt_cleanup(hcpp_context); /* cleanup deques (allocated by hc mm) */
	pthread_key_delete(ws_key);

	free(hcpp_context);
	free(total_steals);
	free(total_push_ind);
	free_hcupc_related_datastructures();
}

static inline void check_in_finish(finish_t * finish) {
    if (finish) {
        hc_atomic_inc(&(finish->counter));
    }
}

static inline void check_out_finish(finish_t * finish) {
    if (finish) {
        // hc_atomic_dec returns true when finish->counter goes to zero
        if (hc_atomic_dec(&(finish->counter))) {
#if HCLIB_LITECTX_STRATEGY
            // finish_deps will be NULL on the root finish
            hclib_ddf_put(finish->finish_deps[0], finish);
#endif /* HCLIB_LITECTX_STRATEGY */
        }
    }
}

static inline void execute_task(task_t* task) {
    finish_t* current_finish = get_current_finish(task);
    /*
     * Update the current finish of this worker to be inherited from the
     * currently executing task so that any asyncs spawned from the currently
     * executing task are registered on the same finish.
     */
    CURRENT_WS_INTERNAL->current_finish = current_finish;

    // task->_fp is of type 'void (*generic_framePtr)(void*)'
#ifdef VERBOSE
    fprintf(stderr, "execute_task: task=%p fp=%p\n", task, task->_fp);
#endif
    (task->_fp)(task->args);
    check_out_finish(current_finish);
    HC_FREE(task);
}

static inline void rt_schedule_async(task_t *async_task, int comm_task,
        int gpu_task) {
#ifdef VERBOSE
    fprintf(stderr, "rt_schedule_async: async_task=%p comm_task=%d "
            "gpu_task=%d\n", async_task, comm_task, gpu_task);
#endif

    if (comm_task) {
        assert(!gpu_task);
#ifdef HC_COMM_WORKER
        // push on comm_worker out_deq if this is a communication task
        semi_conc_deque_locked_push(comm_worker_out_deque, async_task);
#else
        assert(0);
#endif
    } else if (gpu_task) {
#ifdef HC_CUDA
        semi_conc_deque_locked_push(gpu_worker_deque, async_task);
#else
        assert(0);
#endif
    } else {
        // push on worker deq
        const int wid = get_current_worker();
        if (!deque_push(&(hcpp_context->workers[wid]->current->deque),
                    async_task)) {
            // TODO: deque is full, so execute in place
            printf("WARNING: deque full, local execution\n");
            execute_task(async_task);
        }
    }
}

/*
 * A task which has no dependencies on prior tasks through DDFs is always
 * immediately ready for scheduling. A task that is registered on some prior
 * DDFs may be ready for scheduling if all of those DDFs have already been
 * satisfied. If they have not all been satisfied, the execution of this task is
 * registered on each, and it is only placed in a work deque once all DDFs have
 * been satisfied.
 */
inline int is_eligible_to_schedule(task_t * async_task) {
#ifdef VERBOSE
    fprintf(stderr, "is_eligible_to_schedule: async_task=%p ddf_list=%p\n",
            async_task, async_task->ddf_list);
#endif
    if (async_task->ddf_list != NULL) {
    	ddt_t * ddt = (ddt_t *)rt_async_task_to_ddt(async_task);
        return iterate_ddt_frontier(ddt);
    } else {
        return 1;
    }
}

/*
 * If this async is eligible for scheduling, we insert it into the work-stealing
 * runtime. See is_eligible_to_schedule to understand when a task is or isn't
 * eligible for scheduling.
 */
void try_schedule_async(task_t * async_task, int comm_task, int gpu_task) {
    if (is_eligible_to_schedule(async_task)) {
        rt_schedule_async(async_task, comm_task, gpu_task);
    }
}

void spawn_at_hpt(place_t* pl, task_t * task) {
	// get current worker
	hc_workerState* ws = CURRENT_WS_INTERNAL;
	check_in_finish(ws->current_finish);
	set_current_finish(task, ws->current_finish);
	deque_push_place(ws, pl, task);
#ifdef HC_COMM_WORKER_STATS
	const int wid = get_current_worker();
	increment_async_counter(wid);
#endif
}

void spawn(task_t * task) {
    // get current worker
    hc_workerState* ws = CURRENT_WS_INTERNAL;
    check_in_finish(ws->current_finish);
    set_current_finish(task, ws->current_finish);

#ifdef VERBOSE
    fprintf(stderr, "spawn: task=%p\n", task);
#endif
    try_schedule_async(task, 0, 0);
#ifdef HC_COMM_WORKER_STATS
    const int wid = get_current_worker();
    increment_async_counter(wid);
#endif
}

void spawn_escaping(task_t *task, hclib_ddf_t **ddf_list) {
    // get current worker
    set_current_finish(task, NULL);

#ifdef VERBOSE
    fprintf(stderr, "spawn_escaping: task=%p\n", task);
#endif
    set_ddf_list(task, ddf_list);
    hcpp_task_t *t = (hcpp_task_t*) task;
    ddt_init(&(t->ddt), ddf_list);
    try_schedule_async(task, 0, 0);
#ifdef HC_COMM_WORKER_STATS
    const int wid = get_current_worker();
    increment_async_counter(wid);
#endif
}

void spawn_escaping_at(place_t *pl, task_t *task, hclib_ddf_t **ddf_list) {
    // get current worker
    set_current_finish(task, NULL);
	hc_workerState* ws = CURRENT_WS_INTERNAL;

    set_ddf_list(task, ddf_list);
    hcpp_task_t *t = (hcpp_task_t*) task;
    ddt_init(&(t->ddt), ddf_list);
	deque_push_place(ws, pl, task);
#ifdef HC_COMM_WORKER_STATS
    const int wid = get_current_worker();
    increment_async_counter(wid);
#endif

}

void spawn_await(task_t * task, hclib_ddf_t** ddf_list) {
	/*
     * check if this is DDDf_t (remote or owner) and do callback to
     * HabaneroUPC++ for implementation
     */
	check_if_hcupc_dddf(ddf_list);
	// get current worker
	hc_workerState* ws = CURRENT_WS_INTERNAL;
	check_in_finish(ws->current_finish);
	set_current_finish(task, ws->current_finish);

	set_ddf_list(task, ddf_list);
	hcpp_task_t *t = (hcpp_task_t*) task;
	ddt_init(&(t->ddt), ddf_list);
	try_schedule_async(task, 0, 0);
#ifdef HC_COMM_WORKER_STATS
	const int wid = get_current_worker();
	increment_async_counter(wid);
#endif

}

void spawn_commTask(task_t * task) {
#ifdef HC_COMM_WORKER
    hc_workerState* ws = CURRENT_WS_INTERNAL;
    check_in_finish(ws->current_finish);
    set_current_finish(task, ws->current_finish);
    try_schedule_async(task, 1, 0);
#else
    assert(0);
#endif
}

void spawn_gpu_task(task_t *task) {
#ifdef HC_CUDA
    hc_workerState* ws = CURRENT_WS_INTERNAL;
    check_in_finish(ws->current_finish);
    set_current_finish(task, ws->current_finish);
    try_schedule_async(task, 0, 1);
#else
    assert(0);
#endif
}

#ifdef HC_CUDA
extern void *unsupported_place_type_err(place_t *pl);
extern int is_pinned_cpu_mem(void *ptr);

static pending_cuda_op *create_pending_cuda_op(hclib_ddf_t *ddf, void *arg) {
    pending_cuda_op *op = malloc(sizeof(pending_cuda_op));
    op->ddf_to_put = ddf;
    op->arg_to_put = arg;
    CHECK_CUDA(cudaEventCreate(&op->event));
    return op;
}

static void enqueue_pending_cuda_op(pending_cuda_op *op) {
    if (pending_cuda_ops_head) {
        assert(pending_cuda_ops_tail);
        pending_cuda_ops_tail->next = op;
    } else {
        assert(pending_cuda_ops_tail == NULL);
        pending_cuda_ops_head = op;
    }
    pending_cuda_ops_tail = op;
    op->next = NULL;
}

static pending_cuda_op *do_gpu_copy(place_t *dst_pl, place_t *src_pl, void *dst,
        void *src, size_t nbytes, hclib_ddf_t *to_put, void *arg_to_put) {
    assert(to_put);

    if (is_cpu_place(dst_pl)) {
        if (is_cpu_place(src_pl)) {
            // CPU -> CPU
            memcpy(dst, src, nbytes);
            return NULL;
        } else if (is_nvgpu_place(src_pl)) {
            // GPU -> CPU
#ifdef VERBOSE
            fprintf(stderr, "do_gpu_copy: is dst pinned? %s\n",
                    is_pinned_cpu_mem(dst) ? "true" : "false");
#endif
            if (is_pinned_cpu_mem(dst)) {
                CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes,
                            cudaMemcpyDeviceToHost, src_pl->cuda_stream));
                pending_cuda_op *op = create_pending_cuda_op(to_put,
                        arg_to_put);
                CHECK_CUDA(cudaEventRecord(op->event, src_pl->cuda_stream));
                return op;
            } else {
                CHECK_CUDA(cudaMemcpy(dst, src, nbytes,
                            cudaMemcpyDeviceToHost));
                return NULL;
            }
        } else {
            return unsupported_place_type_err(src_pl);
        }
    } else if (is_nvgpu_place(dst_pl)) {
        if (is_cpu_place(src_pl)) {
            // CPU -> GPU
#ifdef VERBOSE
            fprintf(stderr, "do_gpu_copy: is src pinned? %s\n",
                    is_pinned_cpu_mem(src) ? "true" : "false");
#endif
            if (is_pinned_cpu_mem(src)) {
                CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes,
                            cudaMemcpyHostToDevice, dst_pl->cuda_stream));
                pending_cuda_op *op = create_pending_cuda_op(to_put,
                        arg_to_put);
                CHECK_CUDA(cudaEventRecord(op->event, dst_pl->cuda_stream));
                return op;
            } else {
                CHECK_CUDA(cudaMemcpy(dst, src, nbytes,
                            cudaMemcpyHostToDevice));
                return NULL;
            }
        } else if (is_nvgpu_place(src_pl)) {
            // GPU -> GPU
            CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes,
                        cudaMemcpyDeviceToDevice, dst_pl->cuda_stream));
            pending_cuda_op *op = create_pending_cuda_op(to_put, arg_to_put);
            CHECK_CUDA(cudaEventRecord(op->event, dst_pl->cuda_stream));
            return op;
        } else {
            return unsupported_place_type_err(src_pl);
        }
    } else {
        return unsupported_place_type_err(dst_pl);
    }
}

void *gpu_worker_routine(void *finish_ptr) {
    finish_t *finish = finish_ptr;
    set_current_worker(gpu_worker_id);

    semi_conc_deque_t *deque = gpu_worker_deque;
    while (finish->counter > 0) {
        gpu_task_t *task = (gpu_task_t *)semi_conc_deque_non_locked_pop(deque);
        if (task) {
#ifdef VERBOSE
            fprintf(stderr, "gpu_worker: picked up task %p\n", task);
#endif
            pending_cuda_op *op = NULL;
            switch (task->gpu_type) {
                case (GPU_COMM_TASK): {
                    // Run a GPU communication task
                    gpu_comm_task_t *comm_task = &task->gpu_task_def.comm_task;
                    op = do_gpu_copy(comm_task->dst_pl, comm_task->src_pl,
                            comm_task->dst, comm_task->src, comm_task->nbytes,
                            task->ddf_to_put, task->arg_to_put);
                    break;
                }
                case (GPU_COMPUTE_TASK): {
                    // Run a GPU compute task
                    break;
                }
                default:
                    fprintf(stderr, "Unknown GPU task type %d\n",
                            task->gpu_type);
                    exit(1);
            }

            check_out_finish(get_current_finish((task_t *)task));

            if (op) {
#ifdef VERBOSE
                fprintf(stderr, "gpu_worker: task %p produced CUDA pending op "
                        "%p\n", task, op);
#endif
                enqueue_pending_cuda_op(op);
            } else if (task->ddf_to_put) {
                /*
                 * No pending operation implies we did a blocking CUDA
                 * operation, and can immediately put any dependent DDFs.
                 */
                hclib_ddf_put(task->ddf_to_put, task->arg_to_put);
            }

            HC_FREE(task);
        }

        /*
         * Check to see if any pending ops have completed. This implicitly
         * assumes that events will complete in the order they are placed in the
         * queue. This assumption is not necessary for correctness, but it may
         * cause satisfied events in the pending event queue to not be
         * discovered because they are behind an unsatisfied event.
         */
        cudaError_t event_status = cudaErrorNotReady;
        while (pending_cuda_ops_head &&
                (event_status = cudaEventQuery(pending_cuda_ops_head->event)) ==
                    cudaSuccess) {
            hclib_ddf_put(pending_cuda_ops_head->ddf_to_put,
                    pending_cuda_ops_head->arg_to_put);
            CHECK_CUDA(cudaEventDestroy(pending_cuda_ops_head->event));
            pending_cuda_op *old_head = pending_cuda_ops_head;
            pending_cuda_ops_head = pending_cuda_ops_head->next;
            if (pending_cuda_ops_head == NULL) {
                // Empty list
                pending_cuda_ops_tail = NULL;
            }
            free(old_head);
        }
        if (event_status != cudaErrorNotReady && pending_cuda_ops_head) {
            fprintf(stderr, "Unexpected CUDA error from cudaEventQuery: %s\n",
                    cudaGetErrorString(event_status));
            exit(1);
        }
    }
    return NULL;
}
#endif

#ifdef HC_COMM_WORKER
void *communication_worker_routine(void* finish_ptr) {
    finish_t *finish = finish_ptr;
    set_current_worker(communication_worker_id);

	semi_conc_deque_t *deque = comm_worker_out_deque;
	while (finish->counter > 0) {
		// try to pop
		task_t* task = semi_conc_deque_non_locked_pop(deque);
		// Comm worker cannot steal
		if(task) {
#ifdef HC_COMM_WORKER_STATS
			increment_asyncComm_counter();
#endif
			execute_task(task);
		}
	}
    return NULL;
}
#endif

void find_and_run_task(hc_workerState* ws) {
    task_t* task = hpt_pop_task(ws);
    if (!task) {
        while (hcpp_context->done_flags[ws->id].flag) {
            // try to steal
            task = hpt_steal_task(ws);
            if (task) {
#ifdef HC_COMM_WORKER_STATS
                increment_steals_counter(ws->id);
#endif
                break;
            }
        }
    }

    if (task) {
        execute_task(task);
    }
}

#if HCLIB_LITECTX_STRATEGY
static void _hclib_finalize_ctx(LiteCtx *ctx) {
    hclib_end_finish();
    // Signal shutdown to all worker threads
    hcpp_signal_join(hcpp_context->nworkers);
    // Jump back to the system thread context for this worker
    ctx_swap(ctx, CURRENT_WS_INTERNAL->root_ctx, __func__);
    assert(0); // Should never return here
}

static void core_work_loop(void) {
    uint64_t wid;
    do {
        hc_workerState *ws = CURRENT_WS_INTERNAL;
        wid = (uint64_t)ws->id;
        find_and_run_task(ws);
    } while (hcpp_context->done_flags[wid].flag);

    // Jump back to the system thread context for this worker
    hc_workerState *ws = CURRENT_WS_INTERNAL;
    assert(ws->root_ctx);
    ctx_swap(get_curr_lite_ctx(), ws->root_ctx, __func__);
    assert(0); // Should never return here
}

static void crt_work_loop(LiteCtx *ctx) {
    core_work_loop(); // this function never returns
    assert(0); // Should never return here
}

/*
 * With the addition of lightweight context switching, worker creation becomes a
 * bit more complicated because we need all task creation and finish scopes to
 * be performed from beneath an explicitly created context, rather than from a
 * pthread context. To do this, we start worker_routine by creating a proxy
 * context to switch from and create a lightweight context to switch to, which
 * enters crt_work_loop immediately, moving into the main work loop, eventually
 * swapping back to the proxy task
 * to clean up this worker thread when the worker thread is signalled to exit.
 */
static void* worker_routine(void * args) {
    const int wid = *((int *)args);
    set_current_worker(wid);
    hc_workerState* ws = CURRENT_WS_INTERNAL;

#ifdef HC_COMM_WORKER
    if (wid == communication_worker_id) {
        communication_worker_routine(ws->current_finish);
        return NULL;
    }
#endif
#ifdef HC_CUDA
    if (wid == gpu_worker_id) {
        gpu_worker_routine(ws->current_finish);
        return NULL;
    }
#endif

    // Create proxy original context to switch from
    LiteCtx *currentCtx = LiteCtx_proxy_create(__func__);
    ws->root_ctx = currentCtx;

    /*
     * Create the new proxy we will be switching to, which will start with
     * crt_work_loop at the top of the stack.
     */
    LiteCtx *newCtx = LiteCtx_create(crt_work_loop);
    newCtx->arg = args;

    // Swap in the newCtx lite context
    ctx_swap(currentCtx, newCtx, __func__);

#ifdef VERBOSE
    fprintf(stderr, "worker_routine: worker %d exiting, cleaning up proxy %p "
            "and lite ctx %p\n", get_current_worker(), currentCtx, newCtx);
#endif

    // free resources
    LiteCtx_destroy(currentCtx->prev);
    LiteCtx_proxy_destroy(currentCtx);
    return NULL;
}

#else /* default (broken) strategy */

static void* worker_routine(void * args) {
    const int wid = *((int *) args);
    set_current_worker(wid);

    hc_workerState* ws = CURRENT_WS_INTERNAL;

    while (hcpp_context->done_flags[wid].flag) {
        find_and_run_task(ws);
    }

    return NULL;
}
#endif /* HCLIB_LITECTX_STRATEGY */

void teardown() {

}

#if HCLIB_LITECTX_STRATEGY
static void _finish_ctx_resume(void *arg) {
    LiteCtx *currentCtx = get_curr_lite_ctx();
    LiteCtx *finishCtx = arg;
    ctx_swap(currentCtx, finishCtx, __func__);

    fprintf(stderr, "Should not have reached here, currentCtx=%p "
            "finishCtx=%p\n", currentCtx, finishCtx);
    assert(0);
}

void crt_work_loop(LiteCtx *ctx);

// Based on _help_finish_ctx
void _help_wait(LiteCtx *ctx) {
    hclib_ddf_t **continuation_deps = ctx->arg;
    LiteCtx *wait_ctx = ctx->prev;

    hcpp_task_t *task = (hcpp_task_t *)malloc(sizeof(hcpp_task_t));
    task->async_task._fp = _finish_ctx_resume; // reuse _finish_ctx_resume
    task->async_task.is_asyncAnyType = 0;
    task->async_task.ddf_list = NULL;
    task->async_task.args = wait_ctx;

    spawn_escaping((task_t *)task, continuation_deps);

    core_work_loop();
    assert(0);
}

void *hclib_ddf_wait(hclib_ddf_t *ddf) {
	if (ddf->datum != UNINITIALIZED_DDF_DATA_PTR) {
        return (void *)ddf->datum;
    }
    hclib_ddf_t *continuation_deps[] = { ddf, NULL };
    LiteCtx *currentCtx = get_curr_lite_ctx();
    assert(currentCtx);
    LiteCtx *newCtx = LiteCtx_create(_help_wait);
    newCtx->arg = continuation_deps;
    ctx_swap(currentCtx, newCtx, __func__);
    LiteCtx_destroy(currentCtx->prev);

    assert(ddf->datum != UNINITIALIZED_DDF_DATA_PTR);
    return (void *)ddf->datum;
}

static void _help_finish_ctx(LiteCtx *ctx) {
    /*
     * Set up previous context to be stolen when the finish completes (note that
     * the async must ESCAPE, otherwise this finish scope will deadlock on
     * itself).
     */
    finish_t *finish = ctx->arg;
    LiteCtx *hclib_finish_ctx = ctx->prev;

    hcpp_task_t *task = (hcpp_task_t *)malloc(sizeof(hcpp_task_t));
    task->async_task._fp = _finish_ctx_resume;
    task->async_task.is_asyncAnyType = 0;
    task->async_task.ddf_list = NULL;
    task->async_task.args = hclib_finish_ctx;

    /*
     * Create an async to handle the continuation after the finish, whose state
     * is captured in hclib_finish_ctx and whose execution is pending on
     * finish->finish_deps.
     */
    spawn_escaping((task_t *)task, finish->finish_deps);

    /*
     * The main thread is now exiting the finish (albeit in a separate context),
     * so check it out.
     */
    check_out_finish(finish);
    // keep workstealing until this context gets swapped out and destroyed
    core_work_loop(); // this function never returns
    assert(0); // we should never return here
}
#else /* default (broken) strategy */

static inline void slave_worker_finishHelper_routine(finish_t* finish) {
	hc_workerState* ws = CURRENT_WS_INTERNAL;
	int wid = ws->id;

	while(finish->counter > 0) {
		// try to pop
		task_t* task = hpt_pop_task(ws);
		if (!task) {
			while(finish->counter > 0) {
				// try to steal
				task = hpt_steal_task(ws);
				if (task) {
#ifdef HC_COMM_WORKER_STATS
					increment_steals_counter(wid);
#endif
					break;
				}
			}
		}
		if(task) {
			execute_task(task);
		}
	}
}

static void _help_finish(finish_t * finish) {
#ifdef HC_COMM_WORKER
	if (CURRENT_WS_INTERNAL->id == communication_worker_id) {
		communication_worker_routine(finish);
        return;
	}
#endif
#ifdef HC_CUDA
    if (CURRENT_WS_INTERNAL->id == gpu_worker_id) {
        gpu_worker_routine(finish);
        return;
    }
#endif
	slave_worker_finishHelper_routine(finish);
}
#endif /* HCLIB_LITECTX_STRATEGY */

void help_finish(finish_t * finish) {
    // This is called to make progress when an end_finish has been
    // reached but it hasn't completed yet.
    // Note that's also where the master worker ends up entering its work loop

#if HCLIB_THREAD_BLOCKING_STRATEGY
#error Thread-blocking strategy is not yet implemented
#elif HCLIB_LITECTX_STRATEGY
    {
        /*
         * Creating a new context to switch to is necessary here because the
         * current context needs to become the continuation for this finish
         * (which will be switched back to by _finish_ctx_resume, for which an
         * async is created inside _help_finish_ctx).
         */

        // create finish event
        hclib_ddf_t *finish_deps[] = { hclib_ddf_create(), NULL };
        finish->finish_deps = finish_deps;
        // TODO - should only switch contexts after actually finding work
        LiteCtx *currentCtx = get_curr_lite_ctx();
        assert(currentCtx);
        LiteCtx *newCtx = LiteCtx_create(_help_finish_ctx);
        newCtx->arg = finish;
        ctx_swap(currentCtx, newCtx, __func__);
        // destroy the context that resumed this one since it's now defunct
        // (there are no other handles to it, and it will never be resumed)
        LiteCtx_destroy(currentCtx->prev);
        hclib_ddf_free(finish_deps[0]);
    }
#else /* default (broken) strategy */
    _help_finish(finish);
#endif /* HCLIB_???_STRATEGY */

    assert(finish->counter == 0);
}

/*
 * =================== INTERFACE TO USER FUNCTIONS ==========================
 */

void hclib_start_finish() {
    hc_workerState* ws = CURRENT_WS_INTERNAL;
    finish_t * finish = (finish_t*) HC_MALLOC(sizeof(finish_t));
    /*
     * Set finish counter to 1 initially to emulate the main thread inside the
     * finish being a task registered on the finish. When we reach the
     * corresponding end_finish we set up the finish_deps for the continuation
     * and then decrement the counter from the main thread. This ensures that
     * anytime the counter reaches zero, it is safe to do a ddf_put on the
     * finish_deps. If we initialized counter to zero here, any async inside the
     * finish could start and finish before the main thread reaches the
     * end_finish, decrementing the finish counter to zero when it completes.
     * This would make it harder to detect when all tasks within the finish have
     * completed, or just the tasks launched so far.
     */
    finish->counter = 1;
    finish->parent = ws->current_finish;
    check_in_finish(finish->parent); // check_in_finish performs NULL check
    ws->current_finish = finish;
}

void hclib_end_finish() {
    finish_t* current_finish = CURRENT_WS_INTERNAL->current_finish;

    HASSERT(current_finish->counter > 0);
    help_finish(current_finish);
    HASSERT(current_finish->counter == 0);

    check_out_finish(current_finish->parent); // NULL check in check_out_finish

    CURRENT_WS_INTERNAL->current_finish = current_finish->parent;
    HC_FREE(current_finish);
}

int hclib_num_workers() {
	return hcpp_context->nworkers;
}

void gather_commWorker_Stats(int* push_outd, int* push_ind, int* steal_ind) {
	int asyncPush=0, steals=0, asyncCommPush=total_push_outd;
	for(int i=0; i<hclib_num_workers(); i++) {
		asyncPush += total_push_ind[i];
		steals += total_steals[i];
	}
	*push_outd = asyncCommPush;
	*push_ind = asyncPush;
	*steal_ind = steals;
}

double mysecond() {
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + ((double) tv.tv_usec / 1000000);
}

void runtime_statistics(double duration) {
	int asyncPush=0, steals=0, asyncCommPush=total_push_outd;
	for(int i=0; i<hclib_num_workers(); i++) {
		asyncPush += total_push_ind[i];
		steals += total_steals[i];
	}

	double tWork, tOvh, tSearch;
	hcpp_getAvgTime (&tWork, &tOvh, &tSearch);

	double total_duration = user_specified_timer>0 ? user_specified_timer : duration;
	printf("============================ MMTk Statistics Totals ============================\n");
	printf("time.mu\ttotalPushOutDeq\ttotalPushInDeq\ttotalStealsInDeq\ttWork\ttOverhead\ttSearch\n");
	printf("%.3f\t%d\t%d\t%d\t%.4f\t%.4f\t%.5f\n",total_duration,asyncCommPush,asyncPush,steals,tWork,tOvh,tSearch);
	printf("Total time: %.3f ms\n",total_duration);
	printf("------------------------------ End MMTk Statistics -----------------------------\n");
	printf("===== TEST PASSED in %.3f msec =====\n",duration);
}

static void show_stats_header() {
    printf("\n");
    printf("-----\n");
	printf("mkdir timedrun fake\n");
	printf("\n");
	printf("-----\n");
	benchmark_start_time_stats = mysecond();
}

void hclib_user_harness_timer(double dur) {
	user_specified_timer = dur;
}

void showStatsFooter() {
	double end = mysecond();
	HASSERT(benchmark_start_time_stats != 0);
	double dur = (end-benchmark_start_time_stats)*1000;
	runtime_statistics(dur);
}

/*
 * Main entrypoint for runtime initialization, this function must be called by
 * the user program before any HC actions are performed.
 */
static void hclib_init(int* argc, char** argv) {
    assert(hcpp_stats == NULL);
    assert(bind_threads == -1);
    hcpp_stats = getenv("HCPP_STATS");
    bind_threads = (getenv("HCPP_BIND_THREADS") != NULL);

    if (hcpp_stats) {
        show_stats_header();
    }

    const char *hpt_file = getenv("HCPP_HPT_FILE");
    if (hpt_file == NULL) {
        fprintf(stderr, "ERROR: HCPP_HPT_FILE must be provided. If you do not "
                "want to write one manually, one can be auto-generated for your "
                "platform using the hwloc_to_hpt tool.\n");
        exit(2);
    }

    hcpp_entrypoint();
}


static void hclib_finalize() {
#if HCLIB_LITECTX_STRATEGY
    LiteCtx *finalize_ctx = LiteCtx_proxy_create(__func__);
    LiteCtx *finish_ctx = LiteCtx_create(_hclib_finalize_ctx);
    CURRENT_WS_INTERNAL->root_ctx = finalize_ctx;
    ctx_swap(finalize_ctx, finish_ctx, __func__);
    // free resources
    LiteCtx_destroy(finalize_ctx->prev);
    LiteCtx_proxy_destroy(finalize_ctx);
#else /* default (broken) strategy */
    hclib_end_finish();
    hcpp_signal_join(hcpp_context->nworkers);
#endif /* HCLIB_LITECTX_STRATEGY */

    if (hcpp_stats) {
        showStatsFooter();
    }

    hcpp_join(hcpp_context->nworkers);
    hcpp_cleanup();
}

/**
 * @brief Initialize and launch HClib runtime.
 * Implicitly defines a global finish scope.
 * Returns once the computation has completed and the runtime has been
 * finalized.
 *
 * With fibers, using hclib_launch is a requirement for any HC program. All
 * asyncs/finishes must be performed from beneath hclib_launch. Ensuring that
 * the parent of any end finish is a fiber means that the runtime can assume
 * that the current parent is a fiber, and therefore its lifetime is already
 * managed by the runtime. If we allowed both system-managed threads (i.e. the
 * main thread) and fibers to reach end-finishes, we would have to know to
 * create a LiteCtx from the system-managed stacks and save them, but to not do
 * so when the calling context is already a LiteCtx. While this could be
 * supported, this introduces unnecessary complexity into the runtime code. It
 * is simpler to use hclib_launch to ensure that finish scopes are only ever
 * reached from a fiber context, allowing us to assume that it is safe to simply
 * swap out the current context as a continuation without having to check if we
 * need to do extra work to persist it.
 */
void hclib_launch(int * argc, char ** argv, generic_framePtr fct_ptr,
        void * arg) {
    hclib_init(argc, argv);
    hclib_async(fct_ptr, arg, NO_DDF, NO_PHASER, NO_PROP);
    hclib_finalize();
}
