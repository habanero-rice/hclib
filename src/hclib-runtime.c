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

#include <pthread.h>
#include <sys/time.h>
#include <stddef.h>

#include <hclib.h>
#include <hclib-internal.h>
#include <hclib-atomics.h>
#include <hclib-finish.h>
#include <hclib-hpt.h>
#include <hcupc-support.hpp>
#include <hclib-cuda.h>

// #define VERBOSE

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

hc_context *hclib_context = NULL;

static char *hclib_stats = NULL;
static int bind_threads = -1;

void hclib_start_finish();

void log_(const char *file, int line, hclib_worker_state *ws,
          const char *format,
          ...) {
    va_list l;
    FILE *f = stderr;
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
int *total_push_ind;
int *total_steals;

static inline void increment_async_counter(int wid) {
    total_push_ind[wid]++;
}

static inline void increment_steals_counter(int wid) {
    total_steals[wid]++;
}

#if defined(HC_COMM_WORKER) && defined(HC_COMM_WORKER_STATS)
static inline void increment_asyncComm_counter() {
    total_push_outd++;
}
#endif

void set_current_worker(int wid) {
    if (pthread_setspecific(ws_key, hclib_context->workers[wid]) != 0) {
        log_die("Cannot set thread-local worker state");
    }

    if (bind_threads) {
        bind_thread(wid, NULL, 0);
    }
}

int get_current_worker() {
    return ((hclib_worker_state *)pthread_getspecific(ws_key))->id;
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

hclib_worker_state *current_ws() {
    return CURRENT_WS_INTERNAL;
}

// FWD declaration for pthread_create
static void *worker_routine(void *args);
#ifdef HC_COMM_WORKER
static void *communication_worker_routine(void *finish);
#endif
#ifdef HC_CUDA
static void *gpu_worker_routine(void *finish);
#endif

/*
 * Main initialization function for the hclib_context object.
 */
void hclib_global_init() {
    // Build queues
    hclib_context->hpt = read_hpt(&hclib_context->places,
                                  &hclib_context->nplaces, &hclib_context->nproc,
                                  &hclib_context->workers, &hclib_context->nworkers);

#ifdef HC_COMM_WORKER
    HASSERT(hclib_context->nworkers > COMMUNICATION_WORKER_ID);
#endif
#ifdef HC_CUDA
    HASSERT(hclib_context->nworkers > GPU_WORKER_ID);
#endif

    for (int i = 0; i < hclib_context->nworkers; i++) {
        hclib_worker_state *ws = hclib_context->workers[i];
        ws->context = hclib_context;
        ws->current_finish = NULL;
        ws->curr_ctx = NULL;
        ws->root_ctx = NULL;
    }
    hclib_context->done_flags = (worker_done_t *)malloc(
                                    hclib_context->nworkers * sizeof(worker_done_t));
#ifdef HC_CUDA
    hclib_context->pinned_host_allocs = NULL;
#endif

    total_push_outd = 0;
    total_steals = (int *)malloc(hclib_context->nworkers * sizeof(int));
    HASSERT(total_steals);
    total_push_ind = (int *)malloc(hclib_context->nworkers * sizeof(int));
    HASSERT(total_push_ind);
    for (int i = 0; i < hclib_context->nworkers; i++) {
        total_steals[i] = 0;
        total_push_ind[i] = 0;
        hclib_context->done_flags[i].flag = 1;
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

    init_hcupc_related_datastructures(hclib_context->nworkers);

    // Sets up the deques and worker contexts for the parsed HPT
    hc_hpt_init(hclib_context);

}

void hclib_display_runtime() {
    printf("---------HCLIB_RUNTIME_INFO-----------\n");
    printf(">>> HCLIB_WORKERS\t= %s\n", getenv("HCLIB_WORKERS"));
    printf(">>> HCLIB_HPT_FILE\t= %s\n", getenv("HCLIB_HPT_FILE"));
    printf(">>> HCLIB_BIND_THREADS\t= %s\n", bind_threads ? "true" : "false");
    if (getenv("HCLIB_WORKERS") && bind_threads) {
        printf("WARNING: HCLIB_BIND_THREADS assign cores in round robin. E.g., "
               "setting HCLIB_WORKERS=12 on 2-socket node, each with 12 cores, "
               "will assign both HCUPC++ places on same socket\n");
    }
    printf(">>> HCLIB_STATS\t\t= %s\n", hclib_stats);
    printf("----------------------------------------\n");
}

void hclib_entrypoint() {
    if (hclib_stats) {
        hclib_display_runtime();
    }

    srand(0);

    hclib_context = (hc_context *)malloc(sizeof(hc_context));
    HASSERT(hclib_context);

    /*
     * Parse the platform description from the HPT configuration file and load
     * it into the hclib_context.
     */
    hclib_global_init();

#ifdef HC_COMM_WORKER
    const int have_comm_worker = 1;
#else
    const int have_comm_worker = 0;
#endif
    // init timer stats
    hclib_initStats(hclib_context->nworkers, have_comm_worker);

    /* Create key to store per thread worker_state */
    if (pthread_key_create(&ws_key, NULL) != 0) {
        log_die("Cannot create ws_key for worker-specific data");
    }

    /*
     * set pthread's concurrency. Doesn't seem to do much on Linux, only
     * relevant when there are more pthreads than hardware cores to schedule
     * them on. */
    pthread_setconcurrency(hclib_context->nworkers);

    // Launch the worker threads
    if (hclib_stats) {
        printf("Using %d worker threads (including main thread)\n",
               hclib_context->nworkers);
    }

    // Start workers
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        fprintf(stderr, "Error in pthread_attr_init\n");
        exit(3);
    }

    for (int i = 1; i < hclib_context->nworkers; i++) {
#ifdef HC_COMM_WORKER
        /*
         * If running with a thread dedicated to network communication (e.g. through
         * UPC, MPI, OpenSHMEM) then skip creating that thread as a compute worker.
         */
        HASSERT(COMMUNICATION_WORKER_ID > 0);
        if (i == COMMUNICATION_WORKER_ID) continue;
#endif
#ifdef HC_CUDA
        HASSERT(GPU_WORKER_ID > 0);
        if (i == GPU_WORKER_ID) continue;
#endif

        if (pthread_create(&hclib_context->workers[i]->t, &attr, worker_routine,
                           &hclib_context->workers[i]->id) != 0) {
            fprintf(stderr, "Error launching thread\n");
            exit(4);
        }
    }
    set_current_worker(0);

    // allocate root finish
    hclib_start_finish();

#ifdef HC_COMM_WORKER
    // Kick off a dedicated communication thread
    if (pthread_create(&hclib_context->workers[COMMUNICATION_WORKER_ID]->t, &attr,
                       communication_worker_routine,
                       CURRENT_WS_INTERNAL->current_finish) != 0) {
        fprintf(stderr, "Error launching communication worker\n");
        exit(5);
    }
#endif
#ifdef HC_CUDA
    // Kick off a dedicated thread to manage all GPUs in a node
    if (pthread_create(&hclib_context->workers[GPU_WORKER_ID]->t, &attr,
                       gpu_worker_routine, CURRENT_WS_INTERNAL->current_finish) != 0) {
        fprintf(stderr, "Error launching GPU worker\n");
        exit(5);
    }
#endif
}

void hclib_signal_join(int nb_workers) {
    int i;
    for (i = 0; i < nb_workers; i++) {
        hclib_context->done_flags[i].flag = 0;
    }
}

void hclib_join(int nb_workers) {
    // Join the workers
#ifdef VERBOSE
    fprintf(stderr, "hclib_join: nb_workers = %d\n", nb_workers);
#endif
    for (int i = 1; i < nb_workers; i++) {
        pthread_join(hclib_context->workers[i]->t, NULL);
    }
#ifdef VERBOSE
    fprintf(stderr, "hclib_join: finished\n");
#endif
}

void hclib_cleanup() {
    hc_hpt_cleanup(hclib_context); /* cleanup deques (allocated by hc mm) */
    pthread_key_delete(ws_key);

    free(hclib_context);
    free(total_steals);
    free(total_push_ind);
    free_hcupc_related_datastructures();
}

static inline void check_in_finish(finish_t *finish) {
    if (finish) {
        // FIXME - does this need to be acquire, or can it be relaxed?
        _hclib_atomic_inc_acquire(&finish->counter);
    }
}

static inline void check_out_finish(finish_t *finish) {
    if (finish) {
        // was this the last async to check out?
        if (_hclib_atomic_dec_release(&finish->counter) == 0) {
#if HCLIB_LITECTX_STRATEGY
            HASSERT(!_hclib_promise_is_satisfied(finish->finish_deps[0]->owner));
            hclib_promise_put(finish->finish_deps[0]->owner, finish);
#endif /* HCLIB_LITECTX_STRATEGY */
        }
    }
}

static inline void execute_task(hclib_task_t *task) {
    finish_t *current_finish = task->current_finish;
    /*
     * Update the current finish of this worker to be inherited from the
     * currently executing task so that any asyncs spawned from the currently
     * executing task are registered on the same finish.
     */
    CURRENT_WS_INTERNAL->current_finish = current_finish;

    // task->_fp is of type 'void (*generic_frame_ptr)(void*)'
#ifdef VERBOSE
    fprintf(stderr, "execute_task: task=%p fp=%p\n", task, task->_fp);
#endif
    (task->_fp)(task->args);
    check_out_finish(current_finish);
    free(task);
}

static inline void rt_schedule_async(hclib_task_t *async_task, int comm_task,
                                     int gpu_task, hclib_worker_state *ws) {
#ifdef VERBOSE
    fprintf(stderr, "rt_schedule_async: async_task=%p comm_task=%d "
            "gpu_task=%d place=%p\n", async_task, comm_task, gpu_task,
            async_task->place);
#endif

    if (comm_task) {
        HASSERT(!gpu_task);
#ifdef HC_COMM_WORKER
        // push on comm_worker out_deq if this is a communication task
        semi_conc_deque_locked_push(comm_worker_out_deque, async_task);
#else
        HASSERT(0);
#endif
    } else if (gpu_task) {
#ifdef HC_CUDA
        semi_conc_deque_locked_push(gpu_worker_deque, async_task);
#else
        HASSERT(0);
#endif
    } else {
        // push on worker deq
        if (async_task->place) {
            deque_push_place(ws, async_task->place, async_task);
        } else {
            const int wid = get_current_worker();
#ifdef VERBOSE
            fprintf(stderr, "rt_schedule_async: scheduling on worker wid=%d "
                    "hclib_context=%p\n", wid, hclib_context);
#endif
            if (!deque_push(&(hclib_context->workers[wid]->current->deque),
                            async_task)) {
                // TODO: deque is full, so execute in place
                printf("WARNING: deque full, local execution\n");
                execute_task(async_task);
            }
#ifdef VERBOSE
            fprintf(stderr, "rt_schedule_async: finished scheduling on worker wid=%d\n",
                    wid);
#endif
        }
    }
}

/*
 * A task which has no dependencies on prior tasks through promises is always
 * immediately ready for scheduling. A task that is registered on some prior
 * promises may be ready for scheduling if all of those promises have already been
 * satisfied. If they have not all been satisfied, the execution of this task is
 * registered on each, and it is only placed in a work deque once all promises have
 * been satisfied.
 */
static inline int is_eligible_to_schedule(hclib_task_t *async_task) {
#ifdef VERBOSE
    fprintf(stderr, "is_eligible_to_schedule: async_task=%p future_list=%p\n",
            async_task, async_task->future_list);
#endif
    if (async_task->future_list != NULL) {
        return register_on_all_promise_dependencies(async_task);
    } else {
        return 1;
    }
}

/*
 * If this async is eligible for scheduling, we insert it into the work-stealing
 * runtime. See is_eligible_to_schedule to understand when a task is or isn't
 * eligible for scheduling.
 */
void try_schedule_async(hclib_task_t *async_task, int comm_task, int gpu_task,
                        hclib_worker_state *ws) {
    if (is_eligible_to_schedule(async_task)) {
        rt_schedule_async(async_task, comm_task, gpu_task, ws);
    }
}

void spawn_handler(hclib_task_t *task, place_t *pl,
        int escaping, int comm, int gpu) {

    HASSERT(task);
#ifndef HC_CUDA
    HASSERT(!gpu);
#endif
#ifndef HC_COMM_WORKER
    HASSERT(!comm);
#endif
    /*
     * check if this is DDDf_t (remote or owner) and do callback to
     * HabaneroUPC++ for implementation.
     *
     * TODO not sure exactly what this does, just copy-pasted. Ask Kumar
     * FIXME - shouldn't this have an #ifdef guard?
     */
    if (task->future_list) {
        check_if_hcupc_distributed_futures(task->future_list);
    }

    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    if (!escaping) {
        check_in_finish(ws->current_finish);
        task->current_finish = ws->current_finish;
        HASSERT(task->current_finish != NULL);
    } else {
        // If escaping task, don't register with current finish
        HASSERT(task->current_finish == NULL);
    }

#ifdef VERBOSE
    fprintf(stderr, "spawn_handler: task=%p\n", task);
#endif

    try_schedule_async(task, comm, gpu, ws);

#ifdef HC_COMM_WORKER_STATS
    const int wid = get_current_worker();
    increment_async_counter(wid);
#endif
}

void spawn_at_hpt(place_t *pl, hclib_task_t *task) {
    // get current worker
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    check_in_finish(ws->current_finish);
    task->current_finish = ws->current_finish;
    task->place = pl;
    try_schedule_async(task, 0, 0, ws);
#ifdef HC_COMM_WORKER_STATS
    const int wid = get_current_worker();
    increment_async_counter(wid);
#endif
}

void spawn(hclib_task_t *task) {
    spawn_handler(task, NULL, 0, 0, 0);
}

void spawn_escaping(hclib_task_t *task, hclib_future_t **future_list) {
    spawn_handler(task, NULL, 1, 0, 0);
}

void spawn_escaping_at(place_t *pl, hclib_task_t *task,
                       hclib_future_t **future_list) {
    spawn_handler(task, pl, 1, 0, 0);
}

void spawn_await_at(hclib_task_t *task, hclib_future_t **future_list,
                    place_t *pl) {
    // FIXME - the future_list member may not have been properly
    // initialized on this call path from C++ code (fix later)
    task->future_list = future_list;
    spawn_handler(task, pl, 0, 0, 0);
}

void spawn_await(hclib_task_t *task, hclib_future_t **future_list) {
    spawn_await_at(task, future_list, NULL);
}

void spawn_comm_task(hclib_task_t *task) {
    spawn_handler(task, NULL, 0, 1, 0);
}

void spawn_gpu_task(hclib_task_t *task) {
    spawn_handler(task, NULL, 0, 0, 1);
}

#ifdef HC_CUDA
extern void *unsupported_place_type_err(place_t *pl);
extern int is_pinned_cpu_mem(void *ptr);

static pending_cuda_op *create_pending_cuda_op(hclib_promise_t *promise,
        void *arg) {
    pending_cuda_op *op = malloc(sizeof(pending_cuda_op));
    op->promise_to_put = promise;
    op->arg_to_put = arg;
    CHECK_CUDA(cudaEventCreate(&op->event));
    return op;
}

static void enqueue_pending_cuda_op(pending_cuda_op *op) {
    if (pending_cuda_ops_head) {
        HASSERT(pending_cuda_ops_tail);
        pending_cuda_ops_tail->next = op;
    } else {
        HASSERT(pending_cuda_ops_tail == NULL);
        pending_cuda_ops_head = op;
    }
    pending_cuda_ops_tail = op;
    op->next = NULL;
}

static pending_cuda_op *do_gpu_memset(place_t *pl, void *ptr, int val,
                                      size_t nbytes, hclib_promise_t *to_put, void *arg_to_put) {
    if (is_cpu_place(pl)) {
        memset(ptr, val, nbytes);
        return NULL;
    } else if (is_nvgpu_place(pl)) {
        CHECK_CUDA(cudaMemsetAsync(ptr, val, nbytes, pl->cuda_stream));
        pending_cuda_op *op = create_pending_cuda_op(to_put,
                              arg_to_put);
        CHECK_CUDA(cudaEventRecord(op->event, pl->cuda_stream));
        return op;
    } else {
        return unsupported_place_type_err(pl);
    }
}

static pending_cuda_op *do_gpu_copy(place_t *dst_pl, place_t *src_pl, void *dst,
                                    void *src, size_t nbytes, hclib_promise_t *to_put, void *arg_to_put) {
    HASSERT(to_put);

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
    set_current_worker(GPU_WORKER_ID);
    worker_done_t *done_flag = hclib_context->done_flags + GPU_WORKER_ID;

    semi_conc_deque_t *deque = gpu_worker_deque;
    while (done_flag->flag) {
        gpu_task_t *task = (gpu_task_t *)semi_conc_deque_non_locked_pop(deque);
#ifdef VERBOSE
        fprintf(stderr, "gpu_worker: done flag=%lu task=%p\n",
                done_flag->flag, task);
#endif
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
                                 task->promise_to_put, task->arg_to_put);
                break;
            }
            case (GPU_MEMSET_TASK): {
                gpu_memset_task_t *memset_task =
                    &task->gpu_task_def.memset_task;
                op = do_gpu_memset(memset_task->pl, memset_task->ptr,
                                   memset_task->val, memset_task->nbytes,
                                   task->promise_to_put, task->arg_to_put);
                break;
            }
            case (GPU_COMPUTE_TASK): {
                gpu_compute_task_t *compute_task =
                    &task->gpu_task_def.compute_task;
                /*
                 * Assume that functor_caller enqueues a kernel in
                 * compute_task->stream
                 */
                CHECK_CUDA(cudaSetDevice(compute_task->cuda_id));
                (compute_task->kernel_launcher->functor_caller)(
                    compute_task->niters, compute_task->tile_size,
                    compute_task->stream,
                    compute_task->kernel_launcher->functor_on_heap);
                op = create_pending_cuda_op(task->promise_to_put,
                                            task->arg_to_put);
                CHECK_CUDA(cudaEventRecord(op->event,
                                           compute_task->stream));
                break;
            }
            default:
                fprintf(stderr, "Unknown GPU task type %d\n",
                        task->gpu_type);
                exit(1);
            }

            check_out_finish(task->t.current_finish);

            if (op) {
#ifdef VERBOSE
                fprintf(stderr, "gpu_worker: task %p produced CUDA pending op "
                        "%p\n", task, op);
#endif
                enqueue_pending_cuda_op(op);
            } else if (task->promise_to_put) {
                /*
                 * No pending operation implies we did a blocking CUDA
                 * operation, and can immediately put any dependent promises.
                 */
                hclib_promise_put(task->promise_to_put, task->arg_to_put);
            }

            // free(task);
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
            hclib_promise_put(pending_cuda_ops_head->promise_to_put,
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
void *communication_worker_routine(void *finish_ptr) {
    set_current_worker(COMMUNICATION_WORKER_ID);
    worker_done_t *done_flag = hclib_context->done_flags + COMMUNICATION_WORKER_ID;

#ifdef VERBOSE
    fprintf(stderr, "communication worker spinning up\n");
#endif

    semi_conc_deque_t *deque = comm_worker_out_deque;
    while (done_flag->flag) {
        // try to pop
        hclib_task_t *task = semi_conc_deque_non_locked_pop(deque);
        // Comm worker cannot steal
        if (task) {
#ifdef HC_COMM_WORKER_STATS
            increment_asyncComm_counter();
#endif
#ifdef VERBOSE
            fprintf(stderr, "communication worker popped task %p\n", task);
#endif
            execute_task(task);
        }
    }

#ifdef VERBOSE
    fprintf(stderr, "communication worker exiting\n");
#endif

    return NULL;
}
#endif

void find_and_run_task(hclib_worker_state *ws) {
    hclib_task_t *task = hpt_pop_task(ws);
    if (!task) {
        while (hclib_context->done_flags[ws->id].flag) {
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
    hclib_signal_join(hclib_context->nworkers);
    // Jump back to the system thread context for this worker
    ctx_swap(ctx, CURRENT_WS_INTERNAL->root_ctx, __func__);
    HASSERT(0); // Should never return here
}

static void core_work_loop(void) {
    uint64_t wid;
    do {
        hclib_worker_state *ws = CURRENT_WS_INTERNAL;
        wid = (uint64_t)ws->id;
        find_and_run_task(ws);
    } while (hclib_context->done_flags[wid].flag);

    // Jump back to the system thread context for this worker
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    HASSERT(ws->root_ctx);
    ctx_swap(get_curr_lite_ctx(), ws->root_ctx, __func__);
    HASSERT(0); // Should never return here
}

static void crt_work_loop(LiteCtx *ctx) {
    core_work_loop(); // this function never returns
    HASSERT(0); // Should never return here
}

/*
 * With the addition of lightweight context switching, worker creation becomes a
 * bit more complicated because we need all task creation and finish scopes to
 * be performed from beneath an explicitly created context, rather than from a
 * pthread context. To do this, we start worker_routine by creating a proxy
 * context to switch from and create a lightweight context to switch to, which
 * enters crt_work_loop immediately, moving into the main work loop, eventually
 * swapping back to the proxy task
 * to clean up this worker thread when the worker thread is signaled to exit.
 */
static void *worker_routine(void *args) {
    const int wid = *((int *)args);
    set_current_worker(wid);
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;

#ifdef HC_COMM_WORKER
    if (wid == COMMUNICATION_WORKER_ID) {
        communication_worker_routine(ws->current_finish);
        return NULL;
    }
#endif
#ifdef HC_CUDA
    if (wid == GPU_WORKER_ID) {
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

static void *worker_routine(void *args) {
    const int wid = *((int *) args);
    set_current_worker(wid);

    hclib_worker_state *ws = CURRENT_WS_INTERNAL;

    while (hclib_context->done_flags[wid].flag) {
        find_and_run_task(ws);
    }

    return NULL;
}
#endif /* HCLIB_LITECTX_STRATEGY */

#if HCLIB_LITECTX_STRATEGY
static void _finish_ctx_resume(void *arg) {
    LiteCtx *currentCtx = get_curr_lite_ctx();
    LiteCtx *finishCtx = arg;
    ctx_swap(currentCtx, finishCtx, __func__);

#ifdef VERBOSE
    fprintf(stderr, "Should not have reached here, currentCtx=%p "
            "finishCtx=%p\n", currentCtx, finishCtx);
#endif
    HASSERT(0);
}

// Based on _help_finish_ctx
void _help_wait(LiteCtx *ctx) {
    hclib_future_t **continuation_deps = ctx->arg;
    LiteCtx *wait_ctx = ctx->prev;

    // reusing _finish_ctx_resume
    hclib_async(_finish_ctx_resume, wait_ctx, continuation_deps,
            NO_PHASER, ANY_PLACE, ESCAPING_ASYNC);

    core_work_loop();
    HASSERT(0);
}

void *hclib_future_wait(hclib_future_t *future) {
    if (_hclib_promise_is_satisfied(future->owner)) {
        return future->owner->datum;
    }

    // save current finish scope (in case of worker swap)
    finish_t *current_finish = CURRENT_WS_INTERNAL->current_finish;

    hclib_future_t *continuation_deps[] = { future, NULL };
    LiteCtx *currentCtx = get_curr_lite_ctx();
    HASSERT(currentCtx);
    LiteCtx *newCtx = LiteCtx_create(_help_wait);
    newCtx->arg = continuation_deps;
    ctx_swap(currentCtx, newCtx, __func__);
    LiteCtx_destroy(currentCtx->prev);

    // restore current finish scope (in case of worker swap)
    CURRENT_WS_INTERNAL->current_finish = current_finish;

    HASSERT(_hclib_promise_is_satisfied(future->owner) &&
            "promise must be satisfied before returning from wait");
    return future->owner->datum;
}

static void _help_finish_ctx(LiteCtx *ctx) {
    /*
     * Set up previous context to be stolen when the finish completes (note that
     * the async must ESCAPE, otherwise this finish scope will deadlock on
     * itself).
     */
#ifdef VERBOSE
    printf("_help_finish_ctx: ctx = %p, ctx->arg = %p\n", ctx, ctx->arg);
#endif
    finish_t *finish = ctx->arg;
    LiteCtx *hclib_finish_ctx = ctx->prev;

    /*
     * Create an async to handle the continuation after the finish, whose state
     * is captured in hclib_finish_ctx and whose execution is pending on
     * finish->finish_deps.
     */
    hclib_async(_finish_ctx_resume, hclib_finish_ctx, finish->finish_deps,
            NO_PHASER, ANY_PLACE, ESCAPING_ASYNC);

    /*
     * The main thread is now exiting the finish (albeit in a separate context),
     * so check it out.
     */
    check_out_finish(finish);

    // keep work-stealing until this context gets swapped out and destroyed
    core_work_loop(); // this function never returns
    HASSERT(0); // we should never return here
}
#else /* default (broken) strategy */

static inline void slave_worker_finishHelper_routine(finish_t *finish) {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
#ifdef HC_COMM_WORKER_STATS
    const int wid = ws->id;
#endif

    while (_hclib_atomic_load_relaxed(&finish->counter) > 0) {
        // try to pop
        hclib_task_t *task = hpt_pop_task(ws);
        if (!task) {
            while (_hclib_atomic_load_relaxed(&finish->counter) > 0) {
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
        if (task) {
            execute_task(task);
        }
    }
}

static void _help_finish(finish_t *finish) {
#ifdef HC_COMM_WORKER
    if (CURRENT_WS_INTERNAL->id == COMMUNICATION_WORKER_ID) {
        communication_worker_routine(finish);
        return;
    }
#endif
#ifdef HC_CUDA
    if (CURRENT_WS_INTERNAL->id == GPU_WORKER_ID) {
        gpu_worker_routine(finish);
        return;
    }
#endif
    slave_worker_finishHelper_routine(finish);
}

#endif /* HCLIB_???_STRATEGY */

void help_finish(finish_t *finish) {
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

        // TODO - should only switch contexts after actually finding work

        // Try to execute a sub-task of the current finish scope
        do {
            hclib_worker_state *ws = CURRENT_WS_INTERNAL;
            hclib_task_t *task = hpt_pop_task(ws);
            // Fall through if we have no local tasks.
            if (!task) {
                break;
            }
            // Since the current finish scope is not yet complete,
            // there's a good chance that the task at the top of the
            // deque is a task from the current finish scope.
            // It's safe to continue executing sub-tasks on the current
            // stack, since the finish scope blocks on them anyway.
            else if (task->current_finish == finish) {
                execute_task(task); // !!! May cause a worker-swap!!!
            }
            // For tasks in a different finish scope, we need a new context.
            // FIXME: Figure out a better way to handle this!
            // For now, just put it back in the deque and fall through.
            else {
                deque_push_place(ws, NULL, task);
                break;
            }
        } while (_hclib_atomic_load_relaxed(&finish->counter) > 1);

        // Someone stole our last task...
        // Create a new context to do other work,
        // and suspend this finish scope pending on the outstanding tasks.
        if (_hclib_atomic_load_relaxed(&finish->counter) > 1) {
            // create finish event
            hclib_promise_t *finish_promise = hclib_promise_create();
            hclib_future_t *finish_deps[] = { &finish_promise->future, NULL };
            finish->finish_deps = finish_deps;

            LiteCtx *currentCtx = get_curr_lite_ctx();
            HASSERT(currentCtx);
            LiteCtx *newCtx = LiteCtx_create(_help_finish_ctx);
            newCtx->arg = finish;

#ifdef VERBOSE
            printf("help_finish: newCtx = %p, newCtx->arg = %p\n", newCtx, newCtx->arg);
#endif
            ctx_swap(currentCtx, newCtx, __func__);

            // note: the other context checks out of the current finish scope

            // destroy the context that resumed this one since it's now defunct
            // (there are no other handles to it, and it will never be resumed)
            LiteCtx_destroy(currentCtx->prev);
            hclib_promise_free(finish_promise);
        } else {
            HASSERT(_hclib_atomic_load_relaxed(&finish->counter) == 1);
            // finish->counter == 1 implies that all the tasks are done
            // (it's only waiting on itself now), so just return!
            _hclib_atomic_dec_acq_rel(&finish->counter);
        }
    }
#else /* default (broken) strategy */
    // FIXME - do I need to decrement the finish counter here?
    _help_finish(finish);
#endif /* HCLIB_???_STRATEGY */

    HASSERT(_hclib_atomic_load_relaxed(&finish->counter) == 0);

}

/*
 * =================== INTERFACE TO USER FUNCTIONS ==========================
 */

void hclib_start_finish() {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    finish_t *finish = malloc(sizeof(*finish));
    HASSERT(finish);
    /*
     * Set finish counter to 1 initially to emulate the main thread inside the
     * finish being a task registered on the finish. When we reach the
     * corresponding end_finish we set up the finish_deps for the continuation
     * and then decrement the counter from the main thread. This ensures that
     * anytime the counter reaches zero, it is safe to do a promise_put on the
     * finish_deps. If we initialized counter to zero here, any async inside the
     * finish could start and finish before the main thread reaches the
     * end_finish, decrementing the finish counter to zero when it completes.
     * This would make it harder to detect when all tasks within the finish have
     * completed, or just the tasks launched so far.
     */
    finish->parent = ws->current_finish;
#if HCLIB_LITECTX_STRATEGY
    finish->finish_deps = NULL;
#endif
    check_in_finish(finish->parent); // check_in_finish performs NULL check
    ws->current_finish = finish;
    _hclib_atomic_store_release(&finish->counter, 1);
}

void hclib_end_finish() {
    finish_t *current_finish = CURRENT_WS_INTERNAL->current_finish;

    HASSERT(_hclib_atomic_load_relaxed(&current_finish->counter) > 0);
    help_finish(current_finish);
    HASSERT(_hclib_atomic_load_relaxed(&current_finish->counter) == 0);

    check_out_finish(current_finish->parent); // NULL check in check_out_finish

    // Don't reuse worker-state! (we might not be on the same worker anymore)
    CURRENT_WS_INTERNAL->current_finish = current_finish->parent;
    free(current_finish);
}

// Based on help_finish
void hclib_end_finish_nonblocking_helper(hclib_promise_t *event) {
    finish_t *current_finish = CURRENT_WS_INTERNAL->current_finish;

    HASSERT(_hclib_atomic_load_relaxed(&current_finish->counter) > 0);

    // NOTE: this is a nasty hack to avoid a memory leak here.
    // Previously we were allocating a two-element array of
    // futures here, but there was no good way to free it...
    // Since the promise datum is null until the promise is satisfied,
    // we can use that as the null-terminator for our future list.
    hclib_future_t **finish_deps = (hclib_future_t**)&event->future.owner;
    HASSERT_STATIC(sizeof(event->future) == sizeof(event->datum) &&
            offsetof(hclib_promise_t, future) == 0 &&
            offsetof(hclib_promise_t, datum) == sizeof(hclib_future_t*),
            "ad-hoc null terminator is correctly aligned in promise struct");
    HASSERT(event->datum == NULL && UNINITIALIZED_PROMISE_DATA_PTR == NULL &&
            "ad-hoc null terminator must have value NULL");
    current_finish->finish_deps = finish_deps;

    // Check out this "task" from the current finish
    check_out_finish(current_finish);

    // Check out the current finish from its parent
    check_out_finish(current_finish->parent);
    CURRENT_WS_INTERNAL->current_finish = current_finish->parent;
}

hclib_future_t *hclib_end_finish_nonblocking() {
    hclib_promise_t *event = hclib_promise_create();
    hclib_end_finish_nonblocking_helper(event);
    return &event->future;
}

int hclib_num_workers() {
    return hclib_context->nworkers;
}

void hclib_gather_comm_worker_stats(int *push_outd, int *push_ind,
                                    int *steal_ind) {
    int asyncPush=0, steals=0, asyncCommPush=total_push_outd;
    for(int i=0; i<hclib_num_workers(); i++) {
        asyncPush += total_push_ind[i];
        steals += total_steals[i];
    }
    *push_outd = asyncCommPush;
    *push_ind = asyncPush;
    *steal_ind = steals;
}

static double mysecond() {
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
    hclib_get_avg_time(&tWork, &tOvh, &tSearch);

    double total_duration = user_specified_timer>0 ? user_specified_timer :
                            duration;
    printf("============================ MMTk Statistics Totals ============================\n");
    printf("time.mu\ttotalPushOutDeq\ttotalPushInDeq\ttotalStealsInDeq\ttWork\ttOverhead\ttSearch\n");
    printf("%.3f\t%d\t%d\t%d\t%.4f\t%.4f\t%.5f\n",total_duration,asyncCommPush,
           asyncPush,steals,tWork,tOvh,tSearch);
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
static void hclib_init() {
    HASSERT(hclib_stats == NULL);
    HASSERT(bind_threads == -1);
    hclib_stats = getenv("HCLIB_STATS");
    bind_threads = (getenv("HCLIB_BIND_THREADS") != NULL);

    if (hclib_stats) {
        show_stats_header();
    }

    const char *hpt_file = getenv("HCLIB_HPT_FILE");
    if (hpt_file == NULL) {
        fprintf(stderr, "WARNING: Running without a provided HCLIB_HPT_FILE, "
                "will make a best effort to generate a default HPT.\n");
    }

    hclib_entrypoint();
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
    hclib_signal_join(hclib_context->nworkers);
#endif /* HCLIB_LITECTX_STRATEGY */

    if (hclib_stats) {
        showStatsFooter();
    }

    hclib_join(hclib_context->nworkers);
    hclib_cleanup();
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

void hclib_launch(generic_frame_ptr fct_ptr, void *arg) {
    hclib_init();
#ifdef HCSHMEM      // TODO (vivekk): replace with HCLIB_COMM_WORKER
    hclib_async(fct_ptr, arg, NO_FUTURE, NO_PHASER, ANY_PLACE, 1);
#else
    hclib_async(fct_ptr, arg, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
#endif
    hclib_finalize();
}

