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

// We require at least POSIX 1995 for the nanosleep function
#if !(_POSIX_C_SOURCE >= 199506L)
#define _POSIX_C_SOURCE 199506L
#endif

#include <hclib.h>
#include <hclib-internal.h>
#include <hclib-atomics.h>
#include <hclib-finish.h>
#include <hclib-hpt.h>

// These includes must come AFTER hclib-internal.h
// to ensure that the hclib_config.h settings are included.
#include <pthread.h>
#include <sys/time.h>
#include <stddef.h>
#include <time.h>

#define HCLIB_DO_THREAD_COUNTING \
    ((HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_THREADS) \
     || (HCLIB_WORKER_OPTIONS & HCLIB_WORKER_OPTIONS_NO_JOIN))

static double benchmark_start_time_stats = 0;
static double user_specified_timer = 0;
// TODO use __thread on Linux?
pthread_key_t ws_key;

static _Atomic int _hclib_finished_workers;

hc_context *hclib_context = NULL;

static char *hclib_stats = NULL;
static int bind_threads = -1;

static int64_t _hclib_worker_strategy_val = HCLIB_DEFAULT_WORKER_STRATEGY;
static int64_t _hclib_worker_options_val = HCLIB_DEFAULT_WORKER_OPTIONS;

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

#ifdef HC_COMM_WORKER_STATS
static inline void increment_async_counter(int wid) {
    total_push_ind[wid]++;
}

static inline void increment_steals_counter(int wid) {
    total_steals[wid]++;
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
            if (finish->finish_deps) {
                // trigger non-blocking finish or suspended fiber
                HASSERT(!_hclib_promise_is_satisfied(finish->finish_deps[0]->owner));
                hclib_promise_put(finish->finish_deps[0]->owner, finish);
            }
        }
    }
}

static inline void _set_curr_fiber(fcontext_state_t *ctx) {
    CURRENT_WS_INTERNAL->curr_ctx = ctx;
}

static inline fcontext_state_t *_get_curr_fiber() {
    return CURRENT_WS_INTERNAL->curr_ctx;
}

static inline _Noreturn void _fiber_exit(fcontext_state_t *current,
                                         fcontext_t next) {
    fcontext_swap(next, current);
    HASSERT(0); // UNREACHABLE
}

static __inline__ void _fiber_suspend(fcontext_state_t *current,
                                      fcontext_fn_t transfer_fn,
                                      void *arg) {
    // switching to new context
    fcontext_state_t *fresh_fiber = fcontext_create(transfer_fn);
    _set_curr_fiber(fresh_fiber);
    fcontext_transfer_t swap_data = fcontext_swap(fresh_fiber->context, arg);
    // switched back to this context
    _set_curr_fiber(current);
    // destroy the context that resumed this one since it's now defunct
    // (there are no other handles to it, and it will never be resumed)
    // (NOTE: fresh_fiber might differ from prev_fiber)
    fcontext_state_t *prev_fiber = swap_data.data;
    fcontext_destroy(prev_fiber);
}

hclib_worker_state *current_ws() {
    return CURRENT_WS_INTERNAL;
}

// FWD declaration for pthread_create
static void *worker_routine(void *args);

typedef struct {
    volatile int val;
    volatile pthread_t other;
    pthread_mutex_t lock;
    pthread_cond_t cond;
} hclib_worker_wait_state;

static void _finish_ctx_trigger(void *args) {
    // terminating this thread
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;

    // wake up the suspended thread
    hclib_worker_wait_state *wait_state = args;
    HCHECK(pthread_mutex_lock(&wait_state->lock));
    wait_state->other = pthread_self();
    wait_state->val = ws->id;
    HCHECK(pthread_cond_signal(&wait_state->cond));
    HCHECK(pthread_mutex_unlock(&wait_state->lock));

    // exit, and the woken thread will join this thread
    // (returns the wait_state address as a sanity check)
    pthread_exit(wait_state);
}

static void _swap_blocked_thread(finish_t *finish, hclib_future_t *future) {
    HASSERT(future);

    // set up future deps list
    hclib_future_t *async_deps[] = { future, NULL };

    if (finish) {
        finish->finish_deps = async_deps;
    }

    // cache current thread info
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;

    // set up condition variable for suspending this thread
    hclib_worker_wait_state wait_state;
    wait_state.val = -1;
    HCHECK(pthread_mutex_init(&wait_state.lock, NULL));
    HCHECK(pthread_cond_init(&wait_state.cond, NULL));

    const bool no_worker_join = HCLIB_WORKER_OPTIONS & HCLIB_WORKER_OPTIONS_NO_JOIN;

    // suspend (and yield to new thread)
    {
        // Create an async to handle the continuation after the finish, whose state
        // is captured in wait_state and whose execution is pending on async_deps.
        hclib_async(_finish_ctx_trigger, &wait_state, async_deps,
                NO_PHASER, ANY_PLACE, ESCAPING_ASYNC);

        // This thread is done with this finish
        check_out_finish(finish);

        // Start workers

        pthread_attr_t attr;
        if (no_worker_join) {
            HCHECK(pthread_attr_init(&attr));
            HCHECK(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED));
        }

        const pthread_attr_t *attr_ptr = no_worker_join ? &attr : NULL;

        // TODO - make option for creating the new thread only if some
        // percentage of the total workers are currently busy
        pthread_t new_thread;
        HCHECK(pthread_create(&new_thread, attr_ptr, worker_routine, &ws->id));

        HCHECK(pthread_mutex_lock(&wait_state.lock));

        while (wait_state.val < 0) {
            HCHECK(pthread_cond_wait(&wait_state.cond, &wait_state.lock));
        }

        HCHECK(pthread_mutex_unlock(&wait_state.lock));
    }

    // reset current worker context
    // (context may have swapped)
    set_current_worker(wait_state.val);
    ws = CURRENT_WS_INTERNAL;

    if (!no_worker_join) {
        // join the killed thread
        void *output_ptr;
        HCHECK(pthread_join(wait_state.other, &output_ptr));

        // sanity check return value from _finish_ctx_trigger
        if (output_ptr != &wait_state) {
            fprintf(stderr, "FAIL: %p vs %p\n", output_ptr, &wait_state);
            HASSERT(output_ptr == &wait_state);
        }
    }

    // clean up
    HCHECK(pthread_mutex_destroy(&wait_state.lock));
    HCHECK(pthread_cond_destroy(&wait_state.cond));
}

static void _swap_finish_thread(finish_t *finish) {
    hclib_promise_t finish_promise;
    hclib_promise_init(&finish_promise);
    _swap_blocked_thread(finish, &finish_promise.future);
}

/*
 * Main initialization function for the hclib_context object.
 */
void hclib_global_init() {
    // Build queues
    hclib_context->hpt = read_hpt(&hclib_context->places,
                                  &hclib_context->nplaces, &hclib_context->nproc,
                                  &hclib_context->workers, &hclib_context->nworkers);

    for (int i = 0; i < hclib_context->nworkers; i++) {
        hclib_worker_state *ws = hclib_context->workers[i];
        ws->context = hclib_context;
        ws->current_finish = NULL;
        if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIBERS) {
            ws->curr_ctx = NULL;
            ws->root_ctx = NULL;
        }
    }
    hclib_context->done_flags = (worker_done_t *)malloc(
                                    hclib_context->nworkers * sizeof(worker_done_t));
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
               "setting HCLIB_WORKERS=12 on 2-socket node, each with 12 cores.\n");
    }
    printf(">>> HCLIB_STATS\t\t= %s\n", hclib_stats);
    printf("----------------------------------------\n");
}

void hclib_entrypoint() {
    if (hclib_stats) {
        hclib_display_runtime();
    }

    srand(0);  // XXX - why is this here?

    hclib_context = (hc_context *)malloc(sizeof(hc_context));
    HASSERT(hclib_context);

    /*
     * Parse the platform description from the HPT configuration file and load
     * it into the hclib_context.
     */
    hclib_global_init();

    // init timer stats
    hclib_initStats(hclib_context->nworkers);

    /* Create key to store per thread worker_state */
    if (pthread_key_create(&ws_key, NULL) != 0) {
        log_die("Cannot create ws_key for worker-specific data");
    }

    // Launch the worker threads
    if (hclib_stats) {
        printf("Using %d worker threads (including main thread)\n",
               hclib_context->nworkers);
    }

    // Start workers
    pthread_attr_t attr;
    HCHECK(pthread_attr_init(&attr));

    if (HCLIB_WORKER_OPTIONS & HCLIB_WORKER_OPTIONS_NO_JOIN) {
        HCHECK(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED));
    }

    for (int i = 1; i < hclib_context->nworkers; i++) {
        pthread_t other;
        if (pthread_create(&other, &attr, worker_routine,
                           &hclib_context->workers[i]->id) != 0) {
            fprintf(stderr, "Error launching thread\n");
            exit(4);
        }

        hclib_context->workers[i]->t = other;
    }

    hclib_context->workers[0]->t = pthread_self();

    set_current_worker(0);

    // allocate root finish
    hclib_start_finish();
}

void hclib_signal_join(int nb_workers) {
    int i;
    for (i = 0; i < nb_workers; i++) {
        hclib_context->done_flags[i].flag = 0;
    }
}

void hclib_join(int nb_workers) {
    if (HCLIB_DO_THREAD_COUNTING) {
        _hclib_atomic_inc_release(&_hclib_finished_workers);
        struct timespec delay = { .tv_sec = 0, .tv_nsec = 1000000L };
        while (_hclib_atomic_load_acquire(&_hclib_finished_workers) < nb_workers) {
            nanosleep(&delay, NULL);
        }
    }
    if (HCLIB_WORKER_OPTIONS & HCLIB_WORKER_OPTIONS_NO_JOIN) {
        // Don't need to join the other threads
    }
    else {
        // The worker ID for the master worker can change from zero
        // when using a non-fixed thread pool strategy
        const int wid = (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_THREADS)
            ? CURRENT_WS_INTERNAL->id : 0;
        // Join the workers
        LOG_DEBUG("hclib_join: nb_workers = %d\n", nb_workers);
        for (int i = 0; i < nb_workers; i++) {
            if (i != wid) {
                pthread_t other;
                while (!(other = hclib_context->workers[i]->t)) {
                    // spin wait
                }
                pthread_join(other, NULL);
            }
        }
    }
    LOG_DEBUG("hclib_join: finished\n");
}

void hclib_cleanup() {
    hc_hpt_cleanup(hclib_context); /* cleanup deques (allocated by hc mm) */
    pthread_key_delete(ws_key);

    free(hclib_context);
    free(total_steals);
    free(total_push_ind);
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
    LOG_DEBUG("execute_task: task=%p fp=%p\n", task, task->_fp);
    (task->_fp)(task->args);
    check_out_finish(current_finish);
    free(task);
}

static inline void rt_schedule_async(hclib_task_t *async_task,
                                     hclib_worker_state *ws) {
    LOG_DEBUG("rt_schedule_async: async_task=%p place=%p\n",
            async_task, async_task->place);

    // push on worker deq
    if (async_task->place) {
        deque_push_place(ws, async_task->place, async_task);
    } else {
        const int wid = get_current_worker();
        LOG_DEBUG("rt_schedule_async: scheduling on worker wid=%d "
                "hclib_context=%p\n", wid, hclib_context);
        if (!deque_push(&(hclib_context->workers[wid]->current->deque),
                        async_task)) {
            // TODO: deque is full, so execute in place
            printf("WARNING: deque full, local execution\n");
            execute_task(async_task);
        }
        LOG_DEBUG("rt_schedule_async: finished scheduling on worker wid=%d\n",
                wid);
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
    LOG_DEBUG("is_eligible_to_schedule: async_task=%p future_list=%p\n",
            async_task, async_task->future_list);
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
void try_schedule_async(hclib_task_t *async_task, hclib_worker_state *ws) {
    if (is_eligible_to_schedule(async_task)) {
        rt_schedule_async(async_task, ws);
    }
}

void spawn_handler(hclib_task_t *task, place_t *pl, bool escaping) {

    HASSERT(task);

    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    if (!escaping) {
        check_in_finish(ws->current_finish);
        task->current_finish = ws->current_finish;
        HASSERT(task->current_finish != NULL);
    } else {
        // If escaping task, don't register with current finish
        HASSERT(task->current_finish == NULL);
    }

    LOG_DEBUG("spawn_handler: task=%p\n", task);

    try_schedule_async(task, ws);
}

void spawn_at_hpt(place_t *pl, hclib_task_t *task) {
    // get current worker
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    check_in_finish(ws->current_finish);
    task->current_finish = ws->current_finish;
    task->place = pl;
    try_schedule_async(task, ws);
#ifdef HC_COMM_WORKER_STATS
    const int wid = get_current_worker();
    increment_async_counter(wid);
#endif
}

void spawn(hclib_task_t *task) {
    spawn_handler(task, NULL, false);
}

void spawn_escaping(hclib_task_t *task, hclib_future_t **future_list) {
    spawn_handler(task, NULL, true);
}

void spawn_escaping_at(place_t *pl, hclib_task_t *task,
                       hclib_future_t **future_list) {
    spawn_handler(task, pl, true);
}

void spawn_await_at(hclib_task_t *task, hclib_future_t **future_list,
                    place_t *pl) {
    // FIXME - the future_list member may not have been properly
    // initialized on this call path from C++ code (fix later)
    task->future_list = future_list;
    spawn_handler(task, pl, false);
}

void spawn_await(hclib_task_t *task, hclib_future_t **future_list) {
    spawn_await_at(task, future_list, NULL);
}

static inline bool _try_run_local(hclib_worker_state *ws) {
    // try to pop
    hclib_task_t *task = hpt_pop_task(ws);
    if (task) {
        execute_task(task);
    }
    return task;
}

static inline bool _try_run_global(hclib_worker_state *ws) {
    // try to steal
    hclib_task_t *task = hpt_steal_task(ws);
    if (task) {
#ifdef HC_COMM_WORKER_STATS
        increment_steals_counter(ws->id);
#endif
        execute_task(task);
    }
    return task;
}

void find_and_run_task(hclib_worker_state *ws) {
    if (!_try_run_local(ws)) {
        while (hclib_context->done_flags[ws->id].flag) {
            // try to steal
            _try_run_global(ws);
        }
    }
}

static void _hclib_finalize_ctx(fcontext_transfer_t fiber_data) {
    CURRENT_WS_INTERNAL->root_ctx = fiber_data.prev_context;
    hclib_end_finish();
    // Signal shutdown to all worker threads
    hclib_signal_join(hclib_context->nworkers);
    // Jump back to the system thread context for this worker
    _fiber_exit(_get_curr_fiber(), CURRENT_WS_INTERNAL->root_ctx);
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
    _fiber_exit(_get_curr_fiber(), CURRENT_WS_INTERNAL->root_ctx);
    HASSERT(0); // Should never return here
}

static void crt_work_loop(fcontext_transfer_t fiber_data) {
    CURRENT_WS_INTERNAL->root_ctx = fiber_data.prev_context;
    core_work_loop(); // this function never returns
    HASSERT(0); // Should never return here
}

static void *worker_routine(void *args) {
    const int wid = *((int *) args);
    set_current_worker(wid);

    // With the addition of lightweight context switching, worker creation
    // becomes a bit more complicated because we need all task creation and
    // finish scopes to be performed from beneath an explicitly created
    // context, rather than from a pthread context. To do this, we start
    // worker_routine by creating a proxy context to switch from and create
    // a lightweight context to switch to, which enters crt_work_loop
    // immediately, moving into the main work loop, eventually swapping
    // back to the proxy task to clean up this worker thread when the
    // worker thread is signaled to exit.
    if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIBERS) {
         // Create the new fiber we will be switching to,
         // which will start with crt_work_loop at the top of the stack.
        _fiber_suspend(NULL, crt_work_loop, NULL);
    }

    else {
        hclib_worker_state *ws = CURRENT_WS_INTERNAL;

        do {
            find_and_run_task(ws);

            if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_THREADS) {
                // context may have swapped
                ws = CURRENT_WS_INTERNAL;
            }
        } while (hclib_context->done_flags[ws->id].flag);
    }

    if (HCLIB_DO_THREAD_COUNTING) {
        // context may have swapped
        CURRENT_WS_INTERNAL->t = pthread_self();
        _hclib_atomic_inc_release(&_hclib_finished_workers);
    }

    return NULL;
}

static void _finish_ctx_resume(void *arg) {
    fcontext_t finishCtx = arg;
    _fiber_exit(_get_curr_fiber(), finishCtx);
    HASSERT(0); // UNREACHABLE
}

// Based on _help_finish_ctx
static void _help_wait(fcontext_transfer_t fiber_data) {
    hclib_future_t **continuation_deps = fiber_data.data;
    fcontext_t wait_ctx = fiber_data.prev_context;

    // reusing _finish_ctx_resume
    hclib_async(_finish_ctx_resume, wait_ctx, continuation_deps,
            NO_PHASER, ANY_PLACE, ESCAPING_ASYNC);

    core_work_loop();
    HASSERT(0); // UNREACHABLE
}

void *hclib_future_wait(hclib_future_t *future) {

    if (_hclib_promise_is_satisfied(future->owner)) {
        return future->owner->datum;
    }

    if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIBERS
            || HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_THREADS) {
        // save current finish scope (in case of worker swap)
        finish_t *current_finish = CURRENT_WS_INTERNAL->current_finish;

        if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIBERS) {
            hclib_future_t *continuation_deps[] = { future, NULL };
            _fiber_suspend(_get_curr_fiber(), _help_wait, continuation_deps);
        }
        else {
            HASSERT(HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_THREADS);
            _swap_blocked_thread(NULL, future);
        }

        // restore current finish scope (in case of worker swap)
        CURRENT_WS_INTERNAL->current_finish = current_finish;
    }

    else {
        HASSERT(HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIXED);

        if (HCLIB_WORKER_OPTIONS & HCLIB_WORKER_OPTIONS_HELP_GLOBAL) {
            hclib_worker_state *ws = CURRENT_WS_INTERNAL;
            // save current finish scope (in case of worker swap)
            finish_t *current_finish = ws->current_finish;

            while (!_hclib_promise_is_satisfied(future->owner)) {
                if (!_try_run_local(ws)) {
                    while (!_hclib_promise_is_satisfied(future->owner)) {
                        _try_run_global(ws);
                    }
                    break;
                }
            }

            // restore current finish scope (in case of worker swap)
            CURRENT_WS_INTERNAL->current_finish = current_finish;
        }
        else {
            while (!_hclib_promise_is_satisfied(future->owner)) {
                // spin-wait
            }
        }

    }

    HASSERT(_hclib_promise_is_satisfied(future->owner) &&
            "promise must be satisfied before returning from wait");

    return future->owner->datum;
}

static void _help_finish_ctx(fcontext_transfer_t fiber_data) {
    /*
     * Set up previous context to be stolen when the finish completes (note that
     * the async must ESCAPE, otherwise this finish scope will deadlock on
     * itself).
     */
    finish_t *finish = fiber_data.data;
    fcontext_t hclib_finish_ctx = fiber_data.prev_context;
    LOG_DEBUG("_help_finish_ctx: ctx = %p, ctx->arg = %p\n",
              hclib_finish_ctx, finish);

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

static inline void _worker_finish_help(finish_t *finish) {
    if ((HCLIB_WORKER_OPTIONS) & HCLIB_WORKER_OPTIONS_HELP_FINISH) {
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
        } while (finish->counter > 1);
    }
}


void help_finish(finish_t *finish) {
    // This is called to make progress when an end_finish has been
    // reached but it hasn't completed yet.
    // Note that's also where the master worker ends up entering its work loop

    // Try helping the current finish scope first
    // (internally checks if this option is enabled)
    _worker_finish_help(finish);

    if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIBERS
            || HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_THREADS) {
        /*
         * Creating a new context to switch to is necessary here because the
         * current context needs to become the continuation for this finish
         * (which will be switched back to by _finish_ctx_resume, for which an
         * async is created inside _help_finish_ctx).
         */

        if (_hclib_atomic_load_relaxed(&finish->counter) > 1) {
            // Someone stole our last task...
            // Create a new context to do other work,
            // and suspend this finish scope pending on the outstanding tasks.
            if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIBERS) {
                // create finish event
                hclib_promise_t finish_promise;
                hclib_promise_init(&finish_promise);
                hclib_future_t *finish_deps[] = { &finish_promise.future, NULL };
                finish->finish_deps = finish_deps;

                LOG_DEBUG("help_finish: finish = %p\n", finish);
                _fiber_suspend(_get_curr_fiber(), _help_finish_ctx, finish);
                // note: the other context checks out of the current finish scope
            }
            else {
                HASSERT(HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_THREADS);
                _swap_finish_thread(finish);
            }
        } else {
            HASSERT(_hclib_atomic_load_relaxed(&finish->counter) == 1);
            // finish->counter == 1 implies that all the tasks are done
            // (it's only waiting on itself now), so just return!
            _hclib_atomic_dec_acq_rel(&finish->counter);
        }
    }

    else {
        HASSERT(HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIXED);
        check_out_finish(finish);
        if (HCLIB_WORKER_OPTIONS & HCLIB_WORKER_OPTIONS_HELP_GLOBAL) {
            // Try helping globally (to avoid spin-waiting)
            // (internally checks if this option is enabled)
            hclib_worker_state *ws = CURRENT_WS_INTERNAL;
            while (_hclib_atomic_load_relaxed(&finish->counter) > 0) {
                if (!_try_run_local(ws)) {
                    while (_hclib_atomic_load_relaxed(&finish->counter) > 0) {
                        _try_run_global(ws);
                    }
                    break;
                }
            }
        }
        else {
            while (_hclib_atomic_load_relaxed(&finish->counter) > 0) {
                // spin-wait
            }
        }
    }

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
    finish->finish_deps = NULL;
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

    if (HC_DEBUG_ENABLED) {
        // Read worker strategy value from environment
        const char *env_strategy = getenv("HCLIB_SET_WORKER_STRATEGY");
        if (env_strategy) {
            _hclib_worker_strategy_val = strtol(env_strategy, NULL, 0);
        }
        // Read worker strategy options value from environment
        const char *env_options = getenv("HCLIB_SET_WORKER_OPTIONS");
        if (env_options) {
            _hclib_worker_options_val = strtol(env_options, NULL, 0);
        }
    }

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
    if (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIBERS) {
        _fiber_suspend(NULL, _hclib_finalize_ctx, NULL);
    }
    else {
        hclib_end_finish();
        // Signal shutdown to all worker threads
        hclib_signal_join(hclib_context->nworkers);
    }

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
 * create a fiber from the system-managed stacks and save them, but to not do
 * so when the calling context is already a fiber. While this could be
 * supported, this introduces unnecessary complexity into the runtime code. It
 * is simpler to use hclib_launch to ensure that finish scopes are only ever
 * reached from a fiber context, allowing us to assume that it is safe to simply
 * swap out the current context as a continuation without having to check if we
 * need to do extra work to persist it.
 */

void hclib_launch(generic_frame_ptr fct_ptr, void *arg) {
    hclib_init();
    hclib_async(fct_ptr, arg, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    hclib_finalize();
}

/**
 * @brief Get current worker context-management strategy value.
 * See inc/hclib-worker-config.h for the definitions of possible values.
 */
int64_t hclib_worker_strategy() {
    return _hclib_worker_strategy_val;
}

/**
 * @brief Get current worker strategy options mask value.
 * See inc/hclib-worker-config.h for the definitions of possible values.
 */
int64_t hclib_worker_options() {
    return _hclib_worker_options_val;
}
