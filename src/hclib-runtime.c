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

#ifndef __USE_GNU
#define __USE_GNU
#endif
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <dlfcn.h>

#include <hclib.h>
#include <hclib-internal.h>
#include <hclib-atomics.h>
#include <hclib-finish.h>
#include <hclib-hpt.h>
#include <hclib-cuda.h>
#include <hclib-locality-graph.h>
#include <hclib-module.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

static double user_specified_timer = 0;
// TODO use __thread on Linux?
pthread_key_t ws_key;

hclib_context *hc_context = NULL;

static int profile_launch_body = 0;
#ifdef HCLIB_STATS
typedef struct _per_worker_stats {
    size_t count_tasks;
    size_t count_steals;

    /*
     * Blocking operations that imply context creation and switching, which can
     * be expensive.
     */
    size_t count_end_finishes;
    size_t count_future_waits;
    size_t count_end_finishes_nonblocking;
    size_t count_ctx_creates;
} per_worker_stats;
static per_worker_stats *worker_stats = NULL;
#endif

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

static unsigned long long current_time_ns() {
#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    unsigned long long s = 1000000000ULL * (unsigned long long)mts.tv_sec;
    return (unsigned long long)mts.tv_nsec + s;
#else
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (((unsigned long long)t.tv_nsec)) + s;
#endif
}

unsigned long long hclib_current_time_ns() {
    return current_time_ns();
}

unsigned long long hclib_current_time_ms() {
    return current_time_ns() / 1000000;
}


static void set_current_worker(int wid) {
    int err;
    if ((err = pthread_setspecific(ws_key, hc_context->workers[wid])) != 0) {
        log_die("Cannot set thread-local worker state");
    }

    /*
     * don't bother worrying about core affinity on Mac OS since no one will be
     * running performance tests there anyway and it doesn't support
     * pthread_setaffinity_np.
     */
#ifndef __MACH__
    /*
     * Using pthread_setaffinity_np can interfere with other tools trying to
     * control affinity (e.g. if you are using srun/aprun/taskset from outside
     * the HClib process). For now we disable this.
     */
#if 0
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if (wid >= hc_context->ncores) {
        /*
         * If we are spawning more worker threads than there are cores, allow
         * the extras to float around.
         */
        int i;
        for (i = 0; i < hc_context->ncores; i++) {
            CPU_SET(i, &cpu_set);
        }
    } else {
        // Pin worker i to core i
        CPU_SET(wid, &cpu_set);
    }

    if ((err = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set),
                &cpu_set)) != 0) {
        fprintf(stderr, "WARNING: Failed setting pthread affinity of worker "
                "thread %d, ncores=%d: %s\n", wid, hc_context->ncores,
                strerror(err));
    }
#endif
#endif
}

int hclib_get_current_worker() {
    hclib_worker_state *ws = (hclib_worker_state *)pthread_getspecific(ws_key);
    assert(ws);
    return ws->id;
}

unsigned hclib_get_current_worker_pending_work() {
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

hclib_locale_t *default_dist_func(const int dim,
        const hclib_loop_domain_t *subloops, const hclib_loop_domain_t *loops,
        const int mode) {
    static hclib_locale_t *central_place = NULL;
    if (!central_place) {
        central_place = hclib_get_central_place();
    }
    return central_place;
}

/*
 * Main initialization function for the hclib_context object.
 */
void hclib_global_init() {
    /*
     * Build queues, this returns to us a tree structure of place and worker
     * nodes representing the hardware available on the current system. We
     * augment this tree based on module registrations to add logical (rather
     * than physical) places.
     */
    int nworkers;
    hclib_locality_graph *graph;
    hclib_worker_paths *worker_paths;

    const char *locality_graph_path = getenv("HCLIB_LOCALITY_FILE");
    if (locality_graph_path) {
        load_locality_info(locality_graph_path, &nworkers, &graph,
                &worker_paths);
    } else {
        fprintf(stderr, "WARNING: HCLIB_LOCALITY_FILE not provided, generating "
                "sane default locality information\n");
        generate_locality_info(&nworkers, &graph, &worker_paths);
    }
    check_locality_graph(graph, worker_paths, nworkers);

#ifdef VERBOSE
    print_locality_graph(graph);
    print_worker_paths(worker_paths, nworkers);
#endif

    hc_context->ncores = sysconf(_SC_NPROCESSORS_ONLN);
    hc_context->nworkers = nworkers;
    hc_context->graph = graph;
    hc_context->worker_paths = worker_paths;
    const int perr = posix_memalign((void **)&hc_context->done_flags, 64,
            nworkers * sizeof(worker_done_t));
    HASSERT(perr == 0);
    hc_context->workers = (hclib_worker_state **)calloc(nworkers,
            sizeof(hclib_worker_state *));
    assert(hc_context->workers);

    for (int i = 0; i < hc_context->nworkers; i++) {
        hclib_worker_state *ws = (hclib_worker_state *)calloc(1,
                sizeof(hclib_worker_state));
        ws->context = hc_context;
        ws->id = i;
        ws->nworkers = hc_context->nworkers;
        ws->paths = worker_paths + i;
        hc_context->done_flags[i].flag = 1;
        hc_context->workers[i] = ws;
    }
}

static void load_dependencies(const char **module_dependencies,
        int n_module_dependencies) {
    int i;
    char *hclib_home = getenv("HCLIB_HOME");
    if (hclib_home == NULL) {
        fprintf(stderr, "Expected environment variable HCLIB_HOME to be set to "
                "the root directory of the checked out HClib repo\n");
        exit(1);
    }

    char module_path_buf[1024];
    for (i = 0; i < n_module_dependencies; i++) {
        const char *module_name = module_dependencies[i];
        sprintf(module_path_buf, "%s/modules/%s/lib/libhclib_%s.so", hclib_home,
                module_name, module_name);
        void *handle = dlopen(module_path_buf, RTLD_LAZY);
        if (handle == NULL) {
            fprintf(stderr, "Failed dynamically loading %s for \"%s\" "
                    "dependency\n", module_path_buf, module_name);
            exit(1);
        }
    }
}

static void hclib_entrypoint(const char **module_dependencies,
        int n_module_dependencies) {
    /*
     * Assert that the completion flag structures are each on separate cache
     * lines.
     */
    HASSERT(sizeof(worker_done_t) == 64);

    load_dependencies(module_dependencies, n_module_dependencies);

    hclib_call_module_pre_init_functions();

    srand(0);

    hc_context = (hclib_context *)malloc(sizeof(hclib_context));
    HASSERT(hc_context);

    /*
     * Parse the platform description from the HPT configuration file and load
     * it into the hclib_context.
     */
    hclib_global_init();

    // init timer stats, TODO make this account for resource management threads
    hclib_init_stats(0, hc_context->nworkers);

    /* Create key to store per thread worker_state */
    if (pthread_key_create(&ws_key, NULL) != 0) {
        log_die("Cannot create ws_key for worker-specific data");
    }

    /*
     * set pthread's concurrency. Doesn't seem to do much on Linux, only
     * relevant when there are more pthreads than hardware cores to schedule
     * them on. */
    pthread_setconcurrency(hc_context->nworkers);

#ifdef HCLIB_STATS
    worker_stats = (per_worker_stats *)calloc(hc_context->nworkers,
            sizeof(per_worker_stats));
    HASSERT(worker_stats);
#endif

    // Launch the worker threads
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        fprintf(stderr, "Error in pthread_attr_init\n");
        exit(3);
    }

    // Start workers
    for (int i = 1; i < hc_context->nworkers; i++) {
        if (pthread_create(&hc_context->workers[i]->t, &attr, worker_routine,
                           &hc_context->workers[i]->id) != 0) {
            fprintf(stderr, "Error launching thread\n");
            exit(4);
        }

    }
    set_current_worker(0);

    const unsigned dist_id = hclib_register_dist_func(default_dist_func);
    HASSERT(dist_id == HCLIB_DEFAULT_LOOP_DIST);

    // Initialize any registered modules
    hclib_call_module_post_init_functions();

    // allocate root finish
    hclib_start_finish();
}

void hclib_signal_join(int nb_workers) {
    int i;
    for (i = 0; i < nb_workers; i++) {
        hc_context->done_flags[i].flag = 0;
    }
}

void hclib_join(int nb_workers) {
    // Join the workers
#ifdef VERBOSE
    fprintf(stderr, "hclib_join: nb_workers = %d\n", nb_workers);
#endif
    for (int i = 1; i < nb_workers; i++) {
        pthread_join(hc_context->workers[i]->t, NULL);
    }
#ifdef VERBOSE
    fprintf(stderr, "hclib_join: finished\n");
#endif
}

void hclib_cleanup() {
    pthread_key_delete(ws_key);

    hclib_call_finalize_functions();

    free(hc_context);
}

static inline void check_in_finish(finish_t *finish) {
    if (finish) {
        hc_atomic_inc(&(finish->counter));
    }
}

static inline void check_out_finish(finish_t *finish) {
    if (finish) {
        // hc_atomic_dec returns true when finish->counter goes to zero
        if (hc_atomic_dec(&(finish->counter))) {
            hclib_promise_put(finish->finish_dep->owner, finish);
        }
    }
}

static inline void execute_task(hclib_task_t *task) {
    finish_t *current_finish = get_current_finish(task);
    /*
     * Update the current finish of this worker to be inherited from the
     * currently executing task so that any asyncs spawned from the currently
     * executing task are registered on the same finish.
     */
    CURRENT_WS_INTERNAL->current_finish = current_finish;
#ifdef VERBOSE
    fprintf(stderr, "execute_task: setting current finish of %p to %p for task "
            "%p\n", CURRENT_WS_INTERNAL, current_finish, task);
    fprintf(stderr, "execute_task: task=%p fp=%p\n", task, task->_fp);
#endif

#ifdef HCLIB_STATS
    worker_stats[CURRENT_WS_INTERNAL->id].count_tasks++;
#endif

    // task->_fp is of type 'void (*generic_frame_ptr)(void*)'
    (task->_fp)(task->args);
    check_out_finish(current_finish);
    HC_FREE(task);
}

static inline void rt_schedule_async(hclib_task_t *async_task,
        hclib_worker_state *ws) {
#ifdef VERBOSE
    fprintf(stderr, "rt_schedule_async: async_task=%p locale=%p\n", async_task,
            async_task->locale);
#endif

    if (async_task->locale) {
        // If task was explicitly created at a locale, place it there
        if (!deque_push_locale(ws, async_task->locale, async_task)) {
            assert(false);
        }
    } else {
        /*
         * If no explicit locale was provided, place it at a default location.
         * In the old implementation, each worker had the concept of a 'current'
         * locale. For now we just place at locale 0 by default, but having a
         * current locale might be a good thing to implement in the future.
         * TODO.
         */
        const int wid = hclib_get_current_worker();
#ifdef VERBOSE
        fprintf(stderr, "rt_schedule_async: scheduling on worker wid=%d "
                "hc_context=%p hc_context->graph=%p\n", wid, hc_context,
                hc_context->graph);
#endif
        if (!deque_push(&(hc_context->graph->locales[0].deques[wid].deque),
                    async_task)) {
            // Deque is full
            assert(false);
        }
#ifdef VERBOSE
        fprintf(stderr, "rt_schedule_async: finished scheduling on worker "
                "wid=%d\n", wid);
#endif
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
    fprintf(stderr, "is_eligible_to_schedule: async_task=%p singleton_future_0=%p\n",
            async_task, async_task->singleton_future_0);
#endif
    if (async_task->singleton_future_0 || async_task->singleton_future_1) {
        hclib_triggered_task_t *triggered_task = (hclib_triggered_task_t *)
                rt_async_task_to_triggered_task(async_task);
        return register_on_all_promise_dependencies(triggered_task);
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
#ifdef VERBOSE
    fprintf(stderr, "try_schedule_async: async_task=%p ws=%p\n", async_task, ws);
#endif
    if (is_eligible_to_schedule(async_task)) {
        rt_schedule_async(async_task, ws);
    }
}

void spawn_handler(hclib_task_t *task, hclib_locale_t *locale,
        hclib_future_t *singleton_future_0, hclib_future_t *singleton_future_1,
        const int escaping) {
    HASSERT(task);

    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    if (!escaping) {
        check_in_finish(ws->current_finish);
        set_current_finish(task, ws->current_finish);
    } else {
        // If escaping task, don't register with current finish
        set_current_finish(task, NULL);
    }

    if (locale) {
        task->locale = locale;
    }

    if (singleton_future_0) {
        task->singleton_future_0 = singleton_future_0;
        task->singleton_future_1 = singleton_future_1;
        hclib_dependent_task_t *t = (hclib_dependent_task_t *) task;
        hclib_triggered_task_init(&(t->deps), singleton_future_0, singleton_future_1);
    }

#ifdef VERBOSE
    fprintf(stderr, "spawn_handler: task=%p escaping=%d\n", task, escaping);
#endif

    try_schedule_async(task, ws);
}

void spawn_at(hclib_task_t *task, hclib_locale_t *locale) {
    spawn_handler(task, locale, NULL, NULL, 0);
}

void spawn(hclib_task_t *task) {
    spawn_handler(task, NULL, NULL, NULL, 0);
}

void spawn_escaping(hclib_task_t *task, hclib_future_t *future) {
    spawn_handler(task, NULL, future, NULL, 1);
}

void spawn_escaping_at(hclib_locale_t *locale, hclib_task_t *task,
        hclib_future_t *future) {
    spawn_handler(task, locale, future, NULL, 1);
}

void spawn_await_at(hclib_task_t *task, hclib_future_t *future1,
        hclib_future_t *future2, hclib_locale_t *locale) {
    spawn_handler(task, locale, future1, future2, 0);
}

void spawn_await(hclib_task_t *task, hclib_future_t *future1,
        hclib_future_t *future2) {
    spawn_await_at(task, future1, future2, NULL);
}

static hclib_task_t *find_and_run_task(hclib_worker_state *ws,
        const int on_fresh_ctx, volatile int *flag, const int flag_val) {
    hclib_task_t *task = locale_pop_task(ws);
    // volatile int *flag = &(hc_context->done_flags[ws->id].flag);
    if (!task) {
        while (*flag != flag_val) {
            // try to steal
            task = locale_steal_task(ws);
            if (task) {
#ifdef HCLIB_STATS
                worker_stats[ws->id].count_steals++;
#endif
                break;
            }
        }
    }

    if (task == NULL) {
        return NULL;
    } else if (task && (on_fresh_ctx || task->non_blocking)) {
        execute_task(task);
        return NULL;
    } else {
        return task;
    }
}

static void _hclib_finalize_ctx(LiteCtx *ctx) {
    hclib_end_finish();
    // Signal shutdown to all worker threads
    hclib_signal_join(hc_context->nworkers);
    // Jump back to the system thread context for this worker
    ctx_swap(ctx, CURRENT_WS_INTERNAL->root_ctx, __func__);
    HASSERT(0); // Should never return here
}

static void core_work_loop(hclib_task_t *starting_task) {

    if (starting_task) {
        execute_task(starting_task);
    }

    uint64_t wid;
    do {
        hclib_worker_state *ws = CURRENT_WS_INTERNAL;
        wid = (uint64_t)ws->id;
        hclib_task_t *must_be_null = find_and_run_task(ws, 1,
                &(hc_context->done_flags[wid].flag), 0);
        HASSERT(must_be_null == NULL);
    } while (hc_context->done_flags[wid].flag);

    // Jump back to the system thread context for this worker
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    HASSERT(ws->root_ctx);
    ctx_swap(get_curr_lite_ctx(), ws->root_ctx, __func__);
    HASSERT(0); // Should never return here
}

static void crt_work_loop(LiteCtx *ctx) {
    core_work_loop(NULL); // this function never returns
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
 * to clean up this worker thread when the worker thread is signalled to exit.
 */
static void *worker_routine(void *args) {
    const int wid = *((int *)args);
    set_current_worker(wid);
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;

    // Create proxy original context to switch from
    LiteCtx *currentCtx = LiteCtx_proxy_create(__func__);
    ws->root_ctx = currentCtx;

    /*
     * Create the new proxy we will be switching to, which will start with
     * crt_work_loop at the top of the stack.
     */
    LiteCtx *newCtx = LiteCtx_create(crt_work_loop);
    newCtx->arg1 = args;
#ifdef HCLIB_STATS
    worker_stats[CURRENT_WS_INTERNAL->id].count_ctx_creates++;
#endif

    // Swap in the newCtx lite context
    ctx_swap(currentCtx, newCtx, __func__);

#ifdef VERBOSE
    fprintf(stderr, "worker_routine: worker %d exiting, cleaning up proxy %p "
            "and lite ctx %p\n", hclib_get_current_worker(), currentCtx, newCtx);
#endif

    // free resources
    LiteCtx_destroy(currentCtx->prev);
    LiteCtx_proxy_destroy(currentCtx);
    return NULL;
}

static void _finish_ctx_resume(void *arg) {
    LiteCtx *currentCtx = get_curr_lite_ctx();
    LiteCtx *finishCtx = arg;
    ctx_swap(currentCtx, finishCtx, __func__);

    fprintf(stderr, "Should not have reached here, currentCtx=%p "
            "finishCtx=%p\n", currentCtx, finishCtx);
    HASSERT(0);
}

void crt_work_loop(LiteCtx *ctx);

/*
 * Based on _help_finish_ctx, _help_wait is called to swap out the current
 * context when a thread waits on a future.
 */
void _help_wait(LiteCtx *ctx) {
    hclib_future_t *continuation_dep = ctx->arg1;
    hclib_task_t *starting_task = ctx->arg2;
    LiteCtx *wait_ctx = ctx->prev;

    hclib_dependent_task_t *task = (hclib_dependent_task_t *)calloc(
            1, sizeof(hclib_dependent_task_t));
    HASSERT(task);
    task->async_task._fp = _finish_ctx_resume; // reuse _finish_ctx_resume
    task->async_task.args = wait_ctx;

    spawn_escaping((hclib_task_t *)task, continuation_dep);

    core_work_loop(starting_task);
    HASSERT(0);
}

int hclib_future_is_satisfied(hclib_future_t *future) {
    return future->owner->satisfied;
}

void *hclib_future_wait(hclib_future_t *future) {
    if (future->owner->satisfied) {
        return (void *)future->owner->datum;
    }
#ifdef HCLIB_STATS
    worker_stats[CURRENT_WS_INTERNAL->id].count_future_waits++;
#endif
    finish_t *current_finish = CURRENT_WS_INTERNAL->current_finish;

    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    hclib_task_t *need_to_swap_ctx = NULL;
    while (future->owner->satisfied == 0 &&
            need_to_swap_ctx == NULL) {
        // This is a bit hacky...
        need_to_swap_ctx = find_and_run_task(ws, 0,
                &(future->owner->satisfied), 1);
    }

    if (need_to_swap_ctx) {
        LiteCtx *currentCtx = get_curr_lite_ctx();
        HASSERT(currentCtx);
        LiteCtx *newCtx = LiteCtx_create(_help_wait);
        newCtx->arg1 = future;
        newCtx->arg2 = need_to_swap_ctx;
#ifdef HCLIB_STATS
        worker_stats[CURRENT_WS_INTERNAL->id].count_ctx_creates++;
#endif
        ctx_swap(currentCtx, newCtx, __func__);
        LiteCtx_destroy(currentCtx->prev);
    }

    // Restore the finish state from before the wait
    CURRENT_WS_INTERNAL->current_finish = current_finish;

    HASSERT(future->owner->satisfied);
    return (void *)future->owner->datum;
}

/*
 * _help_finish_ctx is the function we switch to on a new context when
 * encountering an end finish to allow the current hardware thread to make
 * useful progress.
 */
static void _help_finish_ctx(LiteCtx *ctx) {
    /*
     * Set up previous context to be stolen when the finish completes (note that
     * the async must ESCAPE, otherwise this finish scope will deadlock on
     * itself).
     */
#ifdef VERBOSE
    printf("_help_finish_ctx: ctx = %p, ctx->arg = %p\n", ctx, ctx->arg);
#endif
    finish_t *finish = ctx->arg1;
    hclib_task_t *starting_task = ctx->arg2;
    HASSERT(finish && starting_task);
    LiteCtx *hclib_finish_ctx = ctx->prev;

    hclib_dependent_task_t *task = (hclib_dependent_task_t *)calloc(
            1, sizeof(hclib_dependent_task_t));
    HASSERT(task);
    task->async_task._fp = _finish_ctx_resume;
    task->async_task.args = hclib_finish_ctx;

    /*
     * Create an async to handle the continuation after the finish, whose state
     * is captured in hclib_finish_ctx and whose execution is pending on
     * finish->finish_dep.
     */
    spawn_escaping((hclib_task_t *)task, finish->finish_dep);

    /*
     * The main thread is now exiting the finish (albeit in a separate context),
     * so check it out.
     */
    check_out_finish(finish);

    // keep workstealing until this context gets swapped out and destroyed
    core_work_loop(starting_task); // this function never returns
    HASSERT(0); // we should never return here
}

void help_finish(finish_t *finish) {
    /*
     * Creating a new context to switch to is necessary here because the
     * current context needs to become the continuation for this finish
     * (which will be switched back to by _finish_ctx_resume, for which an
     * async is created inside _help_finish_ctx).
     */

    if (finish->counter == 1) {
        /*
         * Quick optimization: if no asyncs remain in this finish scope, just
         * return. finish counter will be 1 here because we haven't checked out
         * the main thread (this thread) yet.
         */
        return;
    }

    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    hclib_task_t *need_to_swap_ctx = NULL;
    while (finish->counter > 1 && need_to_swap_ctx == NULL) {
        need_to_swap_ctx = find_and_run_task(ws, 0, &(finish->counter), 1);
    }

    if (need_to_swap_ctx) {
        // create finish event
        hclib_promise_t *finish_promise = hclib_promise_create();
        finish->finish_dep = &finish_promise->future;
        LiteCtx *currentCtx = get_curr_lite_ctx();
        HASSERT(currentCtx);
        LiteCtx *newCtx = LiteCtx_create(_help_finish_ctx);
        newCtx->arg1 = finish;
        newCtx->arg2 = need_to_swap_ctx;
#ifdef HCLIB_STATS
        worker_stats[CURRENT_WS_INTERNAL->id].count_ctx_creates++;
#endif

#ifdef VERBOSE
        printf("help_finish: newCtx = %p, newCtx->arg = %p\n", newCtx, newCtx->arg);
#endif
        ctx_swap(currentCtx, newCtx, __func__);
        /*
         * destroy the context that resumed this one since it's now defunct
         * (there are no other handles to it, and it will never be resumed)
         */
        LiteCtx_destroy(currentCtx->prev);
        hclib_promise_free(finish_promise);

        HASSERT(finish->counter == 0);
    } else {
        HASSERT(finish->counter == 1);
    }
}

/*
 * =================== INTERFACE TO USER FUNCTIONS ==========================
 */

void hclib_start_finish() {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    finish_t *finish = (finish_t *)calloc(1, sizeof(finish_t));
    HASSERT(finish);
    /*
     * Set finish counter to 1 initially to emulate the main thread inside the
     * finish being a task registered on the finish. When we reach the
     * corresponding end_finish we set up the finish_dep for the continuation
     * and then decrement the counter from the main thread. This ensures that
     * anytime the counter reaches zero, it is safe to do a promise_put on the
     * finish_dep. If we initialized counter to zero here, any async inside the
     * finish could start and finish before the main thread reaches the
     * end_finish, decrementing the finish counter to zero when it completes.
     * This would make it harder to detect when all tasks within the finish have
     * completed, or just the tasks launched so far.
     */
    finish->counter = 1;
    finish->parent = ws->current_finish;
    check_in_finish(finish->parent); // check_in_finish performs NULL check
    ws->current_finish = finish;

#ifdef VERBOSE
    fprintf(stderr, "hclib_start_finish: entering finish for %p and setting its current finish "
            "from %p to %p\n", ws, finish->parent, ws->current_finish);
#endif
}

void hclib_end_finish() {
    finish_t *current_finish = CURRENT_WS_INTERNAL->current_finish;
#ifdef VERBOSE
    fprintf(stderr, "hclib_end_finish: ending finish %p on worker %p\n",
            current_finish, CURRENT_WS_INTERNAL);
#endif
#ifdef HCLIB_STATS
    worker_stats[CURRENT_WS_INTERNAL->id].count_end_finishes++;
#endif

    HASSERT(current_finish);
    HASSERT(current_finish->counter > 0);
    help_finish(current_finish);

    check_out_finish(current_finish->parent); // NULL check in check_out_finish

#ifdef VERBOSE
    fprintf(stderr, "hclib_end_finish: out of finish, setting current finish "
            "of %p to %p from %p\n", CURRENT_WS_INTERNAL,
            current_finish->parent, current_finish);
#endif
    CURRENT_WS_INTERNAL->current_finish = current_finish->parent;
    HC_FREE(current_finish);
}

void hclib_end_finish_nonblocking_helper(hclib_promise_t *event) {
    finish_t *current_finish = CURRENT_WS_INTERNAL->current_finish;
#ifdef HCLIB_STATS
    worker_stats[CURRENT_WS_INTERNAL->id].count_end_finishes_nonblocking++;
#endif

    HASSERT(current_finish->counter > 0);

    // Based on help_finish
    current_finish->finish_dep = &event->future;

    // Check out this "task" from the current finish
    check_out_finish(current_finish);

    // Check out the current finish from its parent
    check_out_finish(current_finish->parent);
    CURRENT_WS_INTERNAL->current_finish = current_finish->parent;
#ifdef VERBOSE
    fprintf(stderr, "hclib_end_finish_nonblocking_helper: out of finish, "
            "setting current finish of %p to %p from %p\n", CURRENT_WS_INTERNAL,
            current_finish->parent, current_finish);
#endif
}

hclib_future_t *hclib_end_finish_nonblocking() {
    hclib_promise_t *event = hclib_promise_create();
    hclib_end_finish_nonblocking_helper(event);
    return &event->future;
}

int hclib_num_workers() {
    return hc_context->nworkers;
}

double mysecond() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + ((double) tv.tv_usec / 1000000);
}

void hclib_user_harness_timer(double dur) {
    user_specified_timer = dur;
}

/*
 * Main entrypoint for runtime initialization, this function must be called by
 * the user program before any HC actions are performed.
 */
static void hclib_init(const char **module_dependencies,
        int n_module_dependencies) {
    if (getenv("HCLIB_PROFILE_LAUNCH_BODY")) {
        profile_launch_body = 1;
    }

    hclib_entrypoint(module_dependencies, n_module_dependencies);
}


static void (* volatile save_fp)(void *) = NULL;
static void * volatile save_data = NULL;
static void * volatile save_context = NULL;
void hclib_run_on_main_ctx(void (*fp)(void *), void *data) {
    HASSERT(hclib_get_current_worker() == 0);
    /*
     * May trigger these is an operation on the main context calls another
     * operation on the main context.
     */
    HASSERT(save_fp == NULL); HASSERT(fp != NULL);

    save_fp = fp;
    save_data = data;
    save_context = get_curr_lite_ctx();

    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    ctx_swap(save_context, ws->root_ctx, __func__);

    save_fp = NULL;
    save_data = NULL;
    save_context = NULL;
}

/*
 * Get information on the number of pending tasks along this worker's pop path,
 * as an estimate of how much this worker should create new work or how much it
 * might want to save on overhead and continue executing sequentially.
 */
size_t hclib_current_worker_backlog() {
    hclib_worker_state *ws = CURRENT_WS_INTERNAL;
    return workers_backlog(ws);
}

static void hclib_finalize() {
    LiteCtx *finalize_ctx = LiteCtx_proxy_create(__func__);
    LiteCtx *finish_ctx = LiteCtx_create(_hclib_finalize_ctx);
#ifdef HCLIB_STATS
    worker_stats[CURRENT_WS_INTERNAL->id].count_ctx_creates++;
#endif
    CURRENT_WS_INTERNAL->root_ctx = finalize_ctx;
    ctx_swap(finalize_ctx, finish_ctx, __func__);
    while (save_fp) {
        save_fp(save_data);
        ctx_swap(finalize_ctx, save_context, __func__);
    }
    // free resources
    LiteCtx_destroy(finalize_ctx->prev);
    LiteCtx_proxy_destroy(finalize_ctx);

    hclib_join(hc_context->nworkers);

#ifdef HCLIB_STATS
    int i;
    printf("===== HClib statistics: =====\n");
    size_t sum_end_finishes = 0;
    size_t sum_future_waits = 0;
    size_t sum_end_finishes_nonblocking = 0;
    size_t sum_ctx_creates = 0;
    for (i = 0; i < hc_context->nworkers; i++) {
        printf("  Worker %d: %lu tasks, %lu steals\n", i,
                worker_stats[i].count_tasks, worker_stats[i].count_steals);
        sum_end_finishes += worker_stats[i].count_end_finishes;
        sum_future_waits += worker_stats[i].count_future_waits;
        sum_end_finishes_nonblocking += worker_stats[i].count_end_finishes_nonblocking;
        sum_ctx_creates += worker_stats[i].count_ctx_creates;
    }
    printf("Total: %lu end finishes, %lu future waits, %lu non-blocking end "
            "finishes, %lu ctx creates\n", sum_end_finishes, sum_future_waits,
            sum_end_finishes_nonblocking, sum_ctx_creates);
    free(worker_stats);
#endif

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

void hclib_launch(generic_frame_ptr fct_ptr, void *arg, const char **deps,
        int ndeps) {
    unsigned long long start_time = 0;
    unsigned long long end_time;

    hclib_init(deps, ndeps);

    if (profile_launch_body) {
        start_time = current_time_ns();
    }
    hclib_async(fct_ptr, arg, NO_FUTURE, ANY_PLACE);
    hclib_finalize();
    if (profile_launch_body) {
        end_time = current_time_ns();
        printf("\nHCLIB TIME %llu ns\n", end_time - start_time);
    }
}

