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

static double benchmark_start_time_stats = 0;
static double user_specified_timer = 0;
// TODO use __thread on Linux?
pthread_key_t ws_key;

#ifdef HCPP_COMM_WORKER
semiConcDeque_t * comm_worker_out_deque;
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

/**
 * current - current context pointer
 * next - target context pointer
 * return - pointer to the current context,
 *   with the prev field set to the source context's pointer
 *
 * LiteCtx_swap is used to either swap in a newly created context on the current
 * thread (with an entrypoint function specified) or to switch back to a
 * previously created context.
 *
 * Swapping to a new context occurs in the following scenarios:
 *   1. When creating an initial new lite context as part of hclib_finalize,
 *      under which we perform the global hclib_end_finish.
 *   2. At the entrypoint of each worker thread, to create a lite context for
 *      all worker thread async and finishes to be performed under.
 *   3. From help_finish (called by end_finish), which creates a new lite
 *      context to switch to so that the current stack can be set aside as a
 *      continuation.
 *
 * Swapping back to a previously created context occurs in the following
 * scenarios:
 *   1. At the end of _hclib_finalize_ctx, as cleanup of the temporary lite
 *      context created for hclib_finalize.
 *   2. At the end of crt_work_loop, we switch back to the lite context that
 *      created the current lite context.
 *   3. In the escaping async created that is dependent on each finish, its only
 *      task is to swap out the current stack to the context for the
 *      continuation.
 *
 * NOTE: It is important to know that the boost::context library is designed so
 * that a fiber exiting its main entrypoint function will immediately call
 * exit(0). Therefore, it is important to be careful at the end of a fiber
 * entrypoint to always know what context to switch back to. If you do not swap
 * in another context, the entrypoint will exit and then your application will
 * exit silently with an exit code of zero. This can be hugely painful to debug.
 * It is good practice to end any function that acts as the entrypoint for a
 * fiber with an 'assert(0)' to ensure that if you do hit this case you get a
 * more sensible error message.
 */
static __inline__ void LiteCtx_swap(LiteCtx *current, LiteCtx *next,
        const char *lbl) {
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_swap[%s]: wid=%d current=%p(%p) next=%p(%p)\n",
            lbl, get_current_worker(), current, current->_fctx.sp, next,
            next->_fctx.sp);
#endif
    next->prev = current;
    LiteCtx *new_current = jump_fcontext(&current->_fctx, next->_fctx, next,
            false);
#ifdef VERBOSE
    fprintf(stderr, "LiteCtx_swap: swapped back in %p(%p)\n", new_current,
            new_current->_fctx.sp);
#endif
    /*
     * When we reach this code, we have returned from swapping out current for
     * next, and someone else has swapped back in current.
     */
    set_curr_lite_ctx(new_current);
}

hc_workerState* current_ws() {
    return CURRENT_WS_INTERNAL;
}

// FWD declaration for pthread_create
static void* worker_routine(void * args);

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

#ifdef HCPP_COMM_WORKER
    comm_worker_out_deque = (semiConcDeque_t *)malloc(sizeof(semiConcDeque_t));
    HASSERT(comm_worker_out_deque);
    semiConcDequeInit(comm_worker_out_deque, NULL);
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
    /* setting current thread as worker 0 */
    // Launch the worker threads
    if (hcpp_stats) {
        printf("Using %d worker threads (including main thread)\n", nb_workers);
    }

    // Start workers
    for (int i = 1; i < nb_workers; i++) {
        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0) {
            fprintf(stderr, "Error in pthread_attr_init\n");
            exit(3);
        }
        if (pthread_create(&hcpp_context->workers[i]->t, &attr, worker_routine,
                &hcpp_context->workers[i]->id) != 0) {
            fprintf(stderr, "Error launching thread\n");
            exit(4);
        }
    }
    set_current_worker(0);
}

static void display_runtime() {
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

#ifdef HCPP_COMM_WORKER
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

    /* Create all worker threads, running worker_routine */
    hcpp_create_worker_threads(hcpp_context->nworkers);

    // allocate root finish
    hclib_start_finish();
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

static inline void rt_schedule_async(task_t* async_task, int comm_task) {
    if(comm_task) {
#ifdef HCPP_COMM_WORKER
        // push on comm_worker out_deq if this is a communication task
        semiConcDequeLockedPush(comm_worker_out_deque, async_task);
#endif
    } else {
        // push on worker deq
        const int wid = get_current_worker();
        if (!dequePush(&(hcpp_context->workers[wid]->current->deque),
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
 * registered on each, and it is only places in a work deque once all DDFs have
 * been satisfied.
 */
inline int is_eligible_to_schedule(task_t * async_task) {
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
void try_schedule_async(task_t * async_task, int comm_task) {
    if (is_eligible_to_schedule(async_task)) {
        rt_schedule_async(async_task, comm_task);
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
    try_schedule_async(task, 0);
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
    try_schedule_async(task, 0);
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
	try_schedule_async(task, 0);
#ifdef HC_COMM_WORKER_STATS
	const int wid = get_current_worker();
	increment_async_counter(wid);
#endif

}

void spawn_commTask(task_t * task) {
#ifdef HCPP_COMM_WORKER
	hc_workerState* ws = CURRENT_WS_INTERNAL;
	check_in_finish(ws->current_finish);
	set_current_finish(task, ws->current_finish);
	try_schedule_async(task, 1);
#else
	assert(0);
#endif
}

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

#ifdef HCPP_COMM_WORKER
inline void master_worker_routine(finish_t* finish) {
	semiConcDeque_t *deque = comm_worker_out_deque;
	while(finish->counter > 0) {
		// try to pop
		task_t* task = semiConcDequeNonLockedPop(deque);
		// Comm worker cannot steal
		if(task) {
#ifdef HC_COMM_WORKER_STATS
			increment_asyncComm_counter();
#endif
			execute_task(task);
		}
	}
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
    set_curr_lite_ctx(ctx);
    LiteCtx *main_thread = ctx->prev;

    hclib_end_finish();
    // Signal shutdown to all worker threads
    hcpp_signal_join(hcpp_context->nworkers);
    /*
     * If we are the main thread, then simply switch back to the main
     * application thread. If we are a worker thread, switch back to the pthread
     * entrypoint worker_routine instead. If a worker thread ends up picking up
     * this continuation, that implies the main thread is currently off
     * somewhere in core_work_loop and will be signalled by hcpp_signal_join
     * causing it to jump back to root_ctx (which for the main thread is
     * equivalent to main_thread here).
     */
    if (get_current_worker() == 0) {
        LiteCtx_swap(ctx, main_thread, "_hclib_finalize_ctx");
    } else {
        LiteCtx_swap(get_curr_lite_ctx(), CURRENT_WS_INTERNAL->root_ctx,
                "core_work_loop");
    }
    assert(0); // Should never return here
}

static void core_work_loop() {
    uint64_t wid;
    do {
        hc_workerState *ws = CURRENT_WS_INTERNAL;
        wid = (uint64_t)ws->id;
        find_and_run_task(ws);
    } while (hcpp_context->done_flags[wid].flag);

    // Jump back to the context for worker_routine
    hc_workerState *ws = CURRENT_WS_INTERNAL;
    assert(ws->root_ctx);
    LiteCtx_swap(get_curr_lite_ctx(), ws->root_ctx, "core_work_loop");
}

static void crt_work_loop(LiteCtx *ctx) {
    set_curr_lite_ctx(ctx);
    LiteCtx *original = ctx->prev;
    core_work_loop();
    /*
     * switch back to whichever thread created this work loop context, either
     * the main entrypoint of a worker thread (worker_routine) or a thread that
     * hit an end finish.
     */
    LiteCtx_swap(ctx, original, "crt_work_loop");
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

    // Create proxy original context to switch from
    LiteCtx *currentCtx = LiteCtx_proxy_create("worker_routine");
    ws->root_ctx = currentCtx;

    /*
     * Create the new proxy we will be switching to, which will start with
     * crt_work_loop at the top of the stack.
     */
    LiteCtx *newCtx = LiteCtx_create(crt_work_loop);
    newCtx->arg = args;

    // Swap in the newCtx lite context
    LiteCtx_swap(currentCtx, newCtx, "worker_routine");

#ifdef VERBOSE
    fprintf(stderr, "worker_routine: worker %d exiting, cleaning up proxy %p "
            "and lite ctx %p\n", get_current_worker(), currentCtx, newCtx);
#endif

    // free resources
    LiteCtx_destroy(newCtx);
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
    /*
     * TODO I believe we will leak currentCtx here, we may want to add a flag on
     * switching contexts that says to destroy the previous context, and set it
     * here.
     */
    LiteCtx *currentCtx = get_curr_lite_ctx();
    LiteCtx *finishCtx = arg;
    LiteCtx_swap(currentCtx, finishCtx, "_finish_ctx_resume");

    fprintf(stderr, "Should not have reached here, currentCtx=%p "
            "finishCtx=%p\n", currentCtx, finishCtx);
    assert(0);
}

void crt_work_loop(LiteCtx *ctx);

static void _help_finish_ctx(LiteCtx *ctx) {
    // Remember the current context
    set_curr_lite_ctx(ctx);
    // Set up previous context to be stolen when the finish completes
    // (note that the async must ESCAPE, otherwise this finish scope will deadlock on itself)
    // finish_t *finish = ((volatile LiteCtx * volatile)ctx)->arg;
    finish_t *finish = ctx->arg;
    LiteCtx *hclib_finish_ctx = ctx->prev;

    hcpp_task_t *task = (hcpp_task_t *)malloc(sizeof(hcpp_task_t));
    task->async_task._fp = _finish_ctx_resume;
    task->async_task.is_asyncAnyType = 0;
    task->async_task.ddf_list = NULL;
    task->async_task.args = hclib_finish_ctx;

    spawn_escaping((task_t *)task, finish->finish_deps);

    // keep workstealing until this context gets swapped out and destroyed
    check_out_finish(finish);
    core_work_loop();
    assert(0); // This is the entrypoint of a fiber, so we should never return here.
}
#else /* default (broken) strategy */
static void _help_finish(finish_t * finish) {
#ifdef HCPP_COMM_WORKER
	if(CURRENT_WS_INTERNAL->id == 0) {
		master_worker_routine(finish);
	}
	else {
		slave_worker_finishHelper_routine(finish);
	}
#else
	slave_worker_finishHelper_routine(finish);
#endif
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
        LiteCtx_swap(currentCtx, newCtx, "help_finish");
        // free resources
        // LiteCtx_destroy(newCtx);
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
    LiteCtx *finalize_ctx = LiteCtx_proxy_create("hclib_finalize");
    LiteCtx *finish_ctx = LiteCtx_create(_hclib_finalize_ctx);
    CURRENT_WS_INTERNAL->root_ctx = finalize_ctx;
    LiteCtx_swap(finalize_ctx, finish_ctx, "hclib_finalize");
    // free resources
    LiteCtx_destroy(finish_ctx);
    // LiteCtx_proxy_destroy(finalize_ctx);
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
