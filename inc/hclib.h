/* Copyright (c) 2013, Rice University

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

#ifndef HCLIB_H_
#define HCLIB_H_

#include "hclib_common.h"
#include "hclib-task.h"
#include "hclib-promise.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file Interface to HCLIB
 */

/**
 * @defgroup HClib Finish/Async/Forasync
 * @brief Core API, Finish/Async/Forasync for structured parallelism.
 *
 * @{
 **/

/**
 * @brief Function prototype executable by an async.
 * @param[in] arg           Arguments to the function
 */
typedef void (*async_fct_t)(void * arg);
typedef void *(*future_fct_t)(void *arg);

size_t hclib_current_worker_backlog();

void hclib_launch(async_fct_t fct_ptr, void * arg, const char **deps,
        int ndeps);

/**
 * Time keeping utilities.
 */
unsigned long long hclib_current_time_ns();
unsigned long long hclib_current_time_ms();

/**
 * Register a function to be called when a thread in the hclib runtime is idle,
 * i.e. is unable to find work through the hclib deques via either popping or
 * stealing. This method can be used by the user to create more work for the
 * runtime to do.
 */
void hclib_set_idle_callback(void (*set_idle_callback)(unsigned, unsigned));

void hclib_run_on_main_ctx(void (*fp)(void *), void *data);

#define HCLIB_DEFAULT_LOOP_DIST 0

unsigned hclib_register_dist_func(loop_dist_func func);

loop_dist_func hclib_lookup_dist_func(unsigned id);

/*
 * Async definition and API
 */

// forward declaration for promise_st in hclib-promise.h
struct hclib_promise_st;

/**
 * @brief Spawn a new task asynchronously.
 * @param[in] fct_ptr           The function to execute
 * @param[in] arg               Argument to the async
 * @param[in] future_list       The list of promises the async depends on
 * @param[in] property          Flag to pass information to the runtime
 */
void hclib_async(generic_frame_ptr fp, void *arg,
        hclib_future_t **futures, const int nfutures,
        hclib_locale_t *locale);

/**
 * A variant of hclib_async that promises the created task will not block (i.e.
 * will not wait on a future, close a finish scope, etc.)
 */
void hclib_async_nb(generic_frame_ptr fp, void *arg, hclib_locale_t *locale);

/*
 * Spawn an async that automatically puts a promise on termination.
 */
hclib_future_t *hclib_async_future(future_fct_t fp, void *arg,
        hclib_future_t **futures, const int nfutures, hclib_locale_t *locale);

/*
 * Locale-aware memory management functions.
 */
hclib_future_t *hclib_allocate_at(size_t nbytes, hclib_locale_t *locale);
hclib_future_t *hclib_reallocate_at(void *ptr, size_t new_nbytes,
        hclib_locale_t *locale);
hclib_future_t *hclib_memset_at(void *ptr, int pattern, size_t nbytes,
        hclib_locale_t *locale);
void hclib_free_at(void *ptr, hclib_locale_t *locale);

/*
 * If HCLIB_ASYNC_COPY_USE_FUTURE_AS_SRC is passed to the src parameter of
 * hclib_async_copy, this function will assume (and assert) that it is also
 * being passed a single future to wait on using futures/nfutures. It will use
 * the value returned by a get() on that future as the source argument for this
 * copy. This can be useful when the operation this copy is waiting on is also
 * determining the source location for the copy, and allows for a more concise
 * expression of this pattern.
 */
#define HCLIB_ASYNC_COPY_USE_FUTURE_AS_SRC (void *)0x1
hclib_future_t *hclib_async_copy(hclib_locale_t *dst_locale, void *dst,
        hclib_locale_t *src_locale, void *src, size_t nbytes,
        hclib_future_t **futures, const int nfutures);

/*
 * Forasync definition and API
 */

/** @brief forasync mode to control chunking strategy. */
typedef int forasync_mode_t;

/** @brief Forasync mode to recursively chunk the iteration space. */
#define FORASYNC_MODE_RECURSIVE 1
/** @brief Forasync mode to perform static chunking of the iteration space. */
#define FORASYNC_MODE_FLAT 0
/** @brief To indicate an async need not register with any finish scopes. */
#define ESCAPING_ASYNC ((int) 0x2)
#define COMM_ASYNC     ((int) 0x4)

/**
 * @brief Function prototype for a 1-dimension forasync.
 * @param[in] arg               Argument to the loop iteration
 * @param[in] index             Current iteration index
 */
typedef void (*forasync1D_Fct_t)(void *arg, int index);

/**
 * @brief Function prototype for a 2-dimensions forasync.
 * @param[in] arg               Argument to the loop iteration
 * @param[in] index_outer       Current outer iteration index
 * @param[in] index_inner       Current inner iteration index
 */
typedef void (*forasync2D_Fct_t)(void *arg, int index_outer, int index_inner);

/**
 * @brief Function prototype for a 3-dimensions forasync.
 * @param[in] arg               Argument to the loop iteration
 * @param[in] index_outer       Current outer iteration index
 * @param[in] index_mid         Current intermediate iteration index
 * @param[in] index_inner       Current inner iteration index
 */
typedef void (*forasync3D_Fct_t)(void *arg, int index_outer, int index_mid,
        int index_inner);

/**
 * @brief Parallel for loop 'forasync' (up to 3 dimensions).
 *
 * Execute iterations of a loop in parallel. The loop domain allows 
 * to specify bounds as well as tiling information. Tiling of size one,
 * is equivalent to spawning each individual iteration as an async.
 *
 * @param[in] forasync_fct      The function pointer to execute.
 * @param[in] argv              Argument to the function
 * @param[in] future_list       dependences 
 * @param[in] dim               Dimension of the loop
 * @param[in] domain            Loop domains to iterate over (array of size 'dim').
 * @param[in] mode              Forasync mode to control chunking strategy (flat chunking or recursive).
 */
void hclib_forasync(void *forasync_fct, void *argv, int dim,
                    hclib_loop_domain_t *domain, forasync_mode_t mode);

/*
 * Semantically equivalent to hclib_forasync, but returns a promise that is
 * triggered when all tasks belonging to this forasync have finished.
 */
hclib_future_t *hclib_forasync_future(void *forasync_fct, void *argv,
                                      int dim, hclib_loop_domain_t *domain,
                                      forasync_mode_t mode);

/**
 * @brief starts a new finish scope
 */
void hclib_start_finish();

/**
 * @brief ends the current finish scope
 */
void hclib_end_finish();

/*
 * Get a promise that is triggered when all tasks inside this finish scope have
 * finished, but return immediately.
 */
hclib_future_t *hclib_end_finish_nonblocking();
void hclib_end_finish_nonblocking_helper(hclib_promise_t *event);

/*
 * This API yields the current thread to another task. This API guarantees that
 * if there is currently a task in the pop or steal path of the current thread,
 * it will be found and its execution will start before we return from this
 * yield call. This API does not guarantee that the calling task will be blocked
 * until any tasks yielded to complete, nor are there any guarantees on the
 * number of tasks run during a yield beyond the first one. We do guarantee that
 * we will return to the current task at some point in the future, and that the
 * program will not terminate before this task starts.
 *
 * The optional locale argument indicates where the continuation of the current
 * task should be placed if suspended. If set to NULL, this is up to the
 * runtime.
 */
void hclib_yield(hclib_locale_t *locale);

/*
 * Fetch information on the entrypoint of the current task and the user-provided
 * arguments that were passed to the entrypoint.
 */
void hclib_get_curr_task_info(void (**fp_out)(void *), void **args_out);

/*
 * Print runtime statistics on HClib to the provided file pointer. If HClib
 * statistics are not enabled at compilation through the --enable-stats
 * configuration flag, this call is a no-op.
 */
void hclib_print_runtime_stats(FILE *fp);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* HCLIB_H_ */
