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

#ifndef HCLIB_H_
#define HCLIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "hclib_common.h"
#include "hclib-task.h"
#include "hclib-promise.h"

/**
 * @file Interface to HCLIB
 */

/**
 * @defgroup HClib Finish/Async/Forasync
 * @brief Core API, Finish/Async/Forasync for structured parallelism.
 *
 * @{
 **/

//TODO make this conditional in phased.h
struct _phased_t;

/**
 * @brief Function prototype executable by an async.
 * @param[in] arg           Arguments to the function
 */
typedef void (*asyncFct_t)(void * arg);
typedef void *(*futureFct_t)(void *arg);

void hclib_launch(asyncFct_t fct_ptr, void * arg);

/*
 * Async definition and API
 */

// forward declaration for phased clause defined in phased.h
struct _phased_t;
// forward declaration for promise_st in hclib-promise.h
struct hclib_promise_st;

/**
 * @brief Spawn a new task asynchronously.
 * @param[in] fct_ptr           The function to execute
 * @param[in] arg               Argument to the async
 * @param[in] future_list       The list of promises the async depends on
 * @param[in] phased_clause     Phased clause to specify which phasers the async registers on
 * @param[in] property          Flag to pass information to the runtime
 */
void hclib_async(asyncFct_t fct_ptr, void * arg,
        hclib_future_t **future_list, struct _phased_t * phased_clause,
        place_t *place, int property);

/*
 * Spawn an async that automatically puts a promise on termination.
 */
hclib_future_t *hclib_async_future(futureFct_t fp, void *arg,
        hclib_future_t **future_list, struct _phased_t *phased_clause,
        place_t *place, int property);

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
 * @param[in] phased_clause     Phasers registration
 * @param[in] dim               Dimension of the loop
 * @param[in] domain            Loop domains to iterate over (array of size 'dim').
 * @param[in] mode              Forasync mode to control chunking strategy (flat chunking or recursive).
 */
void hclib_forasync(void *forasync_fct, void *argv,
        hclib_future_t **future_list, int dim, const loop_domain_t *domain,
        forasync_mode_t mode);

/*
 * Semantically equivalent to hclib_forasync, but returns a promise that is
 * triggered when all tasks belonging to this forasync have finished.
 */
hclib_future_t *hclib_forasync_future(void *forasync_fct, void *argv,
        hclib_future_t **future_list, int dim, const loop_domain_t *domain,
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


#define _HCLIB_FINISH_IMPL(END_FUNC) \
for (int _hclib_reserved_symbol=(hclib_start_finish(), 1); \
                _hclib_reserved_symbol; \
                END_FUNC, --_hclib_reserved_symbol)
#define HCLIB_FINISH _HCLIB_FINISH_IMPL(hclib_end_finish())
#define HCLIB_FINISH_NONBLOCKING(OUT_VAR) _HCLIB_FINISH_IMPL(hclib_end_finish_nonblocking(OUT_VAR))

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* HCLIB_H_ */
