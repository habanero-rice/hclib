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
#include "hcpp-task.h"
#include "hcpp-ddf.h"

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

//TODO make this conditional in phased.h
struct _phased_t;

/**
 * @brief Function prototype executable by an async.
 * @param[in] arg           Arguments to the function
 */
typedef void (*asyncFct_t)(void * arg);
typedef void *(*futureFct_t)(void *arg);

void hclib_launch(int * argc, char ** argv, asyncFct_t fct_ptr, void * arg);

/*
 * Async definition and API
 */

// forward declaration for phased clause defined in phased.h
struct _phased_t;
// forward declaration for ddf_st in hcpp-ddf.h
struct hclib_ddf_st;

/**
 * @brief Spawn a new task asynchronously.
 * @param[in] fct_ptr           The function to execute
 * @param[in] arg               Argument to the async
 * @param[in] ddf_list          The list of DDFs the async depends on
 * @param[in] phased_clause     Phased clause to specify which phasers the async registers on
 * @param[in] property          Flag to pass information to the runtime
 */
void hclib_async(asyncFct_t fct_ptr, void * arg,
        struct hclib_ddf_st ** ddf_list, struct _phased_t * phased_clause,
        int property);

/*
 * Spawn an async that automatically puts a DDF on termination. It is the user's
 * responsibility to call hclib_ddf_free on the returned ddf_t.
 */
hclib_ddf_t *hclib_async_future(futureFct_t fp, void *arg,
        hclib_ddf_t **ddf_list, struct _phased_t *phased_clause,
        int property);

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
#define ESCAPING_ASYNC      ((int) 0x2)

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
 * @param[in] ddf_list          DDFs dependences 
 * @param[in] phased_clause     Phasers registration
 * @param[in] dim               Dimension of the loop
 * @param[in] domain            Loop domains to iterate over (array of size 'dim').
 * @param[in] mode              Forasync mode to control chunking strategy (flat chunking or recursive).
 */
void hclib_forasync(void *forasync_fct, void *argv, hclib_ddf_t **ddf_list,
        struct _phased_t *phased_clause, void *accumed_placeholder, int dim,
        loop_domain_t *domain, forasync_mode_t mode);

/**
 * @brief starts a new finish scope
 */
void hclib_start_finish();

/**
 * @brief ends the current finish scope
 */
void hclib_end_finish();

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* HCLIB_H_ */
