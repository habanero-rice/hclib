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
 * hcpp-ddf.h
 *
 * NOTE: Terminology
 *   DDF = data-driven future
 *   DDT = data-driven task (a task that waits on DDF objects)
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Ported from hclib
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_DDF_H_
#define HCPP_DDF_H_

#include <stdlib.h>

/**
 * @file User Interface to HCLIB's Data-Driven Futures
 */

/**
 * @defgroup DDF Data-Driven Future
 * @brief Data-Driven Future API for data-flow like programming.
 *
 * @{
 **/

/**
 * @brief Opaque type for DDFs.
 */
struct hclib_ddf_st;

typedef enum DDF_Kind {
	DDF_KIND_UNKNOWN=0,
	DDF_KIND_SHARED,
	DDF_KIND_DISTRIBUTED_OWNER,
	DDF_KIND_DISTRIBUTED_REMOTE,
} DDF_Kind_t;

/**
 * DDT data-structure to associate DDTs and DDFs.
 * This is exposed so that the runtime knows the size of the struct.
 */
typedef struct hclib_ddt_st {
    // NULL-terminated list of DDFs the DDT is registered on
    struct hclib_ddf_st ** waitingFrontier;
    /*
     * This allows us to chain all DDTs waiting on a same DDF. Whenever a DDT
     * wants to register on a DDF, and that DDF is not ready, we chain the
     * current DDT and the DDF's headDDTWaitList and try to cas on the DDF's
     * headDDTWaitList, with the current DDT.
     */
    struct hclib_ddt_st * nextDDTWaitingOnSameDDF;
} hclib_ddt_t;

// We define a typedef in this unit for convenience
typedef struct hclib_ddf_st {
	int kind;
    volatile void * datum;
    volatile hclib_ddt_t * headDDTWaitList;
} hclib_ddf_t;

/**
 * @brief Allocate and initialize a DDF.
 * @return A DDF.
 */
hclib_ddf_t *hclib_ddf_create();

/**
 * Initialize a pre-Allocated DDF.
 */
void hclib_ddf_init(hclib_ddf_t* ddf);

/**
 * @brief Allocate and initialize an array of DDFs.
 * @param[in] nb_ddfs 				Size of the DDF array
 * @param[in] null_terminated 		If true, create nb_ddfs-1 and set the last element to NULL.
 * @return A contiguous array of DDFs
 */
hclib_ddf_t **hclib_ddf_create_n(size_t nb_ddfs, int null_terminated);

/**
 * @brief Destruct a DDF.
 * @param[in] nb_ddfs 				Size of the DDF array
 * @param[in] null_terminated 		If true, create nb_ddfs-1 and set the last element to NULL.
 * @param[in] ddf 				The DDF to destruct
 */
void hclib_ddf_free_n(hclib_ddf_t ** ddf,  size_t nb_ddfs, int null_terminated);

/**
 * @brief Destruct a DDF.
 * @param[in] ddf 				The DDF to destruct
 */
void hclib_ddf_free(hclib_ddf_t * ddf);

/**
 * @brief Get the value of a DDF.
 * @param[in] ddf 				The DDF to get a value from
 */
void * hclib_ddf_get(hclib_ddf_t * ddf);

/**
 * @brief Put a value in a DDF.
 * @param[in] ddf 				The DDF to get a value from
 * @param[in] datum 			The datum to be put in the DDF
 */
void hclib_ddf_put(hclib_ddf_t * ddf, void * datum);

/*
 * Block the currently executing task on the provided DDF. Returns the datum
 * that was put on ddf.
 */
void *hclib_ddf_wait(hclib_ddf_t *ddf);

/*
 * Some extras
 */
void hclib_ddt_init(hclib_ddt_t * ddt, hclib_ddf_t ** ddf_list);

#endif /* HCPP_DDF_H_ */
