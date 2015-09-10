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
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Ported from hclib
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_DDF_H_
#define HCPP_DDF_H_

namespace hcpp {
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
struct ddf_st;

//================= DDF Support ==================    //
// Copied from https://github.com/habanero-rice/hclib //
//================================================    //

/**
 * DDT data-structure to associate DDTs and DDFs.
 * This is exposed so that the runtime know the size of the struct.
 */
typedef struct ddt_st {
    // NULL-terminated list of DDFs the DDT is registered on
    struct ddf_st ** waitingFrontier;
    // This allows us to chain all DDTs waiting on a same DDF
    // Whenever a DDT wants to register on a DDF, and that DDF is
    // not ready, we chain the current DDT and the DDF's headDDTWaitList
    // and try to cas on the DDF's headDDTWaitList, with the current DDT.
    struct ddt_st * nextDDTWaitingOnSameDDF;
} ddt_t;

// struct ddf_st is the opaque we expose.
// We define a typedef in this unit for convenience
typedef struct ddf_st {
    void * datum;
    struct ddt_st * headDDTWaitList;
} ddf_t;

#define DDF_t	ddf_t

/**
 * @brief Allocate and initialize a DDF.
 * @return A DDF.
 */
ddf_t * ddf_create();

/**
 * @brief Allocate and initialize an array of DDFs.
 * @param[in] nb_ddfs 				Size of the DDF array
 * @param[in] null_terminated 		If true, create nb_ddfs-1 and set the last element to NULL.
 * @return A contiguous array of DDFs
 */
ddf_t ** ddf_create_n(size_t nb_ddfs, int null_terminated);

/**
 * @brief Destruct a DDF.
 * @param[in] nb_ddfs 				Size of the DDF array
 * @param[in] null_terminated 		If true, create nb_ddfs-1 and set the last element to NULL.
 * @param[in] ddf 				The DDF to destruct
 */
void ddf_free_n(ddf_t ** ddf,  size_t nb_ddfs, int null_terminated);

/**
 * @brief Destruct a DDF.
 * @param[in] ddf 				The DDF to destruct
 */
void ddf_free(ddf_t * ddf);

/**
 * @brief Get the value of a DDF.
 * @param[in] ddf 				The DDF to get a value from
 */
void * ddf_get(ddf_t * ddf);

/**
 * @brief Put a value in a DDF.
 * @param[in] ddf 				The DDF to get a value from
 * @param[in] datum 			The datum to be put in the DDF
 */
void ddf_put(ddf_t * ddf, void * datum);

/*
 * Some extras
 */
void ddt_init(ddt_t * ddt, ddf_t ** ddf_list);

}

#endif /* HCPP_DDF_H_ */
