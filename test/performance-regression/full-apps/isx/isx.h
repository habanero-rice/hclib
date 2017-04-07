/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
    * Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef _ISX_H
#define _ISX_H

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "timer.h"
#include "pcg_basic.h"
/*
 * Ensures the command line parameters and values specified in params.h
 * are valid and will not cause problems.
 */
static char * parse_params(const int argc, char ** argv);

/*
 * Sorts a random array by distrectizing the domain of values into buckets.
 * Each bucket is assigned to a PE and all keys belonging to a bucket are sent
 * to the corresponding PE. Each PE then performs a local sort of the keys in its bucket.
 */
static int bucket_sort();

#ifdef PERMUTE
/*
 * Creates a randomly ordered array of PEs used in the exchange_keys function
 */
static void create_permutation_array();

/*
 * Randomly shuffles a generic array
 */
static void shuffle(void * array, size_t n, size_t size);
#endif


/*
 * Generates random keys [0, MAX_KEY_VAL] on each rank using the time and rank as a seed
 */
static KEY_TYPE * make_input(void);

/*
 * Computes the size of each local bucket by iterating all local keys and incrementing
 * their corresponding bucket's size
 */
#ifdef HCLIB_VERSION
static int * count_local_bucket_sizes(KEY_TYPE const * const my_keys,
        int ***bucket_counts_per_chunk_out);
#else
static int * count_local_bucket_sizes(KEY_TYPE const * const my_keys);
#endif

/*
 * Computes the prefix scan of the local bucket sizes to determine the starting locations
 * of each bucket in the local bucketed array.
 * Stores a copy of the bucket offsets in send_offsets for use in exchanging keys because the
 * original bucket_offsets array is modified in the bucketize function
 */
static int * compute_local_bucket_offsets(int const * const local_bucket_sizes,
                                          int ** send_offsets);

/*
 * Rearranges all local keys into their corresponding local bucket.
 * The contents of each bucket are not sorted.
 */
#ifdef HCLIB_VERSION
static KEY_TYPE * bucketize_local_keys(KEY_TYPE const * my_keys,
                                       int * const local_bucket_offsets,
                                       int **bucket_counts_per_chunk);
#else
static KEY_TYPE * bucketize_local_keys(KEY_TYPE const * my_keys,
                                       int * const local_bucket_offsets);
#endif
/*
 * Each PE sends the contents of its local buckets to the PE that owns that bucket.
 */
static KEY_TYPE * exchange_keys(int const * const send_offsets,
                                int const * const local_bucket_sizes,
                                KEY_TYPE const * const local_bucketed_keys);

/*
 * Count the occurence of each key within my bucket. 
 */
static int * count_local_keys(KEY_TYPE const * const my_bucket_keys);

/*
 * Verifies the correctness of the sort. 
 * Ensures all keys after the exchange are within a PE's bucket boundaries.
 * Ensures the final number of keys is equal to the initial.
 */
static int verify_results(int const * const my_local_key_counts, 
                           KEY_TYPE const * const my_local_keys);

/*
 * Seeds each rank based on the rank number and time
 */
#ifdef HCLIB_VERSION
static pcg32_random_t seed_my_rank(const int chunk);
#else
static pcg32_random_t seed_my_rank(void);
#endif

/*
 * Provides a sequential ordering of PE operations
 */
#ifdef DEBUG
static void wait_my_turn();
static void my_turn_complete();
#endif

/*
 * Initializes the sync array needed by shmem collective operations.
 */
static void init_shmem_sync_array(long * const pSync);


static void init_array(int * array, const int size)
{
  for(int i = 0; i < size; ++i){
    array[i] = 0;
  }
}

static int file_exists(char * filename);

static void log_times(char * log_directory);
static void report_summary_stats(void);
static void print_timer_names(FILE * fp);
static void print_run_info(FILE * fp);
static void print_timer_values(FILE * fp);
static double * gather_rank_times(_timer_t * const timer);
static unsigned int * gather_rank_counts(_timer_t * const timer);

#endif
