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
#include <shmem.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <unistd.h> // sleep()
#include <sys/stat.h>
#include <stdint.h>
#include "params.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "timer.h"
#include "pcg_basic.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#define ROOT_PE 0

static KEY_TYPE * make_input(void);
static int bucket_sort(void);
static char * parse_params(const int argc, char ** argv);
static inline pcg32_random_t seed_my_chunk(int chunk);
static inline int * count_local_bucket_sizes(KEY_TYPE const * restrict const my_keys);
static inline int * compute_local_bucket_offsets(int const * restrict const local_bucket_sizes,
                                                 int ** restrict send_offsets);
static inline KEY_TYPE * bucketize_local_keys(KEY_TYPE const * restrict const my_keys,
                                              int * restrict const local_bucket_offsets);
static inline KEY_TYPE * exchange_keys(int const * restrict const send_offsets,
                                       int const * restrict const local_bucket_sizes,
                                       KEY_TYPE const * restrict const my_local_bucketed_keys);
static inline int* count_local_keys(KEY_TYPE const * restrict const my_bucket_keys);
static int verify_results(KEY_TYPE const * restrict const my_local_keys);
static void log_times(char * log_file);
static void report_summary_stats(void);
static void print_timer_names(FILE * fp);
static void print_run_info(FILE * fp);
static void print_timer_values(FILE * fp);
static double * gather_rank_times(_timer_t * const timer);
static unsigned int * gather_rank_counts(_timer_t * const timer);
static void init_shmem_sync_array(long * restrict const pSync);
static int file_exists(char * filename);


// Needed for shmem collective operations
int pWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
double dWrk[_SHMEM_REDUCE_SYNC_SIZE];
long long int llWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync[_SHMEM_REDUCE_SYNC_SIZE];

uint64_t NUM_PES; // Number of parallel workers
uint64_t TOTAL_KEYS; // Total number of keys across all PEs
uint64_t NUM_KEYS_PER_PE; // Number of keys generated on each PE
uint64_t NUM_BUCKETS; // The number of buckets in the bucket sort
uint64_t BUCKET_WIDTH; // The size of each bucket
uint64_t MAX_KEY_VAL; // The maximum possible generated key value

volatile int whose_turn;

long long int receive_offset = 0;
long long int my_bucket_size = 0;

#define PARALLEL_FOR_MODE SHMEM_PARALLEL_FOR_RECURSIVE_MODE
#define CHUNKS_COUNT_LOCAL_KEYS (actual_num_workers)
#define CHUNKS_MAKE_INPUT CHUNKS_PER_PE
int actual_num_workers;
int** local_bucket_sizes_chunk;
int ** my_local_key_counts;
KEY_TYPE*** my_local_bucketed_keys_chunk;
int** local_bucket_offsets_chunk;
/*
 * This variable sets the maximum number of chunks allowed
 * to participate in computation per pe.
 */
int CHUNKS_PER_PE=1;

#define GET_VIRTUAL_RANK(rank, chunk) ((rank * actual_num_workers) + (chunk))

#define SHMEM_BARRIER_AT_START    { timer_start(&timers[TIMER_BARRIER_START]); shmem_barrier_all(); timer_stop(&timers[TIMER_BARRIER_START]); }
#define SHMEM_BARRIER_AT_EXCHANGE { timer_start(&timers[TIMER_BARRIER_EXCHANGE]); shmem_barrier_all(); timer_stop(&timers[TIMER_BARRIER_EXCHANGE]); }
#define SHMEM_BARRIER_AT_END      { timer_start(&timers[TIMER_BARRIER_END]); shmem_barrier_all(); timer_stop(&timers[TIMER_BARRIER_END]); }

// This is done due to current limitation that entrypoint function
// cannot accept arguments. This will be resolved in future version of
// AsyncSHMEM
int m_argc;
char** m_argv;

// #define EXTRA_STATS

#ifdef EXTRA_STATS
float avg_time=0, avg_time_all2all = 0;
#endif

// #define KEY_BUFFER_SIZE (1uLL<<28uLL)
#define KEY_BUFFER_SIZE ((1uLL<<28uLL) + 70000)

// The receive array for the All2All exchange
// KEY_TYPE* my_bucket_keys;
KEY_TYPE my_bucket_keys[KEY_BUFFER_SIZE];

#ifdef PERMUTE
int * permute_array;
#endif

void entrypoint(void *arg) {

  char * log_file = parse_params(m_argc, m_argv);

  init_shmem_sync_array(pSync);

  bucket_sort();

  log_times(log_file);

  //return err;
}

int main (int argc, char ** argv) {
  shmem_init ();
  m_argc = argc;
  m_argv = argv;

#ifdef EXTRA_STATS
  _timer_t stage_time;
  if(shmem_my_pe() == 0) {
    printf("\n-----\nmkdir timedrun fake\n\n");
    timer_start(&stage_time);
  }
#endif
  entrypoint(NULL);

#ifdef EXTRA_STATS
  if(shmem_my_pe() == 0) {
    just_timer_stop(&stage_time);
    double tTime = ( stage_time.stop.tv_sec - stage_time.start.tv_sec ) + ( stage_time.stop.tv_nsec - stage_time.start.tv_nsec )/1E9;
    avg_time *= 1000;
    avg_time_all2all *= 1000;
    printf("\n============================ MMTk Statistics Totals ============================\n");
    if(NUM_ITERATIONS == 1) { //TODO: fix time calculation below for more number of iterations
      printf("time.mu\tt.ATA_KEYS\tt.MAKE_INPUT\tt.COUNT_BUCKET_SIZES\tt.BUCKETIZE\tt.COMPUTE_OFFSETS\tt.LOCAL_SORT\tBARRIER_AT_START\tBARRIER_AT_EXCHANGE\tBARRIER_AT_END\tnWorkers\tnPEs\n");
      double TIMES[TIMER_NTIMERS];
      memset(TIMES, 0x00, sizeof(double) * TIMER_NTIMERS);
      for(uint64_t i=0; i<NUM_PES; i++) {
        for(int t = 0; t < TIMER_NTIMERS; ++t){
          if(timers[t].all_times != NULL){
            TIMES[t] += timers[t].all_times[i];
          }
        }
      }
      for(int t = 0; t < TIMER_NTIMERS; ++t){
        printf("%.3f\t", (TIMES[t]/NUM_PES)*1000);
      }
      printf("%d\t%d\n",actual_num_workers,NUM_PES);
      printf("Total time: %.3f\n",(TIMES[0]/NUM_PES)*1000);
    }
    else {
      printf("time.mu\ttimeAll2All\tnWorkers\tnPEs\n");
      printf("%.3f\t%.3f\t%d\t%d\n",avg_time,avg_time_all2all,actual_num_workers,NUM_PES);
      printf("Total time: %.3f\n",avg_time);
    }

    printf("------------------------------ End MMTk Statistics -----------------------------\n");
    printf("===== TEST PASSED in %.3f msec =====\n",(tTime*1000));
  }
#endif

  shmem_finalize ();
  return 0;
}


// Parses all of the command line input and definitions in params.h
// to set all necessary runtime values and options
static char * parse_params(const int argc, char ** argv)
{
  if(argc != 3)
  {
    if( shmem_my_pe() == 0){
      printf("Usage:  \n");
      printf("  ./%s <total num keys(strong) | keys per pe(weak)> <log_file>\n",argv[0]);
    }

    shmem_finalize();
    exit(1);
  }

  const char* chunks_env = getenv("ISX_PE_CHUNKS");
  CHUNKS_PER_PE = chunks_env ? atoi(chunks_env) : 1;
#pragma omp parallel
#pragma omp single
  actual_num_workers = omp_get_num_threads();

  NUM_PES = (uint64_t) shmem_n_pes();
  MAX_KEY_VAL = DEFAULT_MAX_KEY;
  NUM_BUCKETS = NUM_PES;
  BUCKET_WIDTH = (uint64_t) ceil((double)MAX_KEY_VAL/NUM_BUCKETS);
  char * log_file = argv[2];
  char scaling_msg[64];

  switch(SCALING_OPTION){
    case STRONG:
      {
        TOTAL_KEYS = (uint64_t) atoi(argv[1]);
        NUM_KEYS_PER_PE = (uint64_t) ceil((double)TOTAL_KEYS/NUM_PES);
        sprintf(scaling_msg,"STRONG");
        break;
      }

    case WEAK:
      {
        NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1])) * actual_num_workers;
        sprintf(scaling_msg,"WEAK");
        break;
      }

    case WEAK_ISOBUCKET:
      {
        NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1])) * actual_num_workers;
        BUCKET_WIDTH = ISO_BUCKET_WIDTH; 
        MAX_KEY_VAL = (uint64_t) (NUM_PES * actual_num_workers * BUCKET_WIDTH);
        sprintf(scaling_msg,"WEAK_ISOBUCKET");
        break;
      }

    default:
      {
        if(shmem_my_pe() == 0){
          printf("Invalid scaling option! See params.h to define the scaling option.\n");
        }

        shmem_finalize();
        exit(1);
        break;
      }
  }

  assert(NUM_KEYS_PER_PE % actual_num_workers == 0);
  assert(MAX_KEY_VAL > 0);
  assert(NUM_KEYS_PER_PE > 0);
  assert(NUM_PES > 0);
  assert(MAX_KEY_VAL > NUM_PES);
  assert(NUM_BUCKETS > 0);
  assert(BUCKET_WIDTH > 0);
  
  if(shmem_my_pe() == 0){
    printf("ISx v%1d.%1d\n",MAJOR_VERSION_NUMBER,MINOR_VERSION_NUMBER);
#ifdef PERMUTE
    printf("Random Permute Used in ATA.\n");
#endif
    printf("  Number of Keys per PE: %" PRIu64 "\n", NUM_KEYS_PER_PE);
    printf("  Number of Chunks per PE (ISX_PE_CHUNKS): %d\n",CHUNKS_PER_PE);
    printf("  OpenMP Version, total workers: %d\n",actual_num_workers);
    printf("  Max Key Value: %" PRIu64 "\n", MAX_KEY_VAL);
    printf("  Bucket Width: %" PRIu64 "\n", BUCKET_WIDTH);
    printf("  Number of Iterations: %u\n", NUM_ITERATIONS);
    printf("  Number of PEs: %" PRIu64 "\n", NUM_PES);
    printf("  %s Scaling!\n",scaling_msg);
    }

  return log_file;
}


/*
 * The primary compute function for the bucket sort
 * Executes the sum of NUM_ITERATIONS + BURN_IN iterations, as defined in params.h
 * Only iterations after the BURN_IN iterations are timed
 * Only the final iteration calls the verification function
 */
static int bucket_sort(void)
{
  int err = 0;

  init_timers(NUM_ITERATIONS);

#ifdef PERMUTE
  create_permutation_array();
#endif

  // my_bucket_keys = (KEY_TYPE*) shmem_malloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE));
  // assert(my_bucket_keys);
  my_local_key_counts = malloc(CHUNKS_COUNT_LOCAL_KEYS * sizeof(int*));
  assert(my_local_key_counts);
  for(int i=0; i<CHUNKS_COUNT_LOCAL_KEYS; i++) {
      my_local_key_counts[i] = malloc(BUCKET_WIDTH * sizeof(int));
      assert(my_local_key_counts[i]);
  }

  for(uint64_t i = 0; i < (NUM_ITERATIONS + BURN_IN); ++i)
  {
    for(int i=0; i<CHUNKS_COUNT_LOCAL_KEYS; i++) memset(my_local_key_counts[i], 0x00, BUCKET_WIDTH * sizeof(int));
    local_bucket_sizes_chunk = malloc(CHUNKS_PER_PE* sizeof(int*));
    assert(local_bucket_sizes_chunk);

    // Reset timers after burn in 
    if(i == BURN_IN){ init_timers(NUM_ITERATIONS); } 

    SHMEM_BARRIER_AT_START;

    timer_start(&timers[TIMER_TOTAL]);

    KEY_TYPE * my_keys = make_input();

    int * local_bucket_sizes = count_local_bucket_sizes(my_keys);

    int * send_offsets;
    int * local_bucket_offsets = compute_local_bucket_offsets(local_bucket_sizes,
                                                                   &send_offsets);

    KEY_TYPE * my_local_bucketed_keys =  bucketize_local_keys(my_keys, local_bucket_offsets);

    KEY_TYPE * my_bucket_keys = exchange_keys(send_offsets, 
                                              local_bucket_sizes,
                                              my_local_bucketed_keys);

    my_bucket_size = receive_offset;

    count_local_keys(my_bucket_keys);

    SHMEM_BARRIER_AT_END;

    timer_stop(&timers[TIMER_TOTAL]);

    for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) { 
      free(local_bucket_sizes_chunk[chunk]);
    }  
    free(local_bucket_sizes_chunk);

    // Only the last iteration is verified
    if(i == NUM_ITERATIONS) { 
      err = verify_results(my_bucket_keys);
    }

    // Reset receive_offset used in exchange_keys
    receive_offset = 0;

    free(my_local_bucketed_keys);
    free(my_keys);
    free(local_bucket_sizes);
    free(local_bucket_offsets);
    free(send_offsets);

    shmem_barrier_all();
  }

  for(int i=0; i<CHUNKS_COUNT_LOCAL_KEYS; i++) free(my_local_key_counts[i]);
  free(my_local_key_counts);

  return err;
}

/*
 * Generates uniformly random keys [0, MAX_KEY_VAL] on each rank using the time and rank
 * number as a seed
 */
static KEY_TYPE * make_input(void)
{
  timer_start(&timers[TIMER_INPUT]);

  KEY_TYPE * restrict const my_keys = malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
  assert(my_keys);

  int chunk;
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
  for(chunk=0; chunk<CHUNKS_MAKE_INPUT; chunk++) {
    const uint64_t keys_per_chunk = NUM_KEYS_PER_PE / CHUNKS_MAKE_INPUT;
    const uint64_t start_index = chunk * keys_per_chunk;
    const uint64_t max_index = start_index + keys_per_chunk;
    pcg32_random_t rng = seed_my_chunk(chunk);

    KEY_TYPE * restrict my_keys_1D = &(my_keys[start_index]);   
    for(uint64_t i=start_index; i<max_index; i++) {
       *my_keys_1D = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
       my_keys_1D += 1;
    }
  }

  timer_stop(&timers[TIMER_INPUT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: Initial Keys: ", my_rank);
  for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif
  return my_keys;
}

static inline void init_array(int * array, const int size)
{
  for(int i = 0; i < size; ++i){
    array[i] = 0;
  }
}


/*
 * Computes the size of each bucket by iterating all keys and incrementing
 * their corresponding bucket's size
 */
static inline int * count_local_bucket_sizes(KEY_TYPE const * restrict const my_keys)
{
  int * restrict const local_bucket_sizes = malloc(NUM_BUCKETS * sizeof(int));
  assert(local_bucket_sizes);
  memset(local_bucket_sizes, 0x00, NUM_BUCKETS * sizeof(int));

  timer_start(&timers[TIMER_BCOUNT]);

  init_array(local_bucket_sizes, NUM_BUCKETS);

  if(NUM_BUCKETS == 1) {
    local_bucket_sizes[0] = NUM_KEYS_PER_PE;
  }
  else {
    int chunk;
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
    for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
      local_bucket_sizes_chunk[chunk] = malloc(NUM_BUCKETS * sizeof(int));
      assert(local_bucket_sizes_chunk[chunk]);
      memset(local_bucket_sizes_chunk[chunk], 0x00, NUM_BUCKETS * sizeof(int));
      int * restrict const local_bucket_sizes = local_bucket_sizes_chunk[chunk];
      const uint32_t keys_per_chunk = NUM_KEYS_PER_PE / CHUNKS_PER_PE;
      const uint32_t start_index = chunk * keys_per_chunk;
      KEY_TYPE const * restrict const my_keys_1D = &(my_keys[start_index]);
      for(uint64_t i = 0; i < keys_per_chunk; ++i){
        const uint32_t bucket_index = my_keys_1D[i]/BUCKET_WIDTH;
        local_bucket_sizes[bucket_index]++;
      }
    }

    for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) { 
      for(int i=0; i<NUM_BUCKETS; i++) {
        local_bucket_sizes[i] += local_bucket_sizes_chunk[chunk][i];
      }
    }  
  }

  timer_stop(&timers[TIMER_BCOUNT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: local bucket sizes: ", my_rank);
  for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", local_bucket_sizes[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif

  return local_bucket_sizes;
}


/*
 * Computes the prefix scan of the bucket sizes to determine the starting locations
 * of each bucket in the local bucketed array
 * Stores a copy of the bucket offsets for use in exchanging keys because the
 * original bucket_offsets array is modified in the bucketize function
 */
static inline int * compute_local_bucket_offsets(int const * restrict const local_bucket_sizes,
                                                 int ** restrict send_offsets)
{
  int * restrict const local_bucket_offsets = malloc(NUM_BUCKETS * sizeof(int));
  assert(local_bucket_offsets);

  timer_start(&timers[TIMER_BOFFSET]);

  (*send_offsets) = malloc(NUM_BUCKETS * sizeof(int));
  assert(*send_offsets);

  // NOTE: This is a very small computation and hence we are not parallelizing this
  local_bucket_offsets[0] = 0;
  (*send_offsets)[0] = 0;
  int temp = 0;
  for(uint64_t i = 1; i < NUM_BUCKETS; i++){
    temp = local_bucket_offsets[i-1] + local_bucket_sizes[i-1];
    local_bucket_offsets[i] = temp; 
    (*send_offsets)[i] = temp;
  }
  timer_stop(&timers[TIMER_BOFFSET]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: local bucket offsets: ", my_rank);
  for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", local_bucket_offsets[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif
  return local_bucket_offsets;
}

/*
 * Places local keys into their corresponding local bucket.
 * The contents of each bucket are not sorted.
 */
static inline KEY_TYPE * bucketize_local_keys(KEY_TYPE const * restrict const my_keys,
                                              int * restrict const local_bucket_offsets)
{
  KEY_TYPE * restrict const my_local_bucketed_keys = malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
  assert(my_local_bucketed_keys);

  timer_start(&timers[TIMER_BUCKETIZE]);

  my_local_bucketed_keys_chunk = malloc(CHUNKS_PER_PE* sizeof(KEY_TYPE**));
  assert(my_local_bucketed_keys_chunk);
  local_bucket_offsets_chunk = (int**) malloc(CHUNKS_PER_PE* sizeof(int*));
  assert(local_bucket_offsets_chunk);

  int chunk;
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    my_local_bucketed_keys_chunk[chunk] = malloc(NUM_BUCKETS * sizeof(KEY_TYPE*));
    assert(my_local_bucketed_keys_chunk[chunk]);
    local_bucket_offsets_chunk[chunk] = malloc(NUM_BUCKETS * sizeof(int));
    assert(local_bucket_offsets_chunk[chunk]);
    memset(local_bucket_offsets_chunk[chunk], 0x00, NUM_BUCKETS * sizeof(int));
    for(int bucket=0; bucket<NUM_BUCKETS; bucket++) {
      my_local_bucketed_keys_chunk[chunk][bucket] = malloc(sizeof(KEY_TYPE) * local_bucket_sizes_chunk[chunk][bucket]);
      assert(my_local_bucketed_keys_chunk[chunk][bucket]);
    }
    const uint32_t keys_per_chunk = NUM_KEYS_PER_PE / CHUNKS_PER_PE;
    const uint32_t start_index = chunk * keys_per_chunk;

    KEY_TYPE const * restrict const my_keys_1D = &(my_keys[start_index]);
    int * restrict local_bucket_offsets_chunk_1D = local_bucket_offsets_chunk[chunk];
    int const * restrict const local_bucket_sizes_chunk_1D = local_bucket_sizes_chunk[chunk];
    KEY_TYPE** restrict my_local_bucketed_keys_chunk_2D = my_local_bucketed_keys_chunk[chunk];

    for(uint64_t i = 0; i < keys_per_chunk; ++i){
      const KEY_TYPE key = my_keys_1D[i];
      const uint32_t bucket_index = key / BUCKET_WIDTH;
      uint32_t index = local_bucket_offsets_chunk_1D[bucket_index]++;
      assert(index < local_bucket_sizes_chunk_1D[bucket_index]);
      my_local_bucketed_keys_chunk_2D[bucket_index][index] = key;
    }
  }

  for(int bucket=0; bucket<NUM_BUCKETS; bucket++) {
    uint32_t index = local_bucket_offsets[bucket];
    for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
      memcpy(&(my_local_bucketed_keys[index]), my_local_bucketed_keys_chunk[chunk][bucket], sizeof(KEY_TYPE) * local_bucket_sizes_chunk[chunk][bucket]);
      index += local_bucket_sizes_chunk[chunk][bucket];
    }
    local_bucket_offsets[bucket] = index;
  }

  // free the memory
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    for(int bucket=0; bucket<NUM_BUCKETS; bucket++) {
      free(my_local_bucketed_keys_chunk[chunk][bucket]);
    }
    free(local_bucket_offsets_chunk[chunk]);
    free(my_local_bucketed_keys_chunk[chunk]);
  }
  free(my_local_bucketed_keys_chunk);
  free(local_bucket_offsets_chunk);

  timer_stop(&timers[TIMER_BUCKETIZE]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: local bucketed keys: ", my_rank);
  for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_local_bucketed_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif
  return my_local_bucketed_keys;
}

/*
 * Each PE sends the contents of its local buckets to the PE that owns that bucket.
 */
static inline KEY_TYPE * exchange_keys(int const * restrict const send_offsets,
                                       int const * restrict const local_bucket_sizes,
                                       KEY_TYPE const * restrict const my_local_bucketed_keys)
{
  timer_start(&timers[TIMER_ATA_KEYS]);

  const int my_rank = shmem_my_pe();
  unsigned int total_keys_sent = 0;

  for(uint64_t i = 0; i < NUM_PES; ++i){

#ifdef PERMUTE
    const int target_pe = permute_array[i];
#elif INCAST
    const int target_pe = i;
#else
    const int target_pe = (my_rank + i) % NUM_PES;
#endif

    // Local keys already written with memcpy
    if(target_pe == my_rank){ continue; }

    const int read_offset_from_self = send_offsets[target_pe];
    const int my_send_size = local_bucket_sizes[target_pe];
    const long long int write_offset_into_target = shmem_longlong_fadd(&receive_offset, (long long int)my_send_size, target_pe); 
    if (write_offset_into_target + my_send_size > KEY_BUFFER_SIZE) {
        fprintf(stderr, "%llu %llu\n", write_offset_into_target + my_send_size, KEY_BUFFER_SIZE);
        exit(1);
    }
    assert(write_offset_into_target + my_send_size <= KEY_BUFFER_SIZE);
    assert(read_offset_from_self + my_send_size <= NUM_KEYS_PER_PE);
    shmem_int_put(&(my_bucket_keys[write_offset_into_target]),
                  &(my_local_bucketed_keys[read_offset_from_self]),
                  my_send_size, target_pe);

#ifdef DEBUG
    printf("Rank: %d Target: %d Offset into target: %lld Offset into myself: %d Send Size: %d\n",
        my_rank, target_pe, write_offset_into_target, read_offset_from_self, my_send_size);
#endif

    total_keys_sent += my_send_size;
  }

  // Keys destined for local key buffer can be written with memcpy
  const long long int write_offset_into_self = shmem_longlong_fadd(&receive_offset, (long long int)local_bucket_sizes[my_rank], my_rank);
  const long long int send_offsets_start = send_offsets[my_rank];
  const long long int chunks = local_bucket_sizes[my_rank] / actual_num_workers;
  const long long int max_bucket_size = local_bucket_sizes[my_rank];
  int chunk;
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
  for(chunk=0; chunk<actual_num_workers; chunk++) {
    const long long int write_offset_into_self_worker = write_offset_into_self + (chunk * chunks);
    const long long int send_offsets_start_worker = send_offsets_start + (chunk * chunks);
    long long int send_size = chunks;
    if(chunk+1 == actual_num_workers) {
      long long int leftover = max_bucket_size - (chunks * actual_num_workers);
      send_size += leftover;
    }
    memcpy(&my_bucket_keys[write_offset_into_self_worker],&my_local_bucketed_keys[send_offsets_start_worker],
                        send_size*sizeof(KEY_TYPE));
  }

#ifdef BARRIER_ATA
  SHMEM_BARRIER_AT_EXCHANGE;
#endif

  timer_stop(&timers[TIMER_ATA_KEYS]);
  timer_count(&timers[TIMER_ATA_KEYS], total_keys_sent);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  sprintf(msg,"Rank %d: Bucket Size %lld | Total Keys Sent: %u | Keys after exchange:", 
                        my_rank, receive_offset, total_keys_sent);
  for(long long int i = 0; i < receive_offset; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_bucket_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif

  return my_bucket_keys;
}

/*
 * Counts the occurence of each key in my bucket. 
 * Key indices into the count array are the key's value minus my bucket's 
 * minimum key value to allow indexing from 0.
 * my_bucket_keys: All keys in my bucket unsorted [my_rank * BUCKET_WIDTH, (my_rank+1)*BUCKET_WIDTH)
 */
static inline int* count_local_keys(KEY_TYPE const * restrict const my_bucket_keys)
{
  timer_start(&timers[TIMER_SORT]);

  const int my_rank = shmem_my_pe();
  const int my_min_key = my_rank * BUCKET_WIDTH;
  const int max_chunks = (int) my_bucket_size / actual_num_workers;

  int chunk;
#pragma omp parallel for private(chunk) schedule (static,1) 
  for(chunk=0; chunk<CHUNKS_COUNT_LOCAL_KEYS; chunk++) {
    const int start_index = chunk * max_chunks;
    int * restrict my_local_key_counts_1D = my_local_key_counts[chunk];
    int const * restrict const my_bucket_keys_1D = &(my_bucket_keys[start_index]);
    for(int i=0; i<max_chunks; i++) {
      const unsigned int key_index = my_bucket_keys_1D[i] - my_min_key;
      assert(my_bucket_keys_1D[i] >= my_min_key);
      assert(key_index < BUCKET_WIDTH);
      my_local_key_counts_1D[key_index]++;
    }
  }

  //sequential part here
  const int leftover = my_bucket_size - (max_chunks * CHUNKS_COUNT_LOCAL_KEYS);
  if(leftover) {
    const int chunk = CHUNKS_COUNT_LOCAL_KEYS - 1;
    for(int i=(my_bucket_size-leftover); i<my_bucket_size; i++) {
      const unsigned int key_index = my_bucket_keys[i] - my_min_key;
      assert(my_bucket_keys[i] >= my_min_key);
      assert(key_index < BUCKET_WIDTH);
      my_local_key_counts[chunk][key_index]++;
    }
  }
 
  timer_stop(&timers[TIMER_SORT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[4096];
  sprintf(msg,"Rank %d: Bucket Size %lld | Local Key Counts:", my_rank, my_bucket_size);
  for(int chunk=0; chunk<actual_num_workers; chunk++) {
  for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_local_key_counts[chunk][i]);
  }
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif
  return NULL;
}

typedef struct verify_results_async_t {
  int max_chunks;
  int my_min_key;
  int my_max_key;
} verify_results_async_t;

void verify_results_async(void* args, int chunk) {
  verify_results_async_t* arg = (verify_results_async_t*) args;
  const int max_chunks = arg->max_chunks;
  const int my_min_key = arg->my_min_key;
  const int my_max_key = arg->my_max_key;

  const int start_index = chunk * max_chunks;
  const int max_index = start_index + max_chunks;
  for(int i=start_index; i<max_index; i++) {
    const int key = my_bucket_keys[i];
    if((key < my_min_key) || (key > my_max_key)){
      printf("Rank %d Failed Verification!\n",shmem_my_pe());
      printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
    }
  }
}

/*
 * Verifies the correctness of the sort. 
 * Ensures all keys are within a PE's bucket boundaries.
 * Ensures the final number of keys is equal to the initial.
 */
static int verify_results(KEY_TYPE const * restrict const my_local_keys)
{

  shmem_barrier_all();

  int error = 0;

  const int my_rank = shmem_my_pe();

  const int my_min_key = my_rank * BUCKET_WIDTH;
  const int my_max_key = (my_rank+1) * BUCKET_WIDTH - 1;

  const int max_chunks = (int) my_bucket_size / actual_num_workers;

  int chunk;
#pragma omp parallel for private(chunk) schedule (static,1) 
  // Verify all keys are within bucket boundaries
  for(chunk=0; chunk<actual_num_workers; chunk++) {
    const int start_index = chunk * max_chunks;
    const int max_index = start_index + max_chunks;
    for(int i=start_index; i<max_index; i++) {
      const int key = my_bucket_keys[i];
      if((key < my_min_key) || (key > my_max_key)){
        printf("Rank %d Failed Verification!\n",shmem_my_pe());
        printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
      }
    }
  }

  //sequential part here
  const int leftover = my_bucket_size - (max_chunks * actual_num_workers);
  if(leftover) {
    for(int i=(my_bucket_size-leftover); i<my_bucket_size; i++) {
      const int key = my_local_keys[i];
      if((key < my_min_key) || (key > my_max_key)){
        printf("Rank %d Failed Verification!\n",my_rank);
        printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
        error = 1;
      }
    }
  }

  // Verify the sum of the key population equals the expected bucket size
  long long int bucket_size_test = 0;
  for(int chunk=0; chunk<CHUNKS_COUNT_LOCAL_KEYS; chunk++) {
    for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
      bucket_size_test +=  my_local_key_counts[chunk][i];
    }
  }
  if(bucket_size_test != my_bucket_size){
      printf("Rank %d Failed Verification!\n",my_rank);
      printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, my_bucket_size);
      error = 1;
  }

  // Verify the final number of keys equals the initial number of keys
  static long long int total_num_keys = 0;
  shmem_longlong_sum_to_all(&total_num_keys, &my_bucket_size, 1, 0, 0, NUM_PES, llWrk, pSync);
  shmem_barrier_all();

  if(total_num_keys != (long long int)(NUM_KEYS_PER_PE * NUM_PES)){
    if(my_rank == ROOT_PE){
      printf("Verification Failed!\n");
      printf("Actual total number of keys: %lld Expected %" PRIu64 "\n", total_num_keys, NUM_KEYS_PER_PE * NUM_PES );
      error = 1;
    }
  }

  return error;
}

/*
 * Gathers all the timing information from each PE and prints
 * it to a file. All information from a PE is printed as a row in a tab seperated file
 */
static void log_times(char * log_file)
{
  FILE * fp = NULL;

  for(uint64_t i = 0; i < TIMER_NTIMERS; ++i){
    timers[i].all_times = gather_rank_times(&timers[i]);
    timers[i].all_counts = gather_rank_counts(&timers[i]);
  }

  if(shmem_my_pe() == ROOT_PE)
  {
    int print_names = 0;
    if(file_exists(log_file) != 1){
      print_names = 1;
    }

    if((fp = fopen(log_file, "a+b"))==NULL){
      perror("Error opening log file:");
      exit(1);
    }

    if(print_names == 1){
      print_run_info(fp);
      print_timer_names(fp);
    }
    print_timer_values(fp);

    report_summary_stats();

    fclose(fp);
  }

}

/*
 * Computes the average total time and average all2all time and prints it to the command line
 */
static void report_summary_stats(void)
{
  
  if(timers[TIMER_TOTAL].seconds_iter > 0) {
    const uint32_t num_records = NUM_PES * timers[TIMER_TOTAL].seconds_iter;
    double temp = 0.0;
    for(uint64_t i = 0; i < num_records; ++i){
      temp += timers[TIMER_TOTAL].all_times[i];
    }
#ifdef EXTRA_STATS
    avg_time = temp/num_records;
#endif
      printf("Average total time (per PE): %f seconds\n", temp/num_records);
  }

  if(timers[TIMER_ATA_KEYS].seconds_iter >0) {
    const uint32_t num_records = NUM_PES * timers[TIMER_ATA_KEYS].seconds_iter;
    double temp = 0.0;
    for(uint64_t i = 0; i < num_records; ++i){
      temp += timers[TIMER_ATA_KEYS].all_times[i];
    }
#ifdef EXTRA_STATS
    avg_time_all2all = temp/num_records;
#endif
    printf("Average all2all time (per PE): %f seconds\n", temp/num_records);
  }
}

/*
 * Prints all the labels for each timer as a row to the file specified by 'fp'
 */
static void print_timer_names(FILE * fp)
{
  for(uint64_t i = 0; i < TIMER_NTIMERS; ++i){
    if(timers[i].seconds_iter > 0){
      fprintf(fp, "%s (sec)\t", timer_names[i]);
    }
    if(timers[i].count_iter > 0){
      fprintf(fp, "%s_COUNTS\t", timer_names[i]);
    }
  }
  fprintf(fp,"\n");
}

/*
 * Prints all the relevant runtime parameters as a row to the file specified by 'fp'
 */
static void print_run_info(FILE * fp)
{
  fprintf(fp,"SHMEM\t");
  fprintf(fp,"NUM_PES %" PRIu64 "\t", NUM_PES);
  fprintf(fp,"Max_Key %" PRIu64 "\t", MAX_KEY_VAL); 
  fprintf(fp,"Num_Iters %u\t", NUM_ITERATIONS);

  switch(SCALING_OPTION){
    case STRONG: {
        fprintf(fp,"Strong Scaling: %" PRIu64 " total keys\t", NUM_KEYS_PER_PE * NUM_PES);
        break;
      }
    case WEAK: {
        fprintf(fp,"Weak Scaling: %" PRIu64 " keys per PE\t", NUM_KEYS_PER_PE);
        break;
      }
    case WEAK_ISOBUCKET: {
        fprintf(fp,"Weak Scaling Constant Bucket Width: %" PRIu64 "u keys per PE \t", NUM_KEYS_PER_PE);
        fprintf(fp,"Constant Bucket Width: %" PRIu64 "\t", BUCKET_WIDTH);
        break;
      }
    default:
      {
        fprintf(fp,"Invalid Scaling Option!\t");
        break;
      }

  }

#ifdef PERMUTE
    fprintf(fp,"Randomized All2All\t");
#elif INCAST
    fprintf(fp,"Incast All2All\t");
#else
    fprintf(fp,"Round Robin All2All\t");
#endif

    fprintf(fp,"\n");
}

/*
 * Prints all of the timining information for an individual PE as a row
 * to the file specificed by 'fp'. 
 */
static void print_timer_values(FILE * fp)
{
  unsigned int num_records = NUM_PES * NUM_ITERATIONS; 

  for(uint64_t i = 0; i < num_records; ++i) {
    for(int t = 0; t < TIMER_NTIMERS; ++t){
      if(timers[t].all_times != NULL){
        fprintf(fp,"%f\t", timers[t].all_times[i]);
      }
      if(timers[t].all_counts != NULL){
        fprintf(fp,"%u\t", timers[t].all_counts[i]);
      }
    }
    fprintf(fp,"\n");
  }
}

/* 
 * Aggregates the per PE timing information
 */ 
static double * gather_rank_times(_timer_t * const timer)
{
  if(timer->seconds_iter > 0) {

    assert(timer->seconds_iter == timer->num_iters);

    const unsigned int num_records = NUM_PES * timer->seconds_iter;
    double * my_times = shmem_malloc(timer->seconds_iter * sizeof(double));
    assert(my_times);
    memcpy(my_times, timer->seconds, timer->seconds_iter * sizeof(double));

    double * all_times = shmem_malloc( num_records * sizeof(double));
    assert(all_times);

    shmem_barrier_all();

    shmem_fcollect64(all_times, my_times, timer->seconds_iter, 0, 0, NUM_PES, pSync);
    shmem_barrier_all();

    shmem_free(my_times);

    return all_times;
  }
  else{
    return NULL;
  }
}

/*
 * Aggregates the per PE timing 'count' information 
 */
static unsigned int * gather_rank_counts(_timer_t * const timer)
{
  if(timer->count_iter > 0){
    const unsigned int num_records = NUM_PES * timer->num_iters;

    unsigned int * my_counts = shmem_malloc(timer->num_iters * sizeof(unsigned int));
    assert(my_counts);
    memcpy(my_counts, timer->count, timer->num_iters*sizeof(unsigned int));

    unsigned int * all_counts = shmem_malloc( num_records * sizeof(unsigned int) );
    assert(all_counts);

    shmem_barrier_all();

    shmem_collect32(all_counts, my_counts, timer->num_iters, 0, 0, NUM_PES, pSync);

    shmem_barrier_all();

    shmem_free(my_counts);

    return all_counts;
  }
  else{
    return NULL;
  }

}

/*
 * Seeds each rank based on the worker number, rank and time
 */
static inline pcg32_random_t seed_my_chunk(int chunk)
{
  const unsigned int my_rank = shmem_my_pe();
  const unsigned int my_virtual_rank = GET_VIRTUAL_RANK(my_rank, chunk);
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, (uint64_t) my_virtual_rank, (uint64_t) my_virtual_rank );
  return rng;
}

/*
 * Seeds each rank based on the rank number and time
 */
static inline pcg32_random_t seed_my_rank(void)
{
  const unsigned int my_rank = shmem_my_pe();
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, (uint64_t) my_rank, (uint64_t) my_rank );
  return rng;
}

/*
 * Initializes the work array required for SHMEM collective functions
 */
static void init_shmem_sync_array(long * restrict const pSync)
{
  for(uint64_t i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; ++i){
    pSync[i] = _SHMEM_SYNC_VALUE;
  }
  shmem_barrier_all();
}

/*
 * Tests whether or not a file exists. 
 * Returns 1 if file exists
 * Returns 0 if file does not exist
 */
static int file_exists(char * filename)
{
  struct stat buffer;

  if(stat(filename,&buffer) == 0){
    return 1;
  }
  else {
    return 0;
  }

}

#ifdef DEBUG
static void wait_my_turn()
{
  shmem_barrier_all();
  whose_turn = 0;
  shmem_barrier_all();
  const int my_rank = shmem_my_pe();

  shmem_int_wait_until((int*)&whose_turn, SHMEM_CMP_EQ, my_rank);
  sleep(1);

}

static void my_turn_complete()
{
  const int my_rank = shmem_my_pe();
  const int next_rank = my_rank+1;

  if(my_rank < (NUM_PES-1)){ // Last rank updates no one
    shmem_int_put((int *) &whose_turn, &next_rank, 1, next_rank);
  }
  shmem_barrier_all();
}
#endif

#ifdef PERMUTE
/*
 * Creates a randomly ordered array of PEs used in the exchange_keys function
 */
static void create_permutation_array()
{

  permute_array = (int *) malloc( NUM_PES * sizeof(int) );
  assert(permute_array);

  for(uint64_t i = 0; i < NUM_PES; ++i){
    permute_array[i] = i;
  }

  shuffle(permute_array, NUM_PES, sizeof(int));
}

/*
 * Randomly shuffles a generic array
 */
static void shuffle(void * array, size_t n, size_t size)
{
  char tmp[size];
  char * arr = array;
  size_t stride = size * sizeof(char);
  if(n > 1){
    for(size_t i = 0; i < (n - 1); ++i){
      size_t rnd = (size_t) rand();
      size_t j = i + rnd/(RAND_MAX/(n - i) + 1);
      memcpy(tmp, arr + j*stride, size);
      memcpy(arr + j*stride, arr + i*stride, size);
      memcpy(arr + i*stride, tmp, size);
    }
  }
}
#endif

