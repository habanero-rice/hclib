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
#define HCLIB_VERSION
#define _POSIX_C_SOURCE 199309L

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
#include "isx.h"
#include "timer.h"
#include "pcg_basic.h"
#include <omp.h>

#define ROOT_PE 0
#define INTS_PER_CACHE_LINE (128 / sizeof(int))

#define ISX_PROFILING

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

static int num_threads = -1;

#define SHMEM_BARRIER_AT_START    { timer_start(&timers[TIMER_BARRIER_START]); shmem_barrier_all(); timer_stop(&timers[TIMER_BARRIER_START]); }
#define SHMEM_BARRIER_AT_EXCHANGE { timer_start(&timers[TIMER_BARRIER_EXCHANGE]); shmem_barrier_all(); timer_stop(&timers[TIMER_BARRIER_EXCHANGE]); }
#define SHMEM_BARRIER_AT_END      { timer_start(&timers[TIMER_BARRIER_END]); shmem_barrier_all(); timer_stop(&timers[TIMER_BARRIER_END]); }

// #define EXTRA_STATS

#ifdef EXTRA_STATS
float avg_time=0, avg_time_all2all = 0;
#endif

// #define KEY_BUFFER_SIZE ((1uLL<<28uLL) + 60000)
// #define KEY_BUFFER_SIZE ((1uLL<<28uLL))
// #define KEY_BUFFER_SIZE ((1uLL<<26uLL))
// #define KEY_BUFFER_SIZE ((1uLL<<28uLL) + 80000)
// #define KEY_BUFFER_SIZE ((1uLL<<28uLL) + 50000000)
#define KEY_BUFFER_SIZE ((1uLL<<31uLL))

// The receive array for the All2All exchange
// KEY_TYPE my_bucket_keys[KEY_BUFFER_SIZE];
KEY_TYPE *my_bucket_keys;

#ifdef PERMUTE
int * permute_array;
#endif

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

int main(const int argc,  char ** argv)
{
  // const char *deps[] = { "system", "openshmem" };
            {
  shmem_init();

#pragma omp parallel
#pragma omp single
  num_threads = omp_get_num_threads();
   
    // fprintf(stderr, "Trying to allocate %llu bytes\n", KEY_BUFFER_SIZE * sizeof(KEY_TYPE));
    my_bucket_keys = (KEY_TYPE *)shmem_malloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE));
    assert(my_bucket_keys);

  #ifdef EXTRA_STATS
  _timer_t total_time;
  if(shmem_my_pe() == 0) {
    printf("\n-----\nmkdir timedrun fake\n\n");
    timer_start(&total_time);
  }
#endif

  init_shmem_sync_array(pSync); 

  char * log_file = parse_params(argc, argv);

  int err = bucket_sort();

  log_times(log_file);

  #ifdef EXTRA_STATS
  if(shmem_my_pe() == 0) {
    just_timer_stop(&total_time);
    double tTime = ( total_time.stop.tv_sec - total_time.start.tv_sec ) + ( total_time.stop.tv_nsec - total_time.start.tv_nsec )/1E9;
    avg_time *= 1000;
    avg_time_all2all *= 1000;

    printf("\n============================ MMTk Statistics Totals ============================\n");
    if(NUM_ITERATIONS == 1) { //TODO: fix time calculation below for more number of iterations
      printf("time.mu\tt.ATA_KEYS\tt.MAKE_INPUT\tt.COUNT_BUCKET_SIZES\tt.BUCKETIZE\tt.COMPUTE_OFFSETS\tt.LOCAL_SORT\tBARRIER_AT_START\tBARRIER_AT_EXCHANGE\tBARRIER_AT_END\tnWorkers\tnPEs\n");
      double TIMES[TIMER_NTIMERS];
      memset(TIMES, 0x00, sizeof(double) * TIMER_NTIMERS);
      for(int i=0; i<NUM_PES; i++) {
        for(int t = 0; t < TIMER_NTIMERS; ++t){
          if(timers[t].all_times != NULL){
            TIMES[t] += timers[t].all_times[i];
          }
        }
      }
      for(int t = 0; t < TIMER_NTIMERS; ++t){
        printf("%.3f\t", (TIMES[t]/NUM_PES)*1000);
      }
      printf("1\t%d\n",NUM_PES);
      printf("Total time: %.3f\n",(TIMES[0]/NUM_PES)*1000);
    }
    else {
      printf("time.mu\ttimeAll2All\tnWorkers\tnPEs\n");
      printf("%.3f\t%.3f\t1\t%d\n",avg_time,avg_time_all2all,NUM_PES);
      printf("Total time: %.3f\n",avg_time);
    }
    printf("------------------------------ End MMTk Statistics -----------------------------\n");
    printf("===== TEST PASSED in %.3f msec =====\n",(tTime*1000));
  }
#endif

  shmem_finalize();
    }
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

    // shmem_finalize();
    exit(1);
  }

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
        NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1]));
        sprintf(scaling_msg,"WEAK");
        break;
      }

    case WEAK_ISOBUCKET:
      {
        NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1]));
        BUCKET_WIDTH = ISO_BUCKET_WIDTH; 
        MAX_KEY_VAL = (uint64_t) (NUM_PES * BUCKET_WIDTH);
        sprintf(scaling_msg,"WEAK_ISOBUCKET");
        break;
      }

    default:
      {
        if(shmem_my_pe() == 0){
          printf("Invalid scaling option! See params.h to define the scaling option.\n");
        }

        // shmem_finalize();
        exit(1);
        break;
      }
  }

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
    printf("  Max Key Value: %" PRIu64 "\n", MAX_KEY_VAL);
    printf("  Bucket Width: %" PRIu64 "\n", BUCKET_WIDTH);
    printf("  Number of Iterations: %u\n", NUM_ITERATIONS);
    printf("  Number of PEs: %" PRIu64 "\n", NUM_PES);
    printf("  Worker threads per PE: %d\n", omp_get_max_threads());
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

  for(uint64_t i = 0; i < (NUM_ITERATIONS + BURN_IN); ++i)
  {

    // Reset timers after burn in 
    if(i == BURN_IN){
        init_timers(NUM_ITERATIONS);
    } 

    SHMEM_BARRIER_AT_START;

    timer_start(&timers[TIMER_TOTAL]);

    KEY_TYPE * my_keys = make_input();

    int **bucket_counts_per_chunk;
    int * local_bucket_sizes = count_local_bucket_sizes(my_keys,
            &bucket_counts_per_chunk);

    int * send_offsets;
    int * local_bucket_offsets = compute_local_bucket_offsets(local_bucket_sizes,
                                                                   &send_offsets);

    KEY_TYPE * my_local_bucketed_keys =  bucketize_local_keys(my_keys,
            local_bucket_offsets, bucket_counts_per_chunk);

    KEY_TYPE * my_bucket_keys = exchange_keys(send_offsets, 
                                              local_bucket_sizes,
                                              my_local_bucketed_keys);

    my_bucket_size = receive_offset;

    int * my_local_key_counts = count_local_keys(my_bucket_keys);

    SHMEM_BARRIER_AT_END;

    timer_stop(&timers[TIMER_TOTAL]);

    // Only the last iteration is verified
    if(i == NUM_ITERATIONS) { 
      err = verify_results(my_local_key_counts, my_bucket_keys);
    }

    // Reset receive_offset used in exchange_keys
    receive_offset = 0;

    free(my_local_bucketed_keys);
    free(my_keys);
    free(local_bucket_sizes);
    free(local_bucket_offsets);
    free(send_offsets);
    free(my_local_key_counts);
    for (int i = 0; i < num_threads; i++) {
        free(bucket_counts_per_chunk[i]);
    }
    free(bucket_counts_per_chunk);

    shmem_barrier_all();
  }

  return err;
}


/*
 * Generates uniformly random keys [0, MAX_KEY_VAL] on each rank using the time and rank
 * number as a seed
 */
static KEY_TYPE * make_input(void)
{
  timer_start(&timers[TIMER_INPUT]);

  KEY_TYPE * const my_keys = (KEY_TYPE *)malloc(NUM_KEYS_PER_PE *
          sizeof(KEY_TYPE));
  assert(my_keys);

#ifdef ISX_PROFILING
  unsigned long long start = current_time_ns();
#endif

#pragma omp parallel
  {
      const int i = omp_get_thread_num();
      pcg32_random_t rng = seed_my_rank(i);
      uint64_t chunk_size = (NUM_KEYS_PER_PE + num_threads - 1) / num_threads;
      uint64_t start_chunk = i * chunk_size;
      uint64_t end_chunk = (i + 1) * chunk_size;
      if (end_chunk > NUM_KEYS_PER_PE) end_chunk = NUM_KEYS_PER_PE;

      for(uint64_t ii = start_chunk; ii < end_chunk; ++ii) {
          my_keys[ii] = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
      }
  }

#ifdef ISX_PROFILING
  unsigned long long end = current_time_ns();
  if (shmem_my_pe() == 0)
  printf("Making input took %llu ns\n", end - start);
#endif

  timer_stop(&timers[TIMER_INPUT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  // const int my_rank = ::shmem_my_pe();
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


/*
 * Computes the size of each bucket by iterating all keys and incrementing
 * their corresponding bucket's size
 */
static  int * count_local_bucket_sizes(KEY_TYPE const * const my_keys,
        int ***bucket_counts_per_chunk_out)
{
  int * const local_bucket_sizes = (int *)malloc(NUM_BUCKETS * sizeof(int));
  assert(local_bucket_sizes);

  timer_start(&timers[TIMER_BCOUNT]);

  init_array(local_bucket_sizes, NUM_BUCKETS);

#ifdef ISX_PROFILING
  unsigned long long start = current_time_ns();
#endif

  int **bucket_counts_per_chunk = (int **)malloc(num_threads * sizeof(int *));
#pragma omp parallel
  {
      const int i = omp_get_thread_num();

      int *bucket_sizes = NULL;
      if (NUM_BUCKETS >= INTS_PER_CACHE_LINE) {
          bucket_sizes = (int *)calloc(NUM_BUCKETS, sizeof(int));
      } else {
          bucket_sizes = (int *)calloc(INTS_PER_CACHE_LINE, sizeof(int));
      }

      uint64_t chunk_size = (NUM_KEYS_PER_PE + num_threads - 1) / num_threads;
      uint64_t start_chunk = i * chunk_size;  
      uint64_t end_chunk = (i + 1) * chunk_size; 
      if (end_chunk > NUM_KEYS_PER_PE) end_chunk = NUM_KEYS_PER_PE; 

      for (uint64_t ii = start_chunk; ii < end_chunk; ++ii) {
          const uint32_t bucket_index = my_keys[ii] / BUCKET_WIDTH;
          bucket_sizes[bucket_index]++;
      }

      bucket_counts_per_chunk[i] = bucket_sizes;
  }

  for (int i = 0; i < num_threads; i++) {
      int *worker_bucket_sizes = bucket_counts_per_chunk[i];
      for (unsigned b = 0; b < NUM_BUCKETS; b++) {
          local_bucket_sizes[b] += worker_bucket_sizes[b];
      }
      // free(worker_bucket_sizes);
  }
  *bucket_counts_per_chunk_out = bucket_counts_per_chunk;
  // free(bucket_counts_per_chunk);

#ifdef ISX_PROFILING
  unsigned long long end = current_time_ns();
  if (shmem_my_pe() == 0)
  printf("Counting local bucket sizes took %llu ns\n", end - start);
#endif

  timer_stop(&timers[TIMER_BCOUNT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  // const int my_rank = ::shmem_my_pe();
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
static  int * compute_local_bucket_offsets(int const * const local_bucket_sizes,
                                                 int ** send_offsets)
{
  int * const local_bucket_offsets = (int *)malloc(NUM_BUCKETS * sizeof(int));
  assert(local_bucket_offsets);

  timer_start(&timers[TIMER_BOFFSET]);

  (*send_offsets) = (int *)malloc(NUM_BUCKETS * sizeof(int));
  assert(*send_offsets);

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
  // const int my_rank = ::shmem_my_pe();
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
static  KEY_TYPE * bucketize_local_keys(KEY_TYPE const * const my_keys,
                                              int * const local_bucket_offsets,
                                              int **bucket_counts_per_chunk)
{
  KEY_TYPE * const my_local_bucketed_keys = (KEY_TYPE *)malloc(
          NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
  assert(my_local_bucketed_keys);

  timer_start(&timers[TIMER_BUCKETIZE]);

#ifdef ISX_PROFILING
  unsigned long long start = current_time_ns();
#endif

  if (num_threads == 1) {
      for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i){
        const KEY_TYPE key = my_keys[i];
        const uint32_t bucket_index = key / BUCKET_WIDTH;
        uint32_t index;
        assert(local_bucket_offsets[bucket_index] >= 0);
        index = local_bucket_offsets[bucket_index]++;
        assert(index < NUM_KEYS_PER_PE);
        my_local_bucketed_keys[index] = key;
      }
  } else {
      int *chunk_bucket_offsets = (int *)malloc(NUM_BUCKETS * num_threads *
              sizeof(int));

// #pragma omp parallel for schedule(static)
      for (uint64_t b = 0; b < NUM_BUCKETS; b++) {
          chunk_bucket_offsets[b * num_threads + 0] = local_bucket_offsets[b];
          for (int w = 1; w < num_threads; w++) {
              chunk_bucket_offsets[b * num_threads + w] =
                  chunk_bucket_offsets[b * num_threads + w - 1] +
                  bucket_counts_per_chunk[w - 1][b];
          }
      }

#pragma omp parallel
      {
          const int c = omp_get_thread_num();

          uint64_t chunk_size = (NUM_KEYS_PER_PE + num_threads - 1) / num_threads;
          uint64_t start_chunk = c * chunk_size;
          uint64_t end_chunk = (c + 1) * chunk_size;
          if (end_chunk > NUM_KEYS_PER_PE) end_chunk = NUM_KEYS_PER_PE;

          int *tmp = NULL;
          if (NUM_BUCKETS < INTS_PER_CACHE_LINE) {
              tmp = (int *)malloc(INTS_PER_CACHE_LINE * sizeof(int));
          } else {
              tmp = (int *)malloc(NUM_BUCKETS * sizeof(int));
          }
          assert(tmp);

          for (unsigned i = 0; i < NUM_BUCKETS; i++) {
              tmp[i] = chunk_bucket_offsets[i * num_threads + c];
          }

          for (uint64_t i = start_chunk; i < end_chunk; i++) {
              const KEY_TYPE key = my_keys[i];
              const uint32_t bucket_index = key / BUCKET_WIDTH;
              uint32_t index = tmp[bucket_index]++;
              assert(index < NUM_KEYS_PER_PE);
              my_local_bucketed_keys[index] = key;
          }
          free(tmp);
      }

      free(chunk_bucket_offsets);
  }

#ifdef ISX_PROFILING
  unsigned long long end = current_time_ns();
  if (shmem_my_pe() == 0)
      printf("Bucketizing took %llu ns\n", end - start);
#endif

  timer_stop(&timers[TIMER_BUCKETIZE]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  // const int my_rank = ::shmem_my_pe();
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
static  KEY_TYPE * exchange_keys(int const * const send_offsets,
                                       int const * const local_bucket_sizes,
                                       KEY_TYPE const * const my_local_bucketed_keys)
{
  timer_start(&timers[TIMER_ATA_KEYS]);

  const int my_rank = shmem_my_pe();
  // const int my_rank = ::shmem_my_pe();
  unsigned int total_keys_sent = 0;

  // Keys destined for local key buffer can be written with memcpy
  const long long int write_offset_into_self = shmem_longlong_fadd(
          &receive_offset, (long long int)local_bucket_sizes[my_rank], my_rank);
  assert((unsigned long long)write_offset_into_self +
          (unsigned long long)local_bucket_sizes[my_rank] <= KEY_BUFFER_SIZE);

  memcpy(&my_bucket_keys[write_offset_into_self], 
         &my_local_bucketed_keys[send_offsets[my_rank]], 
         local_bucket_sizes[my_rank]*sizeof(KEY_TYPE));

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

    const long long int write_offset_into_target = shmem_longlong_fadd(
            &receive_offset, (long long int)my_send_size, target_pe);

#ifdef DEBUG
    printf("Rank: %d Target: %d Offset into target: %lld Offset into myself: %d Send Size: %d\n",
        my_rank, target_pe, write_offset_into_target, read_offset_from_self, my_send_size);
#endif

    if ((unsigned long long)write_offset_into_target +
            (unsigned long long)my_send_size > KEY_BUFFER_SIZE) {
        fprintf(stderr, "Put offset %llu was greater than KEY_BUFFER_SIZE "
                "%llu\n", (unsigned long long)write_offset_into_target +
                (unsigned long long)my_send_size, KEY_BUFFER_SIZE);
        exit(1);
    }
    shmem_int_put(&(my_bucket_keys[write_offset_into_target]), 
                  &(my_local_bucketed_keys[read_offset_from_self]), 
                  my_send_size, 
                  target_pe);

    total_keys_sent += my_send_size;
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
static  int * count_local_keys(KEY_TYPE const * const my_bucket_keys)
{
  int * const my_local_key_counts = (int *)calloc(BUCKET_WIDTH, sizeof(int));
  assert(my_local_key_counts);

  timer_start(&timers[TIMER_SORT]);

  const int my_rank = shmem_my_pe();
  // const int my_rank = ::shmem_my_pe();
  const int my_min_key = my_rank * BUCKET_WIDTH;

#ifdef ISX_PROFILING
  unsigned long long start = current_time_ns();
#endif

  int *per_chunk_counts = (int *)calloc(num_threads * BUCKET_WIDTH, sizeof(int));
  assert(per_chunk_counts);

#pragma omp parallel
  {
      const int c = omp_get_thread_num();

      long long int chunk_size = (my_bucket_size + num_threads - 1) / num_threads;
      long long int start_chunk = c * chunk_size;
      long long int end_chunk = (c + 1) * chunk_size;
      if (end_chunk > my_bucket_size) end_chunk = my_bucket_size;

      int *counts = per_chunk_counts + (c * BUCKET_WIDTH);

      for (long long int i = start_chunk; i < end_chunk; i++) {
          const unsigned int key_index = my_bucket_keys[i] - my_min_key;

          counts[key_index]++;
      }
  }

#ifdef ISX_PROFILING
  unsigned long long intermediate = current_time_ns();
#endif

#pragma omp parallel
  {
      const int c = omp_get_thread_num();

      unsigned chunk_size = (BUCKET_WIDTH + num_threads - 1) / num_threads;
      unsigned start_chunk = c * chunk_size;
      unsigned end_chunk = (c + 1) * chunk_size;
      if (end_chunk > BUCKET_WIDTH) end_chunk = BUCKET_WIDTH;

      for (int c = 0; c < num_threads; c++) {
          for (unsigned i = start_chunk; i < end_chunk; i++) {
              my_local_key_counts[i] += per_chunk_counts[c * BUCKET_WIDTH + i];
          }
      }
  }

  free(per_chunk_counts);
#ifdef ISX_PROFILING
  unsigned long long end = current_time_ns();
  if (shmem_my_pe() == 0)
  printf("Counting local took %llu ns for stage 1, %llu ns for stage 2, "
          "my_bucket_size = %lld\n", intermediate - start, end - intermediate,
          my_bucket_size);
#endif

  // // Count the occurences of each key in my bucket
  // for(long long int i = 0; i < my_bucket_size; ++i){
  //   const unsigned int key_index = my_bucket_keys[i] - my_min_key;

  //   assert(my_bucket_keys[i] >= my_min_key);
  //   assert(key_index < BUCKET_WIDTH);

  //   my_local_key_counts[key_index]++;
  // }

  timer_stop(&timers[TIMER_SORT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[4096];
  sprintf(msg,"Rank %d: Bucket Size %lld | Local Key Counts:", my_rank, my_bucket_size);
  for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_local_key_counts[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif

  return my_local_key_counts;
}

/*
 * Verifies the correctness of the sort. 
 * Ensures all keys are within a PE's bucket boundaries.
 * Ensures the final number of keys is equal to the initial.
 */
static int verify_results(int const * const my_local_key_counts,
                           KEY_TYPE const * const my_local_keys)
{

  shmem_barrier_all();

  int error = 0;

  const int my_rank = shmem_my_pe();
  // const int my_rank = ::shmem_my_pe();

  const int my_min_key = my_rank * BUCKET_WIDTH;
  const int my_max_key = (my_rank+1) * BUCKET_WIDTH - 1;

#ifdef ISX_PROFILING
  unsigned long long start = current_time_ns();
#endif
  // Verify all keys are within bucket boundaries
  for(long long int i = 0; i < my_bucket_size; ++i){
    const int key = my_local_keys[i];
    if((key < my_min_key) || (key > my_max_key)){
      printf("Rank %d Failed Verification!\n",my_rank);
      printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
      error = 1;
    }
  }

#ifdef ISX_PROFILING
  unsigned long long end = current_time_ns();
  if (shmem_my_pe() == 0)
  printf("Verifying took %llu ns\n", end - start);
#endif

  // Verify the sum of the key population equals the expected bucket size
  long long int bucket_size_test = 0;
  for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
    bucket_size_test +=  my_local_key_counts[i];
  }
  if(bucket_size_test != my_bucket_size){
      printf("Rank %d Failed Verification!\n",my_rank);
      printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, my_bucket_size);
      error = 1;
  }

  // Verify the final number of keys equals the initial number of keys
  static long long int total_num_keys = 0;
  shmem_longlong_sum_to_all(&total_num_keys, &my_bucket_size, 1, 0, 0, NUM_PES,
          llWrk, pSync);
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
  // if(::shmem_my_pe() == ROOT_PE)
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
    double * my_times = (double *)shmem_malloc(timer->seconds_iter * sizeof(double));
    assert(my_times);
    memcpy(my_times, timer->seconds, timer->seconds_iter * sizeof(double));

    double * all_times = (double *)shmem_malloc( num_records * sizeof(double));
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

    unsigned int * my_counts = (unsigned int *)shmem_malloc(
            timer->num_iters * sizeof(unsigned int));
    assert(my_counts);
    memcpy(my_counts, timer->count, timer->num_iters*sizeof(unsigned int));

    unsigned int * all_counts = (unsigned int *)shmem_malloc(
            num_records * sizeof(unsigned int) );
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
 * Seeds each rank based on the rank number and time
 */
static  pcg32_random_t seed_my_rank(const int chunk)
{
  const unsigned int my_rank = shmem_my_pe();
  const unsigned int virtual_rank = my_rank * num_threads + chunk;
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, (uint64_t) virtual_rank, (uint64_t) virtual_rank );
  return rng;
}

/*
 * Initializes the work array required for SHMEM collective functions
 */
static void init_shmem_sync_array(long * const pSync)
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
  // const int my_rank = ::shmem_my_pe();

  shmem_int_wait_until((int*)&whose_turn, SHMEM_CMP_EQ, my_rank);
  sleep(1);

}

static void my_turn_complete()
{
  const int my_rank = shmem_my_pe();
  // const int my_rank = ::shmem_my_pe();
  const int next_rank = my_rank+1;

  if(my_rank < (NUM_PES-1)){ // Last rank updates no one
    shmem_int_put((int *)&whose_turn, &next_rank, 1, next_rank);
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

