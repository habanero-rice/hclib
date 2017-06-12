#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include "params.h"

#include <shmem.h>
#include <cstdint>

#ifndef SCALING_OPTION
#define SCALING_OPTION WEAK
#endif

#if defined(EDISON_DATASET)
#define KEY_BUFFER_SIZE ((1uLL<<31uLL))
#elif defined(DAVINCI_DATASET)
#define KEY_BUFFER_SIZE ((1uLL<<29uLL))
#else
#error No cluster specified
#endif

#ifndef ROOT_PE
#define ROOT_PE 0
#endif

extern int pWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
extern double dWrk[_SHMEM_REDUCE_SYNC_SIZE];
extern long long int llWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
extern long pSync[_SHMEM_REDUCE_SYNC_SIZE];

extern uint64_t NUM_PES; // Number of parallel workers
extern uint64_t TOTAL_KEYS; // Total number of keys across all PEs
extern uint64_t NUM_KEYS_PER_PE; // Number of keys generated on each PE
extern uint64_t NUM_BUCKETS; // The number of buckets in the bucket sort
extern uint64_t BUCKET_WIDTH; // The size of each bucket
extern uint64_t MAX_KEY_VAL; // The maximum possible generated key value

extern long long int receiveOffset;
extern long long int myBucketSize;

extern KEY_TYPE *myBucket;

#endif
