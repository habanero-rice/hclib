//
// Created by sm108 on 5/15/17.
//

#ifndef ISX_PHASES_SORT_COMMON_HPP_H
#define ISX_PHASES_SORT_COMMON_HPP_H

// TODO: fix - cinttypes must be defined before shmem.h
#include <cinttypes>
#include <shmem.h>
#include "params.h"


#ifndef SCALING_OPTION
#define SCALING_OPTION WEAK
#endif

#if defined(EDISON_DATASET)
#define KEY_BUFFER_SIZE ((1uLL << 31uLL))
#elif defined(DAVINCI_DATASET)
#define KEY_BUFFER_SIZE ((1uLL << 27uLL) + 0uLL)
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


void splitKeysForPhases(int numPhases, KEY_TYPE *allKeys, uint64_t keysLength,
                               KEY_TYPE **phases, uint64_t *phaseSizes);


#endif //ISX_PHASES_SORT_COMMON_HPP_H
