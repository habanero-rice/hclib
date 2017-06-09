#include "sort-common.hpp"

#include <cstdlib>
#include <cstdio>
#include <cassert>


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

long long int receiveOffset = 0;
long long int myBucketSize = 0;

KEY_TYPE *myBucket;


void splitKeysForPhases(int numPhases, KEY_TYPE *allKeys, uint64_t keysLength,
                               KEY_TYPE **phases, uint64_t *phaseSizes) {
    uint64_t remainingKeys = keysLength;
    uint64_t keysPerPhase = keysLength / numPhases;

    phases[0] = allKeys;
    phaseSizes[0] = keysPerPhase;
    for (int i = 1; i < numPhases; i++) {
        if (remainingKeys >= keysPerPhase) {
            phaseSizes[i] = keysPerPhase;
            remainingKeys -= keysPerPhase;
        } else {
            phaseSizes[i] = remainingKeys;
            remainingKeys = 0;
        }
        phases[i] = phases[i - 1] + phaseSizes[i];
    }
}