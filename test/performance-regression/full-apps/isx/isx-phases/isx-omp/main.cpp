#include <shmem.h>
#include <cassert>

#include "params.h"
#include "global.hpp"
#include "sort.hpp"
#include "util.hpp"
#include "verification.hpp"

#define SHMEM_BARRIER_AT_START    { /*timer_start(&timers[TIMER_BARRIER_START]);*/ shmem_barrier_all(); /*timer_stop(&timers[TIMER_BARRIER_START]);*/ }
#define SHMEM_BARRIER_AT_END      { /*timer_start(&timers[TIMER_BARRIER_END]);*/ shmem_barrier_all(); /*timer_stop(&timers[TIMER_BARRIER_END]);*/ }

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

int main(int argc, char *argv[]) {
    shmem_init();

    parseArgs(argc, argv);
    assert(myBucket = (KEY_TYPE *) shmem_malloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE)));
    initShmemSyncArray(pSync);

    for (int i = 0; i < BURN_IN + NUM_ITERATIONS; i++) {
        SHMEM_BARRIER_AT_START;
        unsigned long long start = currentTimeNs();
        int *myCounts = sort();
        SHMEM_BARRIER_AT_END;
        unsigned long long end = currentTimeNs();

        if (i >= BURN_IN) {
            fprintf(stderr, "OMP - Execution time[%d]: %lf s\n", shmem_my_pe(), (end - start) / 1e9);
        } else if (shmem_my_pe() == 0) {
            fprintf(stderr, "OMP - Burn in iteration: %lf.\n", (end - start) / 1e9);
        }

        if (i + 1 == BURN_IN + NUM_ITERATIONS) {
            verifyResults(myCounts, myBucket);
        }
        free(myCounts);
    }

    shmem_finalize();
    return 0;
}
