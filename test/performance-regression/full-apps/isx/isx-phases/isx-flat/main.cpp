#include <shmem.h>
#include <cassert>


#include "params.h"
#include "sort.hpp"
#include "parse-args.h"
#include "verification.hpp"
#include "sort-common.hpp"
#include "util.hpp"

#define SHMEM_BARRIER_AT_START    { /*timer_start(&timers[TIMER_BARRIER_START]);*/ shmem_barrier_all(); /*timer_stop(&timers[TIMER_BARRIER_START]);*/ }
#define SHMEM_BARRIER_AT_END      { /*timer_start(&timers[TIMER_BARRIER_END]);*/ shmem_barrier_all(); /*timer_stop(&timers[TIMER_BARRIER_END]);*/ }




int main(int argc, char *argv[]) {
    shmem_init();

    parseArgsFlat(argc, argv);
    assert(myBucket = (KEY_TYPE *) shmem_malloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE)));
    initShmemSyncArray(pSync);

    for (uint64_t i = 0; i < BURN_IN + NUM_ITERATIONS; i++) {
        SHMEM_BARRIER_AT_START;
        unsigned long long start = currentTimeNs();
        int *myCounts = bucketSortFlat();
        SHMEM_BARRIER_AT_END;
        unsigned long long end = currentTimeNs();

        if (i >= BURN_IN && shmem_my_pe() == ROOT_PE) {
            fprintf(stderr, "Flat - Execution time[%d]: %lf s\n", shmem_my_pe(), (end - start) / 1e9);
        } else if (shmem_my_pe() == ROOT_PE) {
            fprintf(stderr, "Flat - Burn in iteration: %lf.\n", (end - start) / 1e9);
        }

        if (i + 1 == BURN_IN + NUM_ITERATIONS) {
            verifyResults(myCounts, myBucket);
            if (shmem_my_pe() == ROOT_PE) {
                fprintf(stderr, "Validation successful\n");
            }
        }
        free(myCounts);
    }

    shmem_finalize();
    return 0;
}
