#include <cinttypes>

#include "util.hpp"
#include <shmem.h>
#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include "global.hpp"

void parseArgs(int &argc, char **&argv) {
    if (argc != 3) {
        if (shmem_my_pe() == 0) {
            printf("Usage:  \n");
            printf("  ./%s <total num keys(strong) | keys per pe(weak)> <log_file>\n", argv[0]);
        }

        exit(1);
    }

    NUM_PES = (uint64_t) shmem_n_pes();
    MAX_KEY_VAL = DEFAULT_MAX_KEY;
    NUM_BUCKETS = NUM_PES;
    BUCKET_WIDTH = (uint64_t) ceil((double) MAX_KEY_VAL / NUM_BUCKETS);
    //char *log_file = argv[2];
    char scaling_msg[64];

    switch (SCALING_OPTION) {
        case STRONG: {
            TOTAL_KEYS = (uint64_t) atoi(argv[1]);
            NUM_KEYS_PER_PE = (uint64_t) ceil((double) TOTAL_KEYS / NUM_PES);
            sprintf(scaling_msg, "STRONG");
            break;
        }

        case WEAK: {
            NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1]));
            sprintf(scaling_msg, "WEAK");
            break;
        }

        case WEAK_ISOBUCKET: {
            NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1]));
            BUCKET_WIDTH = ISO_BUCKET_WIDTH;
            MAX_KEY_VAL = (uint64_t) (NUM_PES * BUCKET_WIDTH);
            sprintf(scaling_msg, "WEAK_ISOBUCKET");
            break;
        }

        default: {
            if (shmem_my_pe() == 0) {
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

    if (shmem_my_pe() == 0) {
        printf("ISx v%1d.%1d\n", MAJOR_VERSION_NUMBER, MINOR_VERSION_NUMBER);
#ifdef PERMUTE
        printf("Random Permute Used in ATA.\n");
#endif
        printf("  Number of Keys per PE: %" PRIu64 "\n", NUM_KEYS_PER_PE);
        printf("  Max Key Value: %" PRIu64 "\n", MAX_KEY_VAL);
        printf("  Bucket Width: %" PRIu64 "\n", BUCKET_WIDTH);
        printf("  Number of Iterations: %u\n", NUM_ITERATIONS);
        printf("  Number of PEs: %" PRIu64 "\n", NUM_PES);
        printf("  Worker threads per PE: %d\n", omp_get_max_threads());
        printf("  %s Scaling!\n", scaling_msg);
    }

    //return log_file;
}

pcg32_random_t seedMyRank(int chunk, int numThreads) {
    int myRank = shmem_my_pe();
    uint64_t virtual_rank = (uint64_t) (myRank * numThreads + chunk);

    pcg32_random_t rng;
    pcg32_srandom_r(&rng, virtual_rank, virtual_rank);
    return rng;
}

unsigned long long currentTimeNs() {
#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    unsigned long long s = 1000000000ULL * (unsigned long long)mts.tv_sec;
    return (unsigned long long)mts.tv_nsec + s;
#else
    struct timespec t = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long) t.tv_sec;
    return (((unsigned long long) t.tv_nsec)) + s;
#endif
}

void initShmemSyncArray(long *const pSync) {
    for (uint64_t i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; ++i) {
        pSync[i] = _SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();
}

void barrierWait(int n, std::atomic<int> &count, std::atomic<bool> &sense, std::atomic<bool> &localSense) {
    localSense.store(!localSense.load());
    if (count.fetch_sub(1) == 1) {
        count.store(n);
        sense.store(localSense.load());
    } else {
        while (sense.load() != localSense.load());
    }
}
