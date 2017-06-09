#include <cinttypes>

#include "util.hpp"
#include <shmem.h>
#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include "sort-common.hpp"

pcg32_random_t seedMyRank()
{
    const unsigned int my_rank = shmem_my_pe();
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, (uint64_t) my_rank, (uint64_t) my_rank );
    return rng;
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