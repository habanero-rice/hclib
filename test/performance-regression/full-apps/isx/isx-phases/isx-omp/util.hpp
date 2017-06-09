#ifndef UTIL_HPP
#define UTIL_HPP

#include <cstdio>
#include <atomic>
#include "pcg_basic.h"

void parseArgs(int &argc, char **&argv);

pcg32_random_t seedMyRank(int chunk, int numThreads);

unsigned long long currentTimeNs();

void initShmemSyncArray(long * const pSync);

void barrierWait(int n, std::atomic<int> &count, std::atomic<bool> &sense, std::atomic<bool> &localSense);

template <class T>
void splitRange(int id, int maxId, T inBegin, T inEnd, T &outBegin, T &outEnd) {
    uint64_t size = (inEnd - inBegin + maxId - 1) / maxId;
    outBegin = size * id;
    outEnd = size * (id + 1);
    if (outEnd > inEnd) {
        outEnd = inEnd;
    }
}

#endif
