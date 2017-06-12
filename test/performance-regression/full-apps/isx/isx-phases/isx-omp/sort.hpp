#ifndef SORT_HPP
#define SORT_HPP

#include <cstdint>
#include <atomic>

#include "params.h"

int *sort();

void makeInput(uint64_t keysLength, KEY_TYPE *&keys, std::atomic<bool> &localSense);

void countLocalBucketSizes(KEY_TYPE const *const keys, uint64_t keysLength,
                           int *&localBucketSizes, int **&bucketSizesPerThread, std::atomic<bool> &localSense);

void computeLocalBucketOffsets(int const *const localBucketSizes, int *&localBucketOffsets,
                               int *&sendOffsets, std::atomic<bool> &localSense);

void bucketizeLocalKeys(KEY_TYPE const *const keys, uint64_t keysLength, int *const localBucketOffsets,
                        int **bucketSizesPerThread, KEY_TYPE *&bucketedKeys, std::atomic<bool> &localSense);

void countLocalKeys(KEY_TYPE const *const bucketedKeys, uint64_t keysLength,
                    int *&myCounts, int *&perChunkCounts, std::atomic<bool> &localSense);

void getWriteOffsets(int const *const localBucketSizes, int64_t *&writeOffsets);

void sendKeys(int const *const bucketOffsets, int const *const bucketSizes, int64_t *writeOffsets,
              int const *const bucketedKeys);

#endif
