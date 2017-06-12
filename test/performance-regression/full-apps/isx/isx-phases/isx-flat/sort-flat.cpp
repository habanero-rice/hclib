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

#define _POSIX_C_SOURCE 199309L

#include <shmem.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h> // sleep()
#include <sys/stat.h>
#include <stdint.h>
//#include "isx.h"
//#include "timer.h"
#include "pcg_basic.h"

#include "sort.hpp"
#include "util.hpp"
#include "sort-common.hpp"


static void makeInput(uint64_t keysLength, KEY_TYPE *&keys);

static void countLocalBucketSizes(KEY_TYPE const *const keys, uint64_t keysLength, int *&localBucketSizes);

static void computeLocalBucketOffsets(int const *const localBucketSizes, int *&localBucketOffsets);

static void bucketizeLocalKeys(KEY_TYPE const *const keys, uint64_t keysLength,
                               int const*const localBucketOffsets, KEY_TYPE *&bucketedKeys);

static void getWriteOffsets(int const *const localBucketSizes, int64_t *&writeOffsets);

static void sendKeys(int64_t *writeOffsets, int const *const bucketSizes, int const *const bucketOffsets,
                     int const *const bucketedKeys);

static void countLocalKeys(KEY_TYPE const *const bucketedKeys, uint64_t keysLength, int *&myCounts);


int* bucketSortFlat() {
    int numPhases = 1;

    KEY_TYPE *allKeys;
    KEY_TYPE *phases[numPhases];
    uint64_t phaseSizes[numPhases];

    int *localBucketSizes[numPhases];
    int *localBucketOffsets[numPhases];
    KEY_TYPE *bucketedKeys[numPhases];
    int64_t *writeOffsets[numPhases];
    int *myCounts;

    receiveOffset = 0;

    assert(allKeys = (KEY_TYPE *const) malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE)));
    assert(myCounts = (int *) malloc(BUCKET_WIDTH * sizeof(int)));
    for (int i = 0; i < numPhases; i++) {
        assert(localBucketSizes[i] = (int*) calloc(NUM_BUCKETS, sizeof(int)));
        assert(localBucketOffsets[i] = (int *) malloc(NUM_BUCKETS * sizeof(int)));
        assert(bucketedKeys[i] = (KEY_TYPE *) malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE)));
        assert(writeOffsets[i] = (int64_t *) malloc(sizeof(int64_t) * NUM_PES));

    }

    makeInput(NUM_KEYS_PER_PE, allKeys);

    splitKeysForPhases(numPhases, allKeys, NUM_KEYS_PER_PE, phases, phaseSizes);

    for (int i = 0; i < numPhases; i++) {
        countLocalBucketSizes(phases[i], phaseSizes[i], localBucketSizes[i]);
        getWriteOffsets(localBucketSizes[i], writeOffsets[i]);
        computeLocalBucketOffsets(localBucketSizes[i], localBucketOffsets[i]);
        bucketizeLocalKeys(phases[i], phaseSizes[i], localBucketOffsets[i], bucketedKeys[i]);
        sendKeys(writeOffsets[i], localBucketSizes[i], localBucketOffsets[i], bucketedKeys[i]);
    }

    myBucketSize = receiveOffset;
    countLocalKeys(myBucket, (uint64_t) myBucketSize, myCounts);

    free(allKeys);
    for (int i = 0; i < numPhases; i++) {
        free(localBucketSizes[i]);
        free(localBucketOffsets[i]);
        free(bucketedKeys[i]);
        free(writeOffsets[i]);
    }

    return myCounts;
}



static void makeInput(uint64_t keysLength, KEY_TYPE *&keys) {
    pcg32_random_t rng = seedMyRank();

    for (uint64_t i = 0; i < keysLength; ++i) {
        keys[i] = pcg32_boundedrand_r(&rng, (uint32_t) MAX_KEY_VAL);
    }
}

static void countLocalBucketSizes(KEY_TYPE const *const keys, uint64_t keysLength, int *&localBucketSizes) {
    for (uint64_t i = 0; i < keysLength; ++i) {
        const uint64_t bucketIndex = keys[i] / BUCKET_WIDTH;
        localBucketSizes[bucketIndex]++;
    }
}

static void computeLocalBucketOffsets(int const *const localBucketSizes, int *&localBucketOffsets) {
    localBucketOffsets[0] = 0;
    for (uint64_t i = 1; i < NUM_BUCKETS; i++) {
        localBucketOffsets[i] = localBucketOffsets[i - 1] + localBucketSizes[i - 1];
    }
}

static void bucketizeLocalKeys(KEY_TYPE const *const keys, uint64_t keysLength,
                               int const*const localBucketOffsets, KEY_TYPE *&bucketedKeys) {
    int *bucketingOffsets = (int *) malloc(NUM_BUCKETS * sizeof(int));
    memcpy(bucketingOffsets, localBucketOffsets, NUM_BUCKETS * sizeof(int));
    for (uint64_t i = 0; i < keysLength; ++i) {
        const KEY_TYPE key = keys[i];
        const uint64_t bucket_index = key / BUCKET_WIDTH;
        assert(bucketingOffsets[bucket_index] >= 0);

        uint32_t index = (uint32_t) bucketingOffsets[bucket_index]++;
        assert(index < keysLength);

        bucketedKeys[index] = key;
    }

    free(bucketingOffsets);
}

static void getWriteOffsets(int const *const localBucketSizes, int64_t *&writeOffsets) {
    const int myRank = shmem_my_pe();

    for (uint64_t i = 0; i < NUM_PES; i++) {
        const int targetRank = (const int) ((myRank + i) % NUM_PES);
        writeOffsets[targetRank] = shmem_longlong_fadd(&receiveOffset, localBucketSizes[targetRank], targetRank);
        assert(writeOffsets[targetRank] + localBucketSizes[targetRank] <= (int64_t) KEY_BUFFER_SIZE);
    }
}

static void sendKeys(int64_t *writeOffsets, int const *const bucketSizes, int const *const bucketOffsets,
                     int const *const bucketedKeys) {
    const int myRank = shmem_my_pe();

    memcpy(&myBucket[writeOffsets[myRank]],
           &bucketedKeys[bucketOffsets[myRank]],
           bucketSizes[myRank] * sizeof(KEY_TYPE));

    for (uint64_t i = 1; i < NUM_PES; i++) {
        const int targetRank = (const int) ((myRank + i) % NUM_PES);
        shmem_int_put(&(myBucket[writeOffsets[targetRank]]),
                      &(bucketedKeys[bucketOffsets[targetRank]]),
                      bucketSizes[targetRank],
                      targetRank);

        writeOffsets[targetRank] += bucketSizes[targetRank];
    }

    shmem_barrier_all();
}

static void countLocalKeys(KEY_TYPE const *const bucketedKeys, uint64_t keysLength, int *&myCounts) {
    memset(myCounts, 0, BUCKET_WIDTH * sizeof(int));

    const int myRank = shmem_my_pe();
    uint64_t myMinKey = myRank * BUCKET_WIDTH;

    // Count the occurences of each key in my bucket
    for(uint64_t i = 0; i < keysLength; ++i){
        uint64_t keyIndex = bucketedKeys[i] - myMinKey;

        assert(bucketedKeys[i] >= (KEY_TYPE) myMinKey);
        assert(keyIndex < BUCKET_WIDTH);

        myCounts[keyIndex]++;
    }
}





























