#include "sort.hpp"

#include <shmem.h>
#include <omp.h>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cinttypes>
#include <unistd.h>
#include <atomic>

#include "global.hpp"
#include "util.hpp"
#include "verification.hpp"

#define INTS_PER_CACHE_LINE (128 / sizeof(int))

#define PL fprintf(stderr, "%d %d %d\n", __LINE__, shmem_my_pe(), omp_get_thread_num());

std::atomic<int> count;
std::atomic<bool> sense;

std::atomic<int> phaseCount;
std::atomic<int> phaseBucketize;

static int numThreads;
static int numWorkers;

int *sort() {
    char *const phases_str = std::getenv("ISX_PHASES");

    int phases = phases_str != nullptr ? atoi(phases_str) : 1;

    KEY_TYPE *allKeys;
    KEY_TYPE *keys[phases];
    uint64_t keysSizes[phases];

    int *localBucketSizes[phases];
    int **bucketSizesPerThread[phases];
    int *localBucketOffsets[phases];
    int *sendOffsets[phases];
    KEY_TYPE *bucketedKeys[phases];
    int64_t *writeOffsets[phases];

    // Out array
    int *myCounts;


    int *perChunkCounts;

    receiveOffset = 0;

    phaseCount.store(0);
    phaseBucketize.store(0);

    int masterId;
#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        std::atomic<bool> localSense;
        localSense.store(true);

        #pragma omp master
        {
            masterId = omp_get_thread_num();
            numThreads = omp_get_num_threads();
            numWorkers = numThreads - 1;
            count.store(numThreads - 1);
            sense.store(true);
        }

        #pragma omp barrier
        makeInput(NUM_KEYS_PER_PE, allKeys, localSense);

        // Initialize key sizes
        uint64_t remainingKeys = NUM_KEYS_PER_PE;
        uint64_t keysPerPhase = NUM_KEYS_PER_PE / phases;

        keys[0] = allKeys;
        keysSizes[0] = keysPerPhase;
        for (int i = 1; i < phases; i++) {
            if (remainingKeys >= keysPerPhase) {
                keysSizes[i] = keysPerPhase;
                remainingKeys -= keysPerPhase;
            } else {
                keysSizes[i] = remainingKeys;
                remainingKeys = 0;
            }
            keys[i] = keys[i - 1] + keysSizes[i];
        }

        for (int i = 0; i < phases; i++) {
            if (threadId != masterId) {
                countLocalBucketSizes(keys[i], keysSizes[i], localBucketSizes[i], bucketSizesPerThread[i], localSense);
                if (threadId == 1) {
                    phaseCount++;
                }

                computeLocalBucketOffsets(localBucketSizes[i], localBucketOffsets[i], sendOffsets[i], localSense);
                bucketizeLocalKeys(keys[i], keysSizes[i], localBucketOffsets[i], bucketSizesPerThread[i],
                                   bucketedKeys[i], localSense);
                if (threadId == 1) {
                    phaseBucketize++;
                }
            }

            if (threadId == masterId) {
                while (phaseCount.load() <= i);
                getWriteOffsets(localBucketSizes[i], writeOffsets[i]);
                while (phaseBucketize.load() <= i);
                sendKeys(sendOffsets[i], localBucketSizes[i], writeOffsets[i], bucketedKeys[i]);
            }
        }


        #pragma omp barrier
        myBucketSize = receiveOffset;
        countLocalKeys(myBucket, myBucketSize, myCounts, perChunkCounts, localSense);
    }

    shmem_quiet();
    shmem_barrier_all();

    free(allKeys);

    for (int i = 0; i < phases; i++) {
        free(writeOffsets[i]);
        free(localBucketSizes[i]);
        free(bucketSizesPerThread[i]);
        for (int j = 0; j < numWorkers; j++) {
            // TODO: fix bug
            // free(bucketSizesPerThread[i][j]);
        }
        free(localBucketOffsets[i]);
        free(sendOffsets[i]);
        free(bucketedKeys[i]);
    }

    return myCounts;
}

void makeInput(uint64_t keysLength, KEY_TYPE *&keys, std::atomic<bool> &localSense) {
#pragma omp single
    {
        keys = (KEY_TYPE *) malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
        assert(keys);
    }

    int threadNum = omp_get_thread_num();
    pcg32_random_t rng = seedMyRank(threadNum, numThreads);

    uint64_t begin, end;
    splitRange(omp_get_thread_num(), numThreads, (uint64_t) 0, keysLength, begin, end);
    for (uint64_t i = begin; i < end; i++) {
        keys[i] = pcg32_boundedrand_r(&rng, (uint32_t) MAX_KEY_VAL);
    }
    #pragma omp barrier
}

void countLocalBucketSizes(KEY_TYPE const *const keys, uint64_t keysLength,
                           int *&localBucketSizes, int **&bucketSizesPerThread, std::atomic<bool> &localSense) {
    int workerNum = omp_get_thread_num() - 1;
#pragma omp single nowait
    {
        localBucketSizes = (int *) calloc(NUM_BUCKETS, sizeof(int));
        bucketSizesPerThread = (int **) malloc(numWorkers * sizeof(int *));
    }
    barrierWait(numWorkers, count, sense, localSense);
    int *bucketSizes = (int *) calloc(NUM_BUCKETS > INTS_PER_CACHE_LINE ?
                                      NUM_BUCKETS : INTS_PER_CACHE_LINE, sizeof(int));

    bucketSizesPerThread[workerNum] = bucketSizes;

    uint64_t begin, end;
    splitRange(workerNum, numWorkers, (uint64_t) 0, keysLength, begin, end);

    for (uint64_t i = begin; i < end; i++) {
        const uint32_t bucketIndex = (const uint32_t) (keys[i] / BUCKET_WIDTH);
        bucketSizes[bucketIndex]++;
    }
    barrierWait(numWorkers, count, sense, localSense);

    #pragma omp single nowait
    {
        for (int i = 0; i < numWorkers; i++) {
            int *threadBucketSizes = bucketSizesPerThread[i];

            for (uint64_t j = 0; j < NUM_BUCKETS; j++) {
                localBucketSizes[j] += threadBucketSizes[j];
            }
        }
    }
    barrierWait(numWorkers, count, sense, localSense);
}

void computeLocalBucketOffsets(int const *const localBucketSizes, int *&localBucketOffsets,
                               int *&sendOffsets, std::atomic<bool> &localSense) {
#pragma omp single nowait
    {
        localBucketOffsets = (int *) malloc(NUM_BUCKETS * sizeof(int));
        sendOffsets = (int *) malloc(NUM_BUCKETS * sizeof(int));

        localBucketOffsets[0] = 0;
        sendOffsets[0] = 0;
        int temp = 0;
        for (uint64_t i = 1; i < NUM_BUCKETS; i++) {
            temp = localBucketOffsets[i - 1] + localBucketSizes[i - 1];
            localBucketOffsets[i] = temp;
            sendOffsets[i] = temp;
        }
    }
    barrierWait(numWorkers, count, sense, localSense);
}

void bucketizeLocalKeys(KEY_TYPE const *const keys, uint64_t keysLength, int *const localBucketOffsets,
                        int **bucketSizesPerThread, KEY_TYPE *&bucketedKeys, std::atomic<bool> &localSense) {
    int workerNum = omp_get_thread_num() - 1;

    int *chunkBucketOffsets;

#pragma omp single nowait
    {
        bucketedKeys = (KEY_TYPE *) malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
    }
    barrierWait(numWorkers, count, sense, localSense);

    chunkBucketOffsets = (int *) malloc(NUM_BUCKETS * numWorkers * sizeof(int));

    for (uint64_t b = 0; b < NUM_BUCKETS; b++) {
        chunkBucketOffsets[b * numWorkers + 0] = localBucketOffsets[b];
        for (int w = 1; w < numWorkers; w++) {
            chunkBucketOffsets[b * numWorkers + w] =
                    chunkBucketOffsets[b * numWorkers + w - 1] + bucketSizesPerThread[w - 1][b];
        }
    }

    int *tmp = (int *) calloc(NUM_BUCKETS > INTS_PER_CACHE_LINE ?
                              NUM_BUCKETS : INTS_PER_CACHE_LINE, sizeof(int));

    for (unsigned i = 0; i < NUM_BUCKETS; i++) {
        tmp[i] = chunkBucketOffsets[i * numWorkers + workerNum];
    }

    uint64_t begin, end;
    splitRange(workerNum, numWorkers, (uint64_t) 0, keysLength, begin, end);
    for (uint64_t i = begin; i < end; i++) {
        const KEY_TYPE key = keys[i];
        const uint32_t bucket_index = (const uint32_t) (key / BUCKET_WIDTH);
        uint32_t index = (uint32_t) tmp[bucket_index]++;
        assert(index < NUM_KEYS_PER_PE);
        bucketedKeys[index] = key;
    }
    barrierWait(numWorkers, count, sense, localSense);

    free(tmp);
    free(chunkBucketOffsets);
}

void countLocalKeys(KEY_TYPE const *const bucketedKeys, uint64_t keysLength, int *&myCounts, int *&perChunkCounts,
                    std::atomic<bool> &localSense) {
    uint64_t myMinKey = shmem_my_pe() * BUCKET_WIDTH;
    int threadNum = omp_get_thread_num();

#pragma omp single
    {
        myCounts = (int *) calloc(BUCKET_WIDTH, sizeof(int));
        perChunkCounts = (int *) calloc(numThreads * BUCKET_WIDTH, sizeof(int));
    }

    int *counts = perChunkCounts + (threadNum * BUCKET_WIDTH);

    uint64_t begin, end;
    splitRange(omp_get_thread_num(), numThreads, (uint64_t) 0, keysLength, begin, end);
    for (uint64_t i = begin; i < end; i++) {
        const unsigned int key_index = bucketedKeys[i] - myMinKey;
        assert(key_index < BUCKET_WIDTH);
        counts[key_index]++;
    }
    #pragma omp barrier

    for (int c = 0; c < numThreads; c++) {
        uint64_t _begin, _end;
        splitRange(omp_get_thread_num(), numThreads, (uint64_t) 0, BUCKET_WIDTH, _begin, _end);
        for (uint64_t i = _begin; i < _end; i++) {
            int *thread_counts = perChunkCounts + (c * BUCKET_WIDTH);
            myCounts[i] += thread_counts[i];
        }
        #pragma omp barrier
    }


#pragma omp single nowait
    {
        free(perChunkCounts);
    }

    #pragma omp barrier
}

void getWriteOffsets(int const *const localBucketSizes, int64_t *&writeOffsets) {
    const int myRank = shmem_my_pe();
    writeOffsets = (int64_t *) malloc(sizeof(int64_t) * NUM_PES);

    for (int i = 0; i < NUM_PES; i++) {
        const int targetRank = (const int) ((myRank + i) % NUM_PES);
        writeOffsets[targetRank] = shmem_longlong_fadd(&receiveOffset, localBucketSizes[targetRank], targetRank);
        assert(writeOffsets[targetRank] + localBucketSizes[targetRank] <= KEY_BUFFER_SIZE);
    }
}

void sendKeys(int const *const bucketOffsets, int const *const bucketSizes, int64_t *writeOffsets,
              int const *const bucketedKeys) {
    const int myRank = shmem_my_pe();

    memcpy(&myBucket[writeOffsets[myRank]],
           &bucketedKeys[bucketOffsets[myRank]],
           bucketSizes[myRank] * sizeof(KEY_TYPE));

    for (int i = 0; i < NUM_PES; i++) {
        const int targetRank = (const int) ((myRank + i) % NUM_PES);
        shmem_int_put(&(myBucket[writeOffsets[targetRank]]),
                      &(bucketedKeys[bucketOffsets[targetRank]]),
                      bucketSizes[targetRank],
                      targetRank);

        writeOffsets[targetRank] += bucketSizes[targetRank];
    }

    shmem_barrier_all();
}
