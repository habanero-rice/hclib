//
// Created by Srdan Milakovic on 4/20/17.
//

#include "verification.hpp"
#include <shmem.h>
#include <cstdio>
#include <cinttypes>

#include "util.hpp"

int verifyResults(int const *const my_local_key_counts, const KEY_TYPE *const my_local_keys) {
    shmem_barrier_all();
    int error = 0;

    const int my_rank = shmem_my_pe();
    // const int my_rank = ::shmem_my_pe();

    const int my_min_key = my_rank * BUCKET_WIDTH;
    const int my_max_key = (my_rank + 1) * BUCKET_WIDTH - 1;

#ifdef ISX_PROFILING
    unsigned long long start = current_time_ns();
#endif
    // Verify all keys are within bucket boundaries
    for (long long int i = 0; i < myBucketSize; ++i) {
        const int key = my_local_keys[i];
        if ((key < my_min_key) || (key > my_max_key)) {
            printf("Rank %d Failed Verification!\n", my_rank);
            printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
            error = 1;
        }
    }

#ifdef ISX_PROFILING
    unsigned long long end = current_time_ns();
  if (shmem_my_pe() == 0)
  printf("Verifying took %llu ns\n", end - start);
#endif

    // Verify the sum of the key population equals the expected bucket size
    long long int bucket_size_test = 0;
    for (uint64_t i = 0; i < BUCKET_WIDTH; ++i) {
        bucket_size_test += my_local_key_counts[i];
    }
    if (bucket_size_test != myBucketSize) {
        printf("Rank %d Failed Verification!\n", my_rank);
        printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, myBucketSize);
        error = 1;
    }

    // Verify the final number of keys equals the initial number of keys
    static long long int total_num_keys = 0;
    shmem_longlong_sum_to_all(&total_num_keys, &myBucketSize, 1, 0, 0, NUM_PES, llWrk, pSync);
    shmem_barrier_all();

    if (total_num_keys != (long long int) (NUM_KEYS_PER_PE * NUM_PES)) {
        if (my_rank == ROOT_PE) {
            printf("Verification Failed!\n");
            printf("Actual total number of keys: %lld Expected %" PRIu64 "\n",
                   total_num_keys, NUM_KEYS_PER_PE * NUM_PES );
            error = 1;
        }
    }

    return error;
}
