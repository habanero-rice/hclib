#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

unsigned long long current_time_ns() {
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (((unsigned long long)t.tv_nsec)) + s;
}

int main(int argc, char **argv) {
    int r, i;
    shmem_init();
 
    if (argc != 3) {
        fprintf(stderr, "usage: %s send-buf-size repeats\n", argv[0]);
        exit(1);
    }

    const int N = atoi(argv[1]);
    const int repeats = atoi(argv[2]);

    int rank, size;
    rank = shmem_my_pe();
    size = shmem_n_pes();

    int *send_buf = (int *)shmem_malloc(N * sizeof(int));
    int *recv_buf = (int *)shmem_malloc(size * N * sizeof(int));
    assert(send_buf && recv_buf);
    long *pSync = (long *)shmem_malloc(repeats * SHMEM_COLLECT_SYNC_SIZE * sizeof(long));
    assert(pSync);
    for (i = 0; i < repeats * SHMEM_COLLECT_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_barrier_all();
    const unsigned long long start = current_time_ns();

    for (r = 0; r < repeats; r++) {
        shmem_fcollect32(recv_buf, send_buf, N, 0, 0, size,
                pSync + (r * SHMEM_COLLECT_SYNC_SIZE));
    }
    shmem_barrier_all();
    const unsigned long long elapsed = current_time_ns() - start;
    if (rank == 0) printf("%llu ns elapsed\n", elapsed);


    shmem_finalize();
    return 0;
}
