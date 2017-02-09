#include <mpi.h>
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
    int r;
    MPI_Init(&argc, &argv);

    if (argc != 3) {
        fprintf(stderr, "usage: %s send-buf-size repeats\n", argv[0]);
        exit(1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = atoi(argv[1]);
    const int repeats = atoi(argv[2]);
    int *send_buf = (int *)malloc(N * sizeof(int));
    int *recv_buf = (int *)malloc(size * N * sizeof(int));
    assert(send_buf && recv_buf);

    MPI_Barrier(MPI_COMM_WORLD);
    const unsigned long long start = current_time_ns();

    for (r = 0; r < repeats; r++) {
        MPI_Allgather(send_buf, N, MPI_INT, recv_buf, N, MPI_INT,
                MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const unsigned long long elapsed = current_time_ns() - start;
    if (rank == 0) printf("%llu ns elapsed\n", elapsed);

    MPI_Finalize();
    return 0;
}
