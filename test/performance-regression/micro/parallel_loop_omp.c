#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "parallel_loop.h"

int main(int argc, char **argv) {
    int i;

    int nthreads;
#pragma omp parallel
#pragma omp master
    {
        nthreads = omp_get_num_threads();
    }
    printf("Using %d OpenMP threads\n", nthreads);

#pragma omp parallel
#pragma omp master
    {
        int i;

        const unsigned long long start_time = hclib_current_time_ns();
#pragma omp taskloop private(i)
        for (i = 0; i < PARALLEL_LOOP_RANGE; i++) {
        }
        const unsigned long long end_time = hclib_current_time_ns();
        printf("OMP parallel loop ran at %f iters/ms\n",
                (double)PARALLEL_LOOP_RANGE / ((double)(end_time -
                        start_time) / 1000.0));
    }

    return 0;
}
