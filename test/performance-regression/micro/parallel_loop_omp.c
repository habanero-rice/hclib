#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "parallel_loop.h"

int main(int argc, char **argv) {
    int nthreads;
#pragma omp parallel default(none) shared(nthreads)
#pragma omp master
    {
        nthreads = omp_get_num_threads();
    }
    printf("Using %d OpenMP threads\n", nthreads);

#pragma omp parallel default(none)
#pragma omp master
    {
        int i;

        const unsigned long long start_time = hclib_current_time_ns();
#pragma omp taskloop default(none) private(i)
        for (i = 0; i < PARALLEL_LOOP_RANGE; i++) {
        }
        const unsigned long long end_time = hclib_current_time_ns();

        printf("METRIC recursive_parallel_iters %d %f\n",
                PARALLEL_LOOP_RANGE,
                (double)PARALLEL_LOOP_RANGE / ((double)(end_time -
                        start_time) / 1000.0));
        printf("METRIC flat_parallel_iters %d %f\n",
                PARALLEL_LOOP_RANGE,
                (double)PARALLEL_LOOP_RANGE / ((double)(end_time -
                        start_time) / 1000.0));
    }

    return 0;
}
