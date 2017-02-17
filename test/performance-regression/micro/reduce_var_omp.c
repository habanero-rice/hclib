#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "reduce_var.h"

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

        unsigned sum = 0;
        const unsigned long long start_time = hclib_current_time_ns();
#pragma omp parallel for default(none) private(i) reduction(+:sum)
        for (i = 0; i < NREDUCERS; i++) {
            sum += 1;
        }
        const unsigned long long end_time = hclib_current_time_ns();

        assert(sum == NREDUCERS);

        printf("METRIC recursive_reduction %d %.20f\n", NREDUCERS,
                (double)NREDUCERS / ((double)(end_time -
                        start_time) / 1000.0));
        printf("METRIC flat_reduction %d %.20f\n", NREDUCERS,
                (double)NREDUCERS / ((double)(end_time -
                        start_time) / 1000.0));
    }

    return 0;
}
