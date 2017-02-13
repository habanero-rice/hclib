#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "reduce_var.h"

int main(int argc, char **argv) {
    int i;

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
#pragma omp taskloop default(none) private(i) reduction(+:sum)
        for (i = 0; i < NREDUCERS; i++) {
            sum += 1;
        }
        const unsigned long long end_time = hclib_current_time_ns();

        assert(sum == NREDUCERS);

        printf("OpenMP reductions at a rate of %f reducers/ms\n",
                (double)NREDUCERS / ((double)(end_time -
                        start_time) / 1000.0));
    }

    return 0;
}
