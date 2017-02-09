#include "hclib.h"
#include "task_wait_flat.h"

#include <omp.h>
#include <stdio.h>

/**
 * Calculate micro-statistics on how quickly we can wait on flat task
 * completion.
 */
int main(int argc, char **argv) {
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
        for (i = 0; i < N_FLAT_TASK_WAITS; i++) {

            int incr = 0;
#pragma omp task firstprivate(incr)
            {
                incr = incr + 1;
            }

#pragma omp taskwait
        }

        const unsigned long long end_time = hclib_current_time_ns();
        printf("Synchronized on OpenMP tasks at a rate of %f tasks per ns\n",
                (double)N_FLAT_TASK_WAITS / (double)(end_time - start_time));
    }
}
