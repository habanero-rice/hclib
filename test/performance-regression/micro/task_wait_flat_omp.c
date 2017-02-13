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

        const unsigned long long group_start_time = hclib_current_time_ns();
        for (i = 0; i < N_FLAT_TASK_WAITS; i++) {

            int incr = 0;
#pragma omp taskgroup
            {
#pragma omp task default(none) firstprivate(incr)
                {
                    incr = incr + 1;
                }
            }
        }
        const unsigned long long group_end_time = hclib_current_time_ns();
        printf("Synchronized on grouped OpenMP tasks at a rate of %f tasks per "
                "us\n", (double)N_FLAT_TASK_WAITS / ((double)(group_end_time -
                        group_start_time) / 1000.0));

        const unsigned long long wait_start_time = hclib_current_time_ns();
        for (i = 0; i < N_FLAT_TASK_WAITS; i++) {

            int incr = 0;
#pragma omp task default(none) firstprivate(incr)
            {
                incr = incr + 1;
            }
#pragma omp taskwait
        }
        const unsigned long long wait_end_time = hclib_current_time_ns();
        printf("Synchronized on flat OpenMP tasks at a rate of %f tasks per "
                "us\n", (double)N_FLAT_TASK_WAITS / ((double)(wait_end_time -
                        wait_start_time) / 1000.0));
    }
}
