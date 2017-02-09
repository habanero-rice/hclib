#include "hclib.h"
#include "task_wait_recursive.h"

#include <omp.h>
#include <stdio.h>

void recursive_task(const size_t depth) {
    if (depth < N_RECURSIVE_TASK_WAITS) {
#pragma omp task firstprivate(depth)
        {
            recursive_task(depth + 1);
        }
#pragma omp taskwait
    }
}

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
        recursive_task(0);
        const unsigned long long end_time = hclib_current_time_ns();
        printf("Synchronized on recursive tasks at a rate of %f task-waits per "
                "us\n", (double)N_RECURSIVE_TASK_WAITS / ((double)(end_time -
                    start_time) / 1000.0));
    }
}
