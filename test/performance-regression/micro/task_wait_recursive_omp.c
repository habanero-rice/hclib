#include "hclib.h"
#include "task_wait_recursive.h"

#include <omp.h>
#include <stdio.h>

static void recursive_wait_task(const size_t depth) {
    if (depth < N_RECURSIVE_TASK_WAITS) {
#pragma omp task default(none) firstprivate(depth)
        {
            recursive_wait_task(depth + 1);
        }
#pragma omp taskwait
    }
}

static void recursive_group_task(const size_t depth) {
    if (depth < N_RECURSIVE_TASK_WAITS) {
#pragma omp taskgroup
        {
#pragma omp task default(none) firstprivate(depth)
            {
                recursive_group_task(depth + 1);
            }
        }
    }
}

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
        const unsigned long long group_start_time = hclib_current_time_ns();
        recursive_group_task(0);
        const unsigned long long group_end_time = hclib_current_time_ns();
        printf("METRIC task_wait_recursive %d %f\n", N_RECURSIVE_TASK_WAITS,
                (double)N_RECURSIVE_TASK_WAITS /
                ((double)(group_end_time - group_start_time) / 1000.0));

        const unsigned long long wait_start_time = hclib_current_time_ns();
        recursive_wait_task(0);
        const unsigned long long wait_end_time = hclib_current_time_ns();
        printf("METRIC task_wait_recursive %d %f\n", N_RECURSIVE_TASK_WAITS,
                (double)N_RECURSIVE_TASK_WAITS /
                ((double)(wait_end_time - wait_start_time) / 1000.0));
    }
}
