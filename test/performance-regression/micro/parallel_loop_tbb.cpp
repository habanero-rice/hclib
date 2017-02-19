#include "hclib.h"

#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

#include <stdio.h>
#include "parallel_loop.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    int nthreads = tbb::task_scheduler_init::default_num_threads();
    printf("Using %d TBB threads\n", nthreads);

    const unsigned long long start_time = hclib_current_time_ns();
    tbb::parallel_for(size_t(0), size_t(PARALLEL_LOOP_RANGE), [&] (size_t i) {
            });
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC flat_parallel_iters %d %.20f\n", PARALLEL_LOOP_RANGE,
            (double)PARALLEL_LOOP_RANGE / ((double)(end_time -
                    start_time) / 1000.0));
    return 0;
}
