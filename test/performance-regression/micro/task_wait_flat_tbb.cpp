#include "hclib.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"

#include <stdio.h>
#include "task_wait_flat.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    int nthreads = tbb::task_scheduler_init::default_num_threads();
    printf("Using %d TBB threads\n", nthreads);

    tbb::task_group g;

    const unsigned long long start_time = hclib_current_time_ns();
    int i;
    for (i = 0; i < N_FLAT_TASK_WAITS; i++) {
        g.run([=] {
                int incr = 0;
                incr = incr + 1;
            });
        g.wait();
    }
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC task_wait_flat %d %.20f\n", N_FLAT_TASK_WAITS,
            (double)N_FLAT_TASK_WAITS / ((double)(end_time -
                    start_time) / 1000.0));

    return 0;
}
