#include "hclib.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"

#include <stdio.h>
#include "task_wait_recursive.h"

static void recursive_task(const size_t depth) {
    if (depth < N_RECURSIVE_TASK_WAITS) {
        tbb::task_group g;
        g.run([=] {
                recursive_task(depth + 1);
            });
        g.wait();
    }
}

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
    recursive_task(0);
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC task_wait_recursive %d %.20f\n", N_RECURSIVE_TASK_WAITS,
            (double)N_RECURSIVE_TASK_WAITS /
            ((double)(end_time - start_time) / 1000.0));

    return 0;
}
