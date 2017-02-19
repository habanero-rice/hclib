#include "hclib.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"

#include <stdio.h>
#include "bin_fan_out.h"

static void recurse(const int depth, tbb::task_group *g) {
    if (depth < BIN_FAN_OUT_DEPTH) {
        g->run([=] {
                recurse(depth + 1, g);
            });

        g->run([=] {
                recurse(depth + 1, g);
            });
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

    tbb::task_group g;

    const unsigned long long start_time = hclib_current_time_ns();
    recurse(0, &g);
    g.wait();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC bin_fan_out %d %.20f\n", BIN_FAN_OUT_DEPTH,
            (double)(1 << BIN_FAN_OUT_DEPTH) /
            ((double)(end_time - start_time) / 1000.0));

    return 0;
}
