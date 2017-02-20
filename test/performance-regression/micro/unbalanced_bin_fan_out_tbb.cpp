#include "hclib.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"

#include <stdio.h>
#include "unbalanced_bin_fan_out.h"

static void recurse(const int depth, const int branch, tbb::task_group *g) {
    const int depth_limit = branch * BIN_FAN_OUT_DEPTH_MULTIPLIER;

    if (depth < depth_limit) {
        g->run([=] {
                recurse(depth + 1, branch, g);
            });

        g->run([=] {
                recurse(depth + 1, branch, g);
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
    int i;
    unsigned ntasks = 0;
    for (i = 0; i < N_BRANCHES; i++) {
        ntasks += (1 << (i * BIN_FAN_OUT_DEPTH_MULTIPLIER));
        recurse(0, i, &g);
    }
    g.wait();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC unbalanced_bin_fan_out %d|%d %.20f\n", N_BRANCHES,
            BIN_FAN_OUT_DEPTH_MULTIPLIER,
            (double)ntasks / ((double)(end_time - start_time) / 1000.0));

    return 0;
}
