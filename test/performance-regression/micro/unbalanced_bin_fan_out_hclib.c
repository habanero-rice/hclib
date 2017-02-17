#include "hclib.h"

#include <stdio.h>
#include "unbalanced_bin_fan_out.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

static size_t pack(const int depth, const int branch) {
    size_t next = branch;
    next = next << 32; 
    next = next | depth;
    return next;
}

static int unpack_depth(const size_t packed) {
    return (packed & 0xffffffff);
}

static int unpack_branch(const size_t packed) {
    int branch = packed >> 32; 
    return branch;
}

static void recurse(void *arg) {
    const int depth = unpack_depth((size_t)arg);
    const int branch = unpack_branch((size_t)arg);
    const int depth_limit = branch * BIN_FAN_OUT_DEPTH_MULTIPLIER;

    if (depth < depth_limit) {
        size_t next = pack(depth + 1, branch);

        hclib_async(recurse, (void *)next, NULL, 0, NULL);
        hclib_async(recurse, (void *)next, NULL, 0, NULL);
    }
}

static void entrypoint(void *arg) {
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    unsigned ntasks = 0;
    const unsigned long long start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        int i;
        for (i = 0; i < N_BRANCHES; i++) {
            ntasks += (1 << (i * BIN_FAN_OUT_DEPTH_MULTIPLIER));
            size_t next = pack(0, i);

            recurse((void *)next);
        }
    }
    hclib_end_finish();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC unbalanced_bin_fan_out %d|%d %.20f\n", N_BRANCHES,
            BIN_FAN_OUT_DEPTH_MULTIPLIER,
            (double)ntasks / ((double)(end_time - start_time) / 1000.0));
}

int main(int argc, char **argv) {
    hclib_launch(entrypoint, NULL, NULL, 0);
}
