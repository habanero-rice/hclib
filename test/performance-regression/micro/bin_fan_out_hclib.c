#include "hclib.h"

#include <stdio.h>
#include "bin_fan_out.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

void recurse(void *arg) {
    const size_t depth = (size_t)arg;

    if (depth < BIN_FAN_OUT_DEPTH) {
        hclib_async(recurse, (void *)(depth + 1), NULL, 0, NULL);
        hclib_async(recurse, (void *)(depth + 1), NULL, 0, NULL);
    }
}

void entrypoint(void *arg) {
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    const unsigned long long start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        recurse(0);
    }
    hclib_end_finish();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("Did binary fan out of depth %d in HClib in %llu ns\n",
            BIN_FAN_OUT_DEPTH, end_time - start_time);
}

int main(int argc, char **argv) {
    int i;

    hclib_launch(entrypoint, NULL, NULL, 0);
}
