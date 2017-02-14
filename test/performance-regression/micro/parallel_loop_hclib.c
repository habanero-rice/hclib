#include "hclib.h"

#include <stdio.h>
#include "parallel_loop.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

void loop_body(void *arg, int index) {
}

void entrypoint(void *arg) {
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    hclib_loop_domain_t domain;
    domain.low = 0;
    domain.high = PARALLEL_LOOP_RANGE;
    domain.stride = 1;
    domain.tile = -1;

    const unsigned long long recursive_start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        hclib_forasync(loop_body, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
    }
    hclib_end_finish();
    const unsigned long long recursive_end_time = hclib_current_time_ns();
    printf("METRIC recursive_parallel_iters %d %f\n", PARALLEL_LOOP_RANGE,
            (double)PARALLEL_LOOP_RANGE / ((double)(recursive_end_time -
                    recursive_start_time) / 1000.0));

    const unsigned long long flat_start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        hclib_forasync(loop_body, NULL, 1, &domain, FORASYNC_MODE_FLAT);
    }
    hclib_end_finish();
    const unsigned long long flat_end_time = hclib_current_time_ns();
    printf("METRIC flat_parallel_iters %d %f\n", PARALLEL_LOOP_RANGE,
            (double)PARALLEL_LOOP_RANGE / ((double)(flat_end_time -
                    flat_start_time) / 1000.0));
}

int main(int argc, char **argv) {
    hclib_launch(entrypoint, NULL, NULL, 0);
}
