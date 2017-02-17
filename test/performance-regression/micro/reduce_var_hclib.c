#include "hclib.h"
#include "hclib_atomic.h"

#include <stdio.h>
#include "reduce_var.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

static void sum_gather(void *a, void *b, void *user_data) {
    int *out = (int *)a;
    int *in = (int *)b;
    *out += *in;
}

static void sum(void *atomic, void *user_data) {
    int *atomic_int = (int *)atomic;
    size_t incr = (size_t)user_data;
    *atomic_int += incr;
}

static void init_int(void *atomic, void *user_data) {
    *((int *)atomic) = 0;
}

void loop_body(void *arg, int index) {
    hclib_atomic_t *atomic = (hclib_atomic_t *)arg;
    hclib_atomic_update(atomic, sum, (void *)1);
}

void entrypoint(void *arg) {
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    hclib_loop_domain_t domain;
    domain.low = 0;
    domain.high = NREDUCERS;
    domain.stride = 1;
    domain.tile = -1;

    hclib_atomic_t *recursive_atomic = hclib_atomic_create(sizeof(int), init_int, NULL);
    hclib_atomic_t *flat_atomic = hclib_atomic_create(sizeof(int), init_int, NULL);

    const unsigned long long recursive_start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        hclib_forasync(loop_body, recursive_atomic, 1, &domain, FORASYNC_MODE_RECURSIVE);
    }
    hclib_end_finish();
    int *final_val = (int *)hclib_atomic_gather(recursive_atomic, sum_gather, NULL);
    const unsigned long long recursive_end_time = hclib_current_time_ns();

    assert(*final_val == NREDUCERS);

    printf("METRIC recursive_reduction %d %.20f\n", NREDUCERS,
            (double)NREDUCERS / ((double)(recursive_end_time -
                    recursive_start_time) / 1000.0));

    const unsigned long long flat_start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        hclib_forasync(loop_body, flat_atomic, 1, &domain, FORASYNC_MODE_FLAT);
    }
    hclib_end_finish();
    final_val = (int *)hclib_atomic_gather(flat_atomic, sum_gather, NULL);
    const unsigned long long flat_end_time = hclib_current_time_ns();

    assert(*final_val == NREDUCERS);

    printf("METRIC flat_reduction %d %.20f\n", NREDUCERS,
            (double)NREDUCERS / ((double)(flat_end_time -
                    flat_start_time) / 1000.0));
}

int main(int argc, char **argv) {
    hclib_launch(entrypoint, NULL, NULL, 0);
}
