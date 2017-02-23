#include "hclib.h"

#include <stdio.h>
#include "prod_cons.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

void empty_task(void *arg) {
    /*
     * Unfortunately need to put something here to compare against OpenMP tasks,
     * otherwise some OpenMP compilers will make the task a no-op.
     */
    int incr = 0;
    incr = incr + 1;
}

void entrypoint(void *arg) {
    int s;
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    hclib_promise_t *signals = (hclib_promise_t *)malloc(
            PROD_CONS_MSGS * sizeof(hclib_promise_t));
    assert(signals);
    for (s = 0; s < PROD_CONS_MSGS; s++) {
        hclib_promise_init(signals + s);
    }

    const unsigned long long start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        int i;
        for (i = 0; i < PROD_CONS_MSGS; i++) {
            hclib_future_t *fut = hclib_get_future_for_promise(signals + i);
            hclib_async(empty_task, NULL, &fut, 1, NULL);
        }

        for (i = 0; i < PROD_CONS_MSGS; i++) {
            hclib_promise_put(signals + i, NULL);
        }
    }
    hclib_end_finish();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC producer_consumer %d %.20f\n", PROD_CONS_MSGS,
            (double)PROD_CONS_MSGS / ((double)(end_time - start_time) /
                1000.0));
}

int main(int argc, char **argv) {
    hclib_launch(entrypoint, NULL, NULL, 0);
    return 0;
}
