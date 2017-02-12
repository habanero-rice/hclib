#include "hclib.h"

#include <stdio.h>
#include "future_spawn.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

void *empty_task(void *arg) {
    /*
     * Unfortunately need to put something here to compare against OpenMP tasks,
     * otherwise some OpenMP compilers will make the task a no-op.
     */
    int incr;
    incr = incr + 1;
    return NULL;
}

void entrypoint(void *arg) {
    int nworkers = hclib_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    hclib_start_finish();
    {
        const unsigned long long start_time = hclib_current_time_ns();

        hclib_future_t *prev = NULL;
        int nlaunched = 0;
        do {
            prev = hclib_async_future(empty_task, NULL, &prev, 1, NULL);
            nlaunched++;
        } while (nlaunched < NFUTURES);

        const unsigned long long end_time = hclib_current_time_ns();
        printf("Generated futures at a rate of %f futures per us\n",
                (double)NFUTURES / ((double)(end_time - start_time) / 1000.0));
    }
    hclib_end_finish();

    const unsigned long long start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        hclib_future_t *prev = NULL;
        int nlaunched = 0;
        do {
            prev = hclib_async_future(empty_task, NULL, &prev, 1, NULL);
            nlaunched++;
        } while (nlaunched < NFUTURES);
    }
    hclib_end_finish();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("Scheduled futures at a rate of %f futures per us\n",
            (double)NFUTURES / ((double)(end_time - start_time) / 1000.0));
}

int main(int argc, char **argv) {
    int i;

    hclib_launch(entrypoint, NULL, NULL, 0);
}
