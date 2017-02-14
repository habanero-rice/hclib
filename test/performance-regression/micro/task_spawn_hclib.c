#include "hclib.h"

#include <stdio.h>
#include "task_spawn.h"

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
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    hclib_start_finish();
    {
        const unsigned long long start_time = hclib_current_time_ns();

        int nlaunched = 0;
        do {
            hclib_async_nb(empty_task, NULL, NULL);

            nlaunched++;
        } while (nlaunched < NTASKS);

        const unsigned long long end_time = hclib_current_time_ns();
        printf("METRIC task_create %d %f\n", NTASKS,
                (double)NTASKS / ((double)(end_time - start_time) / 1000.0));
    }
    hclib_end_finish();

    const unsigned long long start_time = hclib_current_time_ns();
    hclib_start_finish();
    {
        int nlaunched = 0;
        do {
            hclib_async_nb(empty_task, NULL, NULL);

            nlaunched++;
        } while (nlaunched < NTASKS);
    }
    hclib_end_finish();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC task_run %d %f\n", NTASKS,
            (double)NTASKS / ((double)(end_time - start_time) / 1000.0));
}

int main(int argc, char **argv) {
    hclib_launch(entrypoint, NULL, NULL, 0);
}
