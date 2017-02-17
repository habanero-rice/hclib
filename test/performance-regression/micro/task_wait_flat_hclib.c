#include "hclib.h"
#include "task_wait_flat.h"

/**
 * Calculate micro-statistics on how quickly we can wait on flat task
 * completion.
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
    int i;
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    const unsigned long long nb_start_time = hclib_current_time_ns();
    for (i = 0; i < N_FLAT_TASK_WAITS; i++) {
        hclib_start_finish();
        {
            hclib_async_nb(empty_task, NULL, NULL);
        }
        hclib_end_finish();
    }
    const unsigned long long nb_end_time = hclib_current_time_ns();
    printf("METRIC task_wait_flat %d %.20f\n", N_FLAT_TASK_WAITS,
            (double)N_FLAT_TASK_WAITS / ((double)(nb_end_time -
                    nb_start_time) / 1000.0));

    const unsigned long long blocking_start_time = hclib_current_time_ns();
    for (i = 0; i < N_FLAT_TASK_WAITS; i++) {
        hclib_start_finish();
        {
            hclib_async(empty_task, NULL, NULL, 0, NULL);
        }
        hclib_end_finish();
    }
    const unsigned long long blocking_end_time = hclib_current_time_ns();
    printf("METRIC task_wait_flat %d %.20f\n", N_FLAT_TASK_WAITS,
            (double)N_FLAT_TASK_WAITS / ((double)(blocking_end_time -
                    blocking_start_time) / 1000.0));
}

int main(int argc, char **argv) {
    hclib_launch(entrypoint, NULL, NULL, 0);
}
