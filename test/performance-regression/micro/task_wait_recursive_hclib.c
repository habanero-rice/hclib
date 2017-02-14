#include "hclib.h"
#include "task_wait_recursive.h"

/**
 * Calculate micro-statistics on how quickly we can wait on recursive task
 * completion.
 */

void recursive_task(void *arg) {
    const size_t depth = (size_t)arg;
    if (depth < N_RECURSIVE_TASK_WAITS) {
        hclib_start_finish();
        {
            hclib_async(recursive_task, (void *)(depth + 1), NULL, 0, NULL);
        }
        hclib_end_finish();
    }
}

void entrypoint(void *arg) {
    int nworkers = hclib_get_num_workers();

    printf("Using %d HClib workers\n", nworkers);

    const unsigned long long start_time = hclib_current_time_ns();
    recursive_task((void *)0);
    const unsigned long long end_time = hclib_current_time_ns();

    printf("METRIC task_wait_recursive %d %f\n", N_RECURSIVE_TASK_WAITS,
            (double)N_RECURSIVE_TASK_WAITS / ((double)(end_time -
                    start_time) / 1000.0));
}

int main(int argc, char **argv) {
    hclib_launch(entrypoint, NULL, NULL, 0);
}
