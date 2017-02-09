#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "future_spawn.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    int i;

    int nthreads;
#pragma omp parallel
#pragma omp master
    {
        nthreads = omp_get_num_threads();
    }
    printf("Using %d OpenMP threads\n", nthreads);

#pragma omp parallel
#pragma omp master
    {
        const unsigned long long spawn_start_time = hclib_current_time_ns();

        int dep_arr[3];

        int incr = 0;

        int nlaunched = 0;
        do {
            if (nlaunched == 0) {
#pragma omp task firstprivate(incr) depend(out:dep_arr[1])
                {
                    incr = incr + 1;
                }
            } else {
#pragma omp task firstprivate(incr) depend(in:dep_arr[nlaunched]) \
                depend(out:dep_arr[nlaunched + 1])
                {
                    incr = incr + 1;
                }
            }

            nlaunched++;
        } while (nlaunched < NFUTURES);

        const unsigned long long spawn_end_time = hclib_current_time_ns();
        printf("Generated futures at a rate of %f futures per ns\n",
                (double)NFUTURES / (double)(spawn_end_time - spawn_start_time));

#pragma omp taskwait

        const unsigned long long schedule_start_time = hclib_current_time_ns();
        nlaunched = 0;
        do {
            if (nlaunched == 0) {
#pragma omp task firstprivate(incr) depend(out:dep_arr[1])
                {
                    incr = incr + 1;
                }
            } else {
#pragma omp task firstprivate(incr) depend(in:dep_arr[nlaunched]) \
                depend(out:dep_arr[nlaunched + 1])
                {
                    incr = incr + 1;
                }
            }

            nlaunched++;
        } while (nlaunched < NFUTURES);
#pragma omp taskwait

        const unsigned long long schedule_end_time = hclib_current_time_ns();
        printf("Scheduled futures at a rate of %f futures per ns\n",
                (double)NFUTURES / (double)(schedule_end_time -
                    schedule_start_time));
    }
}
