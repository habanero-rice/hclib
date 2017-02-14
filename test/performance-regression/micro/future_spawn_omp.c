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
    int nthreads;
#pragma omp parallel default(none) shared(nthreads)
#pragma omp master
    {
        nthreads = omp_get_num_threads();
    }
    printf("Using %d OpenMP threads\n", nthreads);

#pragma omp parallel default(none)
#pragma omp master
    {

#pragma omp taskgroup
        {
            int dep_arr[3];
            int incr = 0;
            int nlaunched = 0;
            dep_arr[1] = dep_arr[0]; // To disable unused variable warnings

            const unsigned long long spawn_start_time = hclib_current_time_ns();

            do {
                if (nlaunched == 0) {
#pragma omp task default(none) firstprivate(incr) depend(out:dep_arr[1])
                    {
                        incr = incr + 1;
                    }
                } else {
#pragma omp task default(none) firstprivate(incr) depend(in:dep_arr[nlaunched]) \
                    depend(out:dep_arr[nlaunched + 1])
                    {
                        incr = incr + 1;
                    }
                }

                nlaunched++;
            } while (nlaunched < NFUTURES);

            const unsigned long long spawn_end_time = hclib_current_time_ns();
            printf("METRIC future_create %d %f\n", NFUTURES,
                    (double)NFUTURES / ((double)(spawn_end_time - spawn_start_time) / 1000.0));
        }

        const unsigned long long schedule_start_time = hclib_current_time_ns();

#pragma omp taskgroup
        {
            int dep_arr[3];
            int incr = 0;
            int nlaunched = 0;
            dep_arr[1] = dep_arr[0]; // To disable unused variable warnings

            do {
                if (nlaunched == 0) {
#pragma omp task default(none) firstprivate(incr) depend(out:dep_arr[1])
                    {
                        incr = incr + 1;
                    }
                } else {
#pragma omp task default(none) firstprivate(incr) \
                    depend(in:dep_arr[nlaunched]) \
                    depend(out:dep_arr[nlaunched + 1])
                    {
                        incr = incr + 1;
                    }
                }

                nlaunched++;
            } while (nlaunched < NFUTURES);
        }

        const unsigned long long schedule_end_time = hclib_current_time_ns();
        printf("METRIC future_run %d %f\n", NFUTURES,
                (double)NFUTURES / ((double)(schedule_end_time - schedule_start_time) / 1000.0));
    }
}
