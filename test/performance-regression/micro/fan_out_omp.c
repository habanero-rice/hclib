#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "fan_out.h"

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
        int dep_arr[1];

        const unsigned long long start_time = hclib_current_time_ns();

        int incr = 0;

        int nlaunched = 0;
        for (i = 0; i < FAN_OUT; i++) {
#pragma omp task firstprivate(incr) depend(in:dep_arr[0])
            {
                incr = incr + 1;
            }
        }
        const unsigned long long end_time = hclib_current_time_ns();
        printf("Handled %d-wide OpenMP fan out in %llu ns\n", FAN_OUT,
                end_time - start_time);
    }
}
