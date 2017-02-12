#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "fan_out_and_in.h"

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
        int initial_dep[0];
        int dep_arr[0][FAN_OUT_AND_IN];

        int incr = 0;

        const unsigned long long start_time = hclib_current_time_ns();
#pragma omp task depend(out:initial_dep[0])
        {
        }

        int nlaunched = 0;
        for (i = 0; i < FAN_OUT_AND_IN; i++) {
#pragma omp task firstprivate(incr) depend(in:initial_dep[0]) \
            depend(out:dep_arr[0][i])
            {
                incr = incr + 1;
            }
        }

        int wavefront = 1;
        int nfutures = FAN_OUT_AND_IN;
        while (nfutures > 1) {
            int next_nfutures = 0;

            for (i = 0; i < nfutures; i += MAX_NUM_WAITS) {
                int this_n_futures = nfutures - i;
                if (this_n_futures > MAX_NUM_WAITS) this_n_futures = MAX_NUM_WAITS;

#pragma omp task depend(in:dep_arr[wavefront - 1][i:this_n_futures]) \
                depend(out:dep_arr[wavefront][i]) firstprivate(incr)
                {
                    incr = incr + 1;
                }
            }
            nfutures = next_nfutures;
            wavefront++;
        }

#pragma omp taskwait
        const unsigned long long end_time = hclib_current_time_ns();
        printf("Handled %d-wide OpenMP fan out in %llu ns\n", FAN_OUT_AND_IN,
                end_time - start_time);
    }
}
