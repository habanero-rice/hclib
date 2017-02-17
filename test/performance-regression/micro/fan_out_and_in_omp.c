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
        int initial_dep[1];
        int dep_arr[1][FAN_OUT_AND_IN];
        dep_arr[0][0] = dep_arr[0][0]; // To disable unused variable warnings
        initial_dep[0] = initial_dep[0]; // To disable unused variable warnings

        int incr = 0;

        const unsigned long long start_time = hclib_current_time_ns();
#pragma omp taskgroup
        {

#pragma omp task default(none) depend(out:initial_dep[0])
            {
            }

            int i;
            for (i = 0; i < FAN_OUT_AND_IN; i++) {
#pragma omp task default(none) firstprivate(incr) depend(in:initial_dep[0]) \
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

#pragma omp task default(none) depend(in:dep_arr[wavefront - 1][i:this_n_futures]) \
                    depend(out:dep_arr[wavefront][i]) firstprivate(incr)
                    {
                        incr = incr + 1;
                    }
                }
                nfutures = next_nfutures;
                wavefront++;
            }
        }

        const unsigned long long end_time = hclib_current_time_ns();
        printf("METRIC fan_out_and_in %d %.20f\n", FAN_OUT_AND_IN,
                (double)FAN_OUT_AND_IN / ((double)(end_time - start_time) /
                    1000.0));
    }
}
