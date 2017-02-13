#include "hclib.h"

#include <omp.h>
#include <stdio.h>
#include "unbalanced_bin_fan_out.h"

void recurse(const int depth, const int branch) {
    const int depth_limit = branch * BIN_FAN_OUT_DEPTH_MULTIPLIER;

    if (depth < depth_limit) {
#pragma omp task default(none) firstprivate(depth, branch)
        {
            recurse(depth + 1, branch);
        }

#pragma omp task default(none) firstprivate(depth, branch)
        {
            recurse(depth + 1, branch);
        }
    }
}

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
        int i;

        const unsigned long long start_time = hclib_current_time_ns();

#pragma omp taskgroup
        for (i = 0; i < N_BRANCHES; i++) {
            recurse(0, i);
        }

        const unsigned long long end_time = hclib_current_time_ns();
        printf("OpenMP did unbalanced binary fan out w/ %d branches and depth "
                "multiplier %d in %llu ns\n", N_BRANCHES,
                BIN_FAN_OUT_DEPTH_MULTIPLIER, end_time - start_time);
    }
}
