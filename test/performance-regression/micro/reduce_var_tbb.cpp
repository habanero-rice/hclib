#include "hclib.h"

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

#include <stdio.h>
#include "reduce_var.h"

struct Sum {
    float value;

    Sum() : value(0) {
    }

    Sum( Sum& s, tbb::split ) {
        value = 0;
    }

    void operator()( const tbb::blocked_range<int*>& r ) {
        int temp = value;
        for( int* a=r.begin(); a!=r.end(); ++a ) {
            temp += *a;
        }
        value = temp;
    }

    void join( Sum& rhs ) {value += rhs.value;}
};


/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    int nthreads = tbb::task_scheduler_init::default_num_threads();
    printf("Using %d TBB threads\n", nthreads);

    int *vec = (int *)malloc(NREDUCERS * sizeof(int));
    assert(vec);
    int i;
    for (i = 0; i < NREDUCERS; i++) {
        vec[i] = 1;
    }

    const unsigned long long start_time = hclib_current_time_ns();
    Sum total;
    tbb::parallel_reduce(tbb::blocked_range<int*>(vec, vec + NREDUCERS),
            total);
    const unsigned long long end_time = hclib_current_time_ns();

    printf("METRIC recursive_reduction %d %.20f\n", NREDUCERS,
            (double)NREDUCERS / ((double)(end_time -
                    start_time) / 1000.0));
    printf("METRIC flat_reduction %d %.20f\n", NREDUCERS,
            (double)NREDUCERS / ((double)(end_time -
                    start_time) / 1000.0));

    return 0;
}
