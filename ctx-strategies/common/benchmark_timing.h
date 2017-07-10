#ifndef BENCHMARK_TIMING_H_
#define BENCHMARK_TIMING_H_

#include <stdio.h>
#include <sys/time.h>
#include <time.h>

static inline double current_sec(void) {
    struct timeval a;
    gettimeofday(&a, 0);
    return (a.tv_sec * 1000000.0 + a.tv_usec) / 1000000.0;
}

static inline void print_total_time(double start, double end) {
    printf("The computation took %f seconds\r\n", end - start);
}

#endif  /* BENCHMARK_TIMING_H_ */
