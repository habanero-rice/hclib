/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hc.h"

////////////////////////////////////
// TIMING HELPER FUNCTIONS

#include <sys/time.h>

static double get_seconds() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + ((double)tv.tv_usec / 1000000);
}

void print_throughput(long op_count, double elapsed_seconds) {
    long throughput = op_count / elapsed_seconds;
    printf("Throughput (op/s): %ld\n", throughput);
    printf("Execution time:    %f\n", elapsed_seconds);
}

////////////////////////////////////
// ITERATIVE VERSION

long fib_iter(int n) {
    int i, x, y;
    for (i = 0, x = 1, y = 0; i <= n; i++) {
        int t = x;
        x = y;
        y += t;
    }
    return x;
}

////////////////////////////////////
// ASYNC-FINISH VERSION

long fib(int n) {
    if (n < 2) {
        return n;
    } else {
        long lhs, rhs;
        finish {
            async IN(n) OUT(lhs) { lhs = fib(n - 1); }
            async IN(n) OUT(rhs) { rhs = fib(n - 2); }
        }
        return lhs + rhs;
    }
}

////////////////////////////////////
// DRIVER

int main(int argc, char** argv) {
    // set up
    int n = 30;
    long fn = fib_iter(n);
    long fnp1 = fib_iter(n + 1);
    long answer;
    double t_start, t_end;
    // run
    t_start = get_seconds();
    finish { answer = fib(n); }
    // finish
    t_end = get_seconds();
    print_throughput(fnp1, t_end - t_start);
    // check results
    printf("Fib(%d) = %ld = %ld\n", n, fn, answer);
    assert(answer == fn);
}
