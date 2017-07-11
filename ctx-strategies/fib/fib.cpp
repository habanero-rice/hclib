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

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <hclib.h>
#include <assert.h>

#include "hclib.hpp"

////////////////////////////////////
// TIMING HELPER FUNCTIONS

#include <sys/time.h>

static double get_seconds() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + ((double) tv.tv_usec / 1000000);
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
    for (i=0, x=1, y=0; i<=n; i++) {
        int t = x;
        x = y;
        y += t;
    }
    return x;
}


////////////////////////////////////
// ASYNC-FINISH VERSION

typedef struct {
    int n;
    long res;
} FibArgs;

void fib(void * raw_args) {
    FibArgs *args = (FibArgs*) raw_args;
    if (args->n < 2) {
        args->res = args->n;
    }
    else {
        FibArgs lhsArgs = { args->n - 1, 0 };
        FibArgs rhsArgs = { args->n - 2, 0 };
        HCLIB_FINISH {
            hclib_async(fib, &lhsArgs, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
            hclib_async(fib, &rhsArgs, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
        }
        args->res = lhsArgs.res + rhsArgs.res;
    }
}


////////////////////////////////////
// DRIVER

double t_start, t_end;

void taskMain(void *raw_args) {
    t_start = get_seconds();
    fib(raw_args);
    t_end = get_seconds();
}

int main(int argc, char ** argv) {
    // set up
    int n = argv[1] ? atoi(argv[1]) : 35;
    const long fn = fib_iter(n);
    const long fnp1 = fib_iter(n+1);
    long answer;
    // run
    FibArgs args = { n, 0 };
    hclib_launch(taskMain, &args);
    // finish
    answer = args.res;
    print_throughput(fnp1, t_end - t_start);
    // check results
    printf("Fib(%d) = %ld = %ld\n", n, fn, answer);
    assert(answer == fn);
}
