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
// FINISH SCOPE MACRO

#define FINISH GEN_FINISH_SCOPE(MACRO_CONCAT(_hcGenVar_, __COUNTER__))
#define GEN_FINISH_SCOPE(V) for (int V=(hclib_start_finish(), 1); V; hclib_end_finish(), --V)
#define MACRO_CONCAT(a, b) DO_MACRO_CONCAT(a, b)
#define DO_MACRO_CONCAT(a, b) a ## b


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
// DDT VERSION

#if 1
#define MY_ESCAPE_PROP NO_PROP
#else
#define MY_ESCAPE_PROP ESCAPING_ASYNC
#endif

typedef struct {
    int n;
    long resval;
    hclib_promise_t *res;
    hclib_promise_t *subres[3];
} FibDDtArgs;

static FibDDtArgs *setup_fib_ddt_args(int n) {
    FibDDtArgs *args = new FibDDtArgs;
    args->n = n;
    args->res = hclib_promise_create();
    args->subres[2] = NULL;
    return args;
}

static void free_ddt_args(FibDDtArgs *args) {
    hclib_promise_free(args->res);
    delete args;
}

static inline void *promise_get(hclib_promise_t *p) {
    return hclib_future_get(hclib_get_future_for_promise(p));
}

static inline hclib_future_t **ps2fs(hclib_promise_t **ps) {
    static_assert(offsetof(hclib_promise_t, future) == 0,
            "can cast promise ptr directly to future ptr");
    return (hclib_future_t **)ps;
}

void fib_ddt_res(void * raw_args) {
    FibDDtArgs *args = (FibDDtArgs*) raw_args;
    FibDDtArgs *lhs = (FibDDtArgs*) promise_get(args->subres[0]);
    FibDDtArgs *rhs = (FibDDtArgs*) promise_get(args->subres[1]);
    args->resval =  lhs->resval + rhs->resval;
    hclib_promise_put(args->res, args);
    // cleanup
    free_ddt_args(lhs);
    free_ddt_args(rhs);
}

void fib_ddt(void * raw_args) {
    FibDDtArgs *args = (FibDDtArgs*) raw_args;
    if (args->n < 2) {
        args->resval = args->n;
        hclib_promise_put(args->res, args);
    }
    else {
        FibDDtArgs *lhsArgs = setup_fib_ddt_args(args->n - 1);
        FibDDtArgs *rhsArgs = setup_fib_ddt_args(args->n - 2);
        args->subres[0] = lhsArgs->res;
        args->subres[1] = rhsArgs->res;
        // sub-computation asyncs
        hclib_async(fib_ddt, lhsArgs, NO_FUTURE, NO_PHASER, ANY_PLACE, MY_ESCAPE_PROP);
        hclib_async(fib_ddt, rhsArgs, NO_FUTURE, NO_PHASER, ANY_PLACE, MY_ESCAPE_PROP);
        // async-await for sub-results
        hclib_async(fib_ddt_res, args, ps2fs(args->subres), NO_PHASER, ANY_PLACE, MY_ESCAPE_PROP);
    }
}

void fib_ddt_root_await(void * raw_args) { /* no-op */ }


////////////////////////////////////
// DRIVER

double t_start, t_end;

void taskMain(void *raw_args) {
    FibDDtArgs *args = (FibDDtArgs*) raw_args;
    t_start = get_seconds();
    hclib_async(fib_ddt, args, NO_FUTURE, NO_PHASER, ANY_PLACE, MY_ESCAPE_PROP);
    args->subres[0] = NULL; // null terminate after res
    HCLIB_FINISH { // top-level sync before measuring end-time
        hclib_async(fib_ddt_root_await, args, ps2fs(&args->res), NO_PHASER, ANY_PLACE, NO_PROP);
    }
    t_end = get_seconds();
}

int main(int argc, char ** argv) {
    // set up
    int n = argv[1] ? atoi(argv[1]) : 30;
    const long fn = fib_iter(n);
    const long fnp1 = fib_iter(n+1);
    long answer;
    // run
    FibDDtArgs *args = setup_fib_ddt_args(n);
    hclib_launch(taskMain, args);
    // finish
    answer = args->resval;
    print_throughput(fnp1, t_end - t_start);
    // check results
    printf("Fib(%d) = %ld = %ld\n", n, fn, answer);
    assert(answer == fn);
}
