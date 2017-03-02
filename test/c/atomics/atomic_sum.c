/* Copyright (c) 2013, Rice University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
3.  Neither the name of Rice University
     nor the names of its contributors may be used to endorse or
     promote products derived from this software without specific
     prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/**
 * DESC: Simple atomic sum test.
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.h"
#include "hclib_atomic.h"

#define N_ASYNC 100

static void sum(void *atomic, void *user_data) {
    int *atomic_int = (int *)atomic;
    size_t incr = (size_t)user_data;
    *atomic_int += incr;
}

static void sum_gather(void *a, void *b, void *user_data) {
    int *out = (int *)a;
    int *in = (int *)b;
    *out += *in;
}

static void init_int(void *atomic, void *user_data) {
    *((int *)atomic) = 0;
}

void async_fct(void *arg) {
    hclib_atomic_t *atomic = (hclib_atomic_t *)arg;

    hclib_atomic_update(atomic, sum, (void *)1);
}

void entrypoint(void *arg) {
    int i;

    hclib_atomic_t *atomic = hclib_atomic_create(sizeof(int), init_int, NULL);

    hclib_start_finish();

    for (i = 0; i < N_ASYNC; i++) {
        hclib_async(async_fct, atomic, NULL, 0, NULL);
    }

    hclib_end_finish();

    int *final_val = (int *)hclib_atomic_gather(atomic, sum_gather, NULL);

    assert(*final_val == N_ASYNC);

    printf("Passed\n");
}

int main (int argc, char ** argv) {
    char const *deps[] = { "system" };
    hclib_launch(entrypoint, NULL, deps, 1);
    fprintf(stderr, "Finished launch\n");
    return 0;
}
