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

/**
 * DESC: Lazy accumulator
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.h"

void async_fct(void * arg) {
    accum_t * accum = (accum_t *) arg;
    accum_put_int(accum, 1);
}

#define N 1000

int main (int argc, char ** argv) {
    hclib_init(&argc, argv);
    accum_t * accum = accum_create_int(ACCUM_OP_PLUS, ACCUM_MODE_LAZY, 0);
    start_finish();
    accum_register(&accum, 1);
    // spawn asyncs all contributing to the accumulator
    int i;
    for(i=0;i<N;i++) {
        async(async_fct, accum, NULL, NULL, NO_PROP);
    }
    end_finish();
    int res = accum_get_int(accum);
    printf("Accumulator value %d\n", res);
    assert(res == N);
    hclib_finalize();
    return 0;
}
