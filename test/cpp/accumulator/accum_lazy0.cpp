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
 * DESC: Recursive accumulator
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

int ran = 0;

void async_fct(void * arg) {
    printf("Running Async\n");
    ran = 1;
}

void accum_create_n(accum_t ** accums, int n) {
    int i = 0;
    while(i < n) {
        accums[i] = accum_create_int(ACCUM_OP_PLUS, ACCUM_MODE_LAZY, 0);
        i++;
    }
}

void accum_destroy_n(accum_t ** accums, int n) {
    int i = 0;
    while(i < n) {
        accum_destroy(accums[i]);
        i++;
    }
}

void accum_print_n(accum_t ** accums, int n) {
    int i = 0;
    while(i < n) {
        int res = accum_get_int(accums[i]);
        printf("Hello[%d] = %d\n", i, res);
        i++;
    }
}

int main (int argc, char ** argv) {
    hclib_init(&argc, argv);
    int n = 10;
    accum_t * accums_s[n];
    accum_t ** accums =  (accum_t **) accums_s;
    accum_create_n(accums, n);
    start_finish();
    accum_register(accums, n);
    accum_put_int(accums[3], 2);
    accum_put_int(accums[4], 2);
    accum_put_int(accums[5], 2);     
    end_finish();
    accum_print_n(accums, n);
    accum_destroy_n(accums, n);
    hclib_finalize();
    return 0;
}
