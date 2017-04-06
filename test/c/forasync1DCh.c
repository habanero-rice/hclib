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
 * DESC: Fork a bunch of asyncs in a top-level loop
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.h"

#define H1 1024
#define T1 33


//user written code
void forasync_fct1(void *argv, int idx) {
    int *ran = (int *)argv;
    assert(ran[idx] == -1);
    ran[idx] = idx;
}

void init_ran(int *ran, int size) {
    while (size > 0) {
        ran[size-1] = -1;
        size--;
    }
}

void entrypoint(void *arg) {
    int *ran = (int *)arg;
    // This is ok to have these on stack because this
    // code is alive until the end of the program.

    init_ran(ran, H1);
    loop_domain_t loop = {0, H1, 1, T1};
    hclib_start_finish();
    hclib_forasync(forasync_fct1, (void*)ran, NULL, 1, &loop,
            FORASYNC_MODE_FLAT);
    hclib_end_finish();

    printf("Call Finalize\n");
}

int main (int argc, char ** argv) {
    printf("Call Init\n");
    int *ran=(int *)malloc(H1*sizeof(int));
    assert(ran);

    hclib_launch(entrypoint, ran);

    printf("Check results: ");
    int i = 0;
    while(i < H1) {
        assert(ran[i] == i);
        i++;
    }
    printf("OK\n");
    return 0;
}
