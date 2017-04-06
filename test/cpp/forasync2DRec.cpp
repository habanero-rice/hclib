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

#include "hclib_cpp.h"

#define H1 1024
#define H2 512
#define T1 33
#define T2 217

void init_ran(int *ran, int size) {
    while (size > 0) {
        ran[size-1] = -1;
        size--;
    }
}

int main (int argc, char ** argv) {
    printf("Call Init\n");
    int *ran=(int *)malloc(H1*H2*sizeof(int));
    hclib::launch([=]() {
        int i = 0;
        // This is ok to have these on stack because this
        // code is alive until the end of the program.

        init_ran(ran, H1*H2);
        hclib::finish([=]() {
            loop_domain_t loop0 = {0,H1,1,T1};
            loop_domain_t loop1 = {0,H2,1,T2};
            loop_domain_t loop[2] = {loop0, loop1};
            hclib::forasync2D(loop, [=](int idx1, int idx2) {
                    assert(ran[idx1*H2+idx2] == -1);
                    ran[idx1*H2+idx2] = idx1*H2+idx2; }, FORASYNC_MODE_RECURSIVE);
        });
    });

    printf("Check results: ");
    int i = 0;
    while(i < H1*H2) {
        assert(ran[i] == i);
        i++;
    }
    free(ran);
    printf("OK\n");
    return 0;
}
