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
 * DESC: top-level finish for a bunch of asyncs
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

#define NB_ASYNC 127

int ran[NB_ASYNC];

void init_ran(int *ran, int size) {
    while (size >= 0) {
        ran[size] = -1;
        size--;
    }
}

void assert_done(int start, int end) {
    while(start < end) {
        assert(ran[start] == start);
        start++;
    }
}
int main (int argc, char ** argv) {
    printf("Call Init\n");
    int mid = NB_ASYNC/2;
    hclib::launch([&mid]() {
        int i = 0;
        int indices [NB_ASYNC];

        hclib::finish([=, &i, &indices]() {

            // This is ok to have these on stack because this
            // code is alive until the end of the program.
            init_ran(ran, NB_ASYNC);
            printf("Go over [%d:%d]\n", i, mid);
            while(i < mid) {
                indices[i] = i;
                hclib::async([=]() { int index = indices[i];
                    assert(ran[index] == -1); ran[index] = index; });
                i++;
            }
        });

        printf("Midway\n");
        assert_done(0, mid);
        printf("Go over [%d:%d]\n", i, NB_ASYNC);
        while(i < NB_ASYNC) {
            indices[i] = i;
            //Note: Forcefully pass the address we want to write to as a void **
            hclib::async([=]() { int index = indices[i];
                assert(ran[index] == -1); ran[index] = index; });
            i++;
        }
    });
    printf("Check results: ");
    assert_done(mid, NB_ASYNC);
    printf("OK\n");
    return 0;
}
