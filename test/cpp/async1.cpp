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

#define NB_ASYNC 127

int ran[NB_ASYNC];

void init_ran(int *ran, int size) {
    while (size >= 0) {
        ran[size] = -1;
        size--;
    }
}

int main (int argc, char ** argv) {
    printf("Call Init\n");
    hclib::launch([]() {
        // This is ok to have these on stack because this
        // code is alive until the end of the program.
        hclib::finish([]() {
            int i = 0;
            int indices [NB_ASYNC];
            init_ran(ran, NB_ASYNC);

            while (i < NB_ASYNC) {
                indices[i] = i;
                hclib::async([=](){
                    int index = indices[i];
                    assert(ran[index] == -1);
                    ran[index] = index;
                });
                i++;
            }
        });
    });
    printf("Check results: ");
    int i = 0;
    while(i < NB_ASYNC) {
        assert(ran[i] == i);
        i++;
    }
    printf("OK\n");
    return 0;
}
