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
 * DESC: recursive calls with finish (malloc-based)
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.hpp"

#define NB_ASYNC 127

int * ran = NULL;

void assert_done(int start, int end) {
    while(start < end) {
        assert(ran[start] == start);
        start++;
    }
}

void init_ran(int *ran, int size) {
    size--;
    while (size >= 0) {
        ran[size] = -1;
        size--;
    }
}

void spawn_async(volatile int * indices, int i) {
    if (i < NB_ASYNC) {
        hclib::finish([=]() {
            indices[i] = i;
            hclib::async([=]() { int idx = indices[i]; assert(ran[idx] == -1);
                ran[idx] = idx; });
            spawn_async(indices, i+1);
        });
        assert_done(i, i+1);
    }
}

int main (int argc, char ** argv) {
    printf("Call Init\n");
    hclib::launch([]() {
        volatile int * indices = (int *) malloc(sizeof(int)*NB_ASYNC);
        ran = (int *) malloc(sizeof(int)*NB_ASYNC);
        init_ran(ran, NB_ASYNC);
        hclib::finish([=]() {
            spawn_async(indices, 0);
        });
        free((void *)indices);
    });
    printf("Check results: ");
    assert_done(0, NB_ASYNC);
    printf("OK\n");
    return 0;
}
