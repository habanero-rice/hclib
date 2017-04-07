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
#include <unistd.h>

#include "hclib_cpp.h"

#define H1 256
#define T1 33

//user written code
void init_ran(int *ran, int size) {
    while (size > 0) {
        ran[size-1] = -1;
        size--;
    }
}

int main (int argc, char ** argv) {
    printf("Call Init\n");
    int *ran= new int[H1];
    assert(ran);

    hclib::launch([=]() {
            int i = 0;
            // This is ok to have these on stack because this
            // code is alive until the end of the program.

            init_ran(ran, H1);
            loop_domain_t loop = {0, H1, 1, T1};

            hclib::future_t<void> *event = hclib::forasync1D_future(&loop,
                    [=](int idx) {
                        usleep(100000);
                        assert(ran[idx] == -1);
                        ran[idx] = idx;
                        printf("finished %d / %d\n", idx, H1);
                    }, FORASYNC_MODE_FLAT);

            event->wait();
            printf("Call Finalize\n");
        });

    printf("Check results: ");
    int i = 0;
    while(i < H1) {
        assert(ran[i] == i);
        i++;
    }
    delete[] ran;
    printf("OK\n");
    return 0;
}
