/* Copyright (c) 2015, Rice University

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
 * DESC: Fork a bunch of asyncs in a top-level loop
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "hclib_cpp.h"

#define H1 256
#define T1 33

void init_ran(int *ran, int size) {
    while (size > 0) {
        ran[size-1] = -1;
        size--;
    }
}

int main (int argc, char ** argv) {
    printf("Call Init\n");
    int *ran=(int *)malloc(H1*sizeof(int));
    assert(ran);

    hclib::launch(&argc, argv, [=]() {
            int i = 0;
            // This is ok to have these on stack because this
            // code is alive until the end of the program.

            init_ran(ran, H1);

            hclib::ddf_t *event = hclib::nonblocking_finish([=]() {
                loop_domain_t loop = {0, H1, 1, T1};
                hclib::forasync1D(&loop, [=](int idx) {
                        sleep(1);
                        assert(ran[idx] == -1);
                        ran[idx] = idx;
                        printf("finished %d / %d\n", idx, H1);
                    }, FORASYNC_MODE_FLAT);
            });

            hclib::ddf_wait(event);
            printf("Call Finalize\n");
        });

    printf("Check results: ");
    int i = 0;
    while(i < H1) {
        assert(ran[i] == i);
        i++;
    }
    printf("OK\n");
    return 0;
}
