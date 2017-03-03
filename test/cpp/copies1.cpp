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
 * DESC: Counting lambda copies when using forasync
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

#define H3 1024
#define H2 512
#define H1 16
#define T3 33
#define T2 217
#define T1 7

void init_ran(int *ran, int size) {
    while (size > 0) {
        ran[size-1] = -1;
        size--;
    }
}

#include <atomic>
std::atomic<int> globalCopies;
std::atomic<int> globalMoves;

struct CopyCounter {
    int copies;
    int moves;

    CopyCounter(): copies(0), moves(0) {
        printf("Created new counter object.\n");
    }

    CopyCounter(const CopyCounter &other):
        copies(other.copies+1), moves(other.moves) {
            globalCopies++;
        }

    CopyCounter(const CopyCounter &&other):
        copies(other.copies), moves(other.moves+1) {
            globalMoves++;
        }
};



int main (int argc, char ** argv) {
    printf("Call Init\n");
    int *ran=(int *)malloc(H1*H2*H3*sizeof(int));

    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [=] {
        // This is ok to have these on stack because this
        // code is alive until the end of the program.

        init_ran(ran, H1*H2*H3);
        hclib::finish([=]() {
            CopyCounter k {};
            hclib::loop_domain_3d *loop = new hclib::loop_domain_3d(0, H1, T1,
                0, H2, T2, 0, H3, T3);
            hclib::forasync3D(loop, [=](int idx1, int idx2, int idx3) {
                        assert(ran[idx1*H2*H3+idx2*H3+idx3] == -1);
                        ran[idx1*H2*H3+idx2*H3+idx3] = idx1*H2*H3+idx2*H3+idx3;
                    },
                    false, FORASYNC_MODE_RECURSIVE);
        });
    });

    printf("Check results: ");
    int i = 0;
    while(i < H1*H2*H3) {
        assert(ran[i] == i);
        i++;
    }
    free(ran);
    printf("Counted %d copies, %d moves...\n",
            int(globalCopies), int(globalMoves));
    assert(globalCopies + globalMoves <= 2);
    printf("OK\n");
    return 0;
}
