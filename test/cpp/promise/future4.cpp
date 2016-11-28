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
 * DESC: Test if future->wait() preserves finish scope
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "hclib_cpp.h"

int main(int argc, char **argv) {
    bool *ran = new bool{false};
    assert(ran);

    hclib::launch([=]() {
        hclib::promise_t<void> *promise = new hclib::promise_t<void>();

        for (int i = 0; i < 30; i++) {
            hclib::async([=]() { usleep(10000); });
        }

        HCLIB_FINISH {
            hclib::async([=]() {
                usleep(500000);
                *ran = true;
            });

            hclib::async([=]() {
                usleep(20000);
                promise->put();
            });

            usleep(10000);

            promise->future().wait();
        }
        assert(*ran);
    });

    delete ran;
    printf("OK\n");

    return 0;
}