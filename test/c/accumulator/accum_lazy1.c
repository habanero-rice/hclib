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
 * DESC: Lazy accumulator
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.h"

void async_fct(void * arg) {
    accum_t * accum = (accum_t *) arg;
    accum_put_int(accum, 1);
}

#define N 1000

int main (int argc, char ** argv) {
    hclib_init(&argc, argv);
    accum_t * accum = accum_create_int(ACCUM_OP_PLUS, ACCUM_MODE_LAZY, 0);
    start_finish();
    accum_register(&accum, 1);
    // spawn asyncs all contributing to the accumulator
    int i;
    for(i=0;i<N;i++) {
        async(async_fct, accum, NULL, NULL, NO_PROP);
    }
    end_finish();
    int res = accum_get_int(accum);
    printf("Accumulator value %d\n", res);
    assert(res == N);
    hclib_finalize();
    return 0;
}
