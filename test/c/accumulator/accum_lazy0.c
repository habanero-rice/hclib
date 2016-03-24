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
 * DESC: Recursive accumulator
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.h"

int ran = 0;

void async_fct(void * arg) {
    printf("Running Async\n");
    ran = 1;
}

void accum_create_n(accum_t ** accums, int n) {
    int i = 0;
    while(i < n) {
        accums[i] = accum_create_int(ACCUM_OP_PLUS, ACCUM_MODE_LAZY, 0);
        i++;
    }
}

void accum_destroy_n(accum_t ** accums, int n) {
    int i = 0;
    while(i < n) {
        accum_destroy(accums[i]);
        i++;
    }
}

void accum_print_n(accum_t ** accums, int n) {
    int i = 0;
    while(i < n) {
        int res = accum_get_int(accums[i]);
        printf("Hello[%d] = %d\n", i, res);
        i++;
    }
}

int main (int argc, char ** argv) {
    hclib_init(&argc, argv);
    int n = 10;
    accum_t * accums_s[n];
    accum_t ** accums =  (accum_t **) accums_s;
    accum_create_n(accums, n);
    start_finish();
    accum_register(accums, n);
    accum_put_int(accums[3], 2);
    accum_put_int(accums[4], 2);
    accum_put_int(accums[5], 2);     
    end_finish();
    accum_print_n(accums, n);
    accum_destroy_n(accums, n);
    hclib_finalize();
    return 0;
}
