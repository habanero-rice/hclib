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
 * DESC: Fork a bunch of asyncs in a top-level loop
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.h"

#define H3 1024
#define H2 512
#define H1 16
#define T3 33
#define T2 217
#define T1 7


//user written code
void forasync_fct3(void * argv,int idx1,int idx2,int idx3) {
    
    int *ran=(int *)argv;
    //printf("%d_%d_%d ",idx1,idx2,ran[idx1*32+idx2]);
    assert(ran[idx1*H2*H3+idx2*H3+idx3] == -1);
    ran[idx1*H2*H3+idx2*H3+idx3] = idx1*H2*H3+idx2*H3+idx3;
}

void init_ran(int *ran, int size) {
    while (size > 0) {
        ran[size-1] = -1;
        size--;
    }
}

int main (int argc, char ** argv) {
    printf("Call Init\n");
    hclib_init(&argc, argv);
    int i = 0;
    int *ran=(int *)malloc(H1*H2*H3*sizeof(int));
    // This is ok to have these on stack because this
    // code is alive until the end of the program.

    init_ran(ran, H1*H2*H3);
    loop_domain_t loop0 = {0,H1,1,T1};
    loop_domain_t loop1 = {0,H2,1,T2};
    loop_domain_t loop2 = {0,H3,1,T3};
    loop_domain_t loop[3] = {loop0, loop1, loop2};

    hclib_start_finish();
    hclib_forasync(forasync_fct3,(void*)(ran),NULL, NULL,NULL,3,loop,FORASYNC_MODE_RECURSIVE);
    hclib_end_finish();

    printf("Call Finalize\n");
    hclib_finalize();
    printf("Check results: ");
    i=0;
    while(i < H1*H2*H3) {
        assert(ran[i] == i);
        i++;
    }
    printf("OK\n");
    return 0;
}
