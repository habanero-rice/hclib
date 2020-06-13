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

/*
 * hclib-deque.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCLIB_DEQUE_H_
#define HCLIB_DEQUE_H_

#include "hclib-task.h"

/****************************************************/
/* DEQUE API                                        */
/****************************************************/

#define STEAL_CHUNK_SIZE 1

// #define INIT_DEQUE_CAPACITY 16384
#define INIT_DEQUE_CAPACITY 1048576

typedef struct hclib_internal_deque_t {
    /*
     * Head is shared by all threads, both stealers and the thread local to this
     * deque. Head points to the slot containing the oldest created task.
     * Stealing a task implies reading the task pointed to by head and then
     * safely incrementing head.
     */
    volatile int head;

    /*
     * Tail is only manipulated by the thread owning a deque. New tasks are
     * pushed into the slot pointed to by tail, followed by an increment of
     * tail. Local tasks may be acquired by decrementing tail and grabbing the
     * task at the slot pointed to post-decrement.
     */
    volatile int tail;

    volatile hclib_task_t* data[INIT_DEQUE_CAPACITY];
} hclib_internal_deque_t;

void deque_init(hclib_internal_deque_t *deq, void *initValue);
int deque_push(hclib_internal_deque_t *deq, void *entry);
hclib_task_t* deque_pop(hclib_internal_deque_t *deq);
int deque_steal(hclib_internal_deque_t *deq, void **stolen);
void deque_destroy(hclib_internal_deque_t *deq);
unsigned deque_size(hclib_internal_deque_t *deq);

#endif /* HCLIB_DEQUE_H_ */
