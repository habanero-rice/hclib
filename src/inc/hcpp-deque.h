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
 * hcpp-deque.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_DEQUE_H_
#define HCPP_DEQUE_H_

#include "hcpp-task.h"

/****************************************************/
/* DEQUE API                                        */
/****************************************************/

#define INIT_DEQUE_CAPACITY 8096

typedef struct deque_t {
    volatile int head;
    volatile int tail;
    volatile task_t* data[INIT_DEQUE_CAPACITY];
} deque_t;

void deque_init(deque_t * deq, void * initValue);
int deque_push(deque_t* deq, void* entry);
task_t* deque_pop(deque_t * deq);
task_t* deque_steal(deque_t * deq);
void deque_destroy(deque_t* deq);

/****************************************************/
/* Semi Concurrent DEQUE API                        */
/****************************************************/
typedef struct {
    deque_t deque;
    volatile int lock;
} semi_conc_deque_t;

void semi_conc_deque_init(semi_conc_deque_t* deq, void * initValue);
void semi_conc_deque_locked_push(semi_conc_deque_t* deq, void* entry);
task_t* semi_conc_deque_non_locked_pop(semi_conc_deque_t * deq);
void semi_conc_deque_destroy(semi_conc_deque_t * deq);

#endif /* HCPP_DEQUE_H_ */
