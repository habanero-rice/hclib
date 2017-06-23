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

#ifndef HCLIB_DEQUE_H_
#define HCLIB_DEQUE_H_

#include "hclib-task.h"
#include "hclib-atomics.h"

/****************************************************/
/* DEQUE API                                        */
/****************************************************/

#define INIT_DEQUE_CAPACITY (1<<18)

typedef struct deque_t {
    _Atomic int head;
    _Atomic int tail;
    hclib_task_t* data[INIT_DEQUE_CAPACITY];
} deque_t;

int deque_push(deque_t *deq, hclib_task_t *entry);
hclib_task_t* deque_pop(deque_t *deq);
hclib_task_t* deque_steal(deque_t *deq);

#endif /* HCLIB_DEQUE_H_ */
