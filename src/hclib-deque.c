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

#include "hclib-internal.h"
#include "hclib-atomics.h"

/*
 * push an entry onto the tail of the deque
 */
int deque_push(deque_t *deq, hclib_task_t *entry) {
    int tail = _hclib_atomic_load_relaxed(&deq->tail);
    int head = _hclib_atomic_load_relaxed(&deq->head);
    int size = tail - head;
    if (size == INIT_DEQUE_CAPACITY) { /* deque looks full */
        /* may not grow the deque if some interleaving steal occur */
        // std::cout<<getenv("PMI_RANK") <<": Deque full for worker-"<<current_ws()->id << std::endl;
        // HASSERT("DEQUE full, increase deque's size " && 0);
        return 0;
    }
    int n = tail % INIT_DEQUE_CAPACITY;
    deq->data[n] = entry;
    //@ ATOMIC: this should be a RELEASE to ensure synchronization
    _hclib_atomic_inc_release(&deq->tail);
    return 1;
}

/*
 * the steal protocol
 */
hclib_task_t *deque_steal(deque_t *deq) {
    int head;
    /* Cannot read deq->data[head] here
     * Can happen that head=tail=0, then the owner of the deq pushes
     * a new task when stealer is here in the code, resulting in head=0, tail=1
     * All other checks down-below will be valid, but the old value of the buffer head
     * would be returned by the steal rather than the new pushed value.
     */
    int tail;

    head = _hclib_atomic_load_relaxed(&deq->head);
    // ATOMIC: load acquire
    // We want all the writes from the producing thread to read the task data
    // and we're using the tail as the synchronization variable
    tail = _hclib_atomic_load_acquire(&deq->tail);
    if ((tail - head) <= 0) {
        return NULL;
    }

    hclib_task_t *t = deq->data[head % INIT_DEQUE_CAPACITY];
    /* compete with other thieves and possibly the owner (if the size == 1) */
    if (_hclib_atomic_cas_acq_rel(&deq->head, head, head + 1)) { /* competing */
        return t;
    }
    return NULL;
}

/*
 * pop the task out of the deque from the tail
 */
hclib_task_t *deque_pop(deque_t *deq) {
    int tail = _hclib_atomic_dec_relaxed(&deq->tail);
    int head = _hclib_atomic_load_relaxed(&deq->head);

    int size = tail - head;
    if (size < 0) {
        _hclib_atomic_store_relaxed(&deq->tail, head);
        return NULL;
    }
    hclib_task_t *t = deq->data[tail % INIT_DEQUE_CAPACITY];

    if (size > 0) {
        return t;
    }

    /* now the deque appears empty */
    /* I need to compete with the thieves for the last task */
    //@- if (!hc_cas(&deq->head, head, head + 1)) {
    if (!_hclib_atomic_cas_acq_rel(&deq->head, head, head + 1)) {
        t = NULL;
    }

    _hclib_atomic_inc_relaxed(&deq->tail);

    return t;
}

