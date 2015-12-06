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
 * hcpp-deque.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include "hcpp-internal.h"
#include "hcpp-atomics.h"

void dequeInit(deque_t * deq, void * init_value) {
	deq->head = 0;
	deq->tail = 0;
}

/*
 * push an entry onto the tail of the deque
 */
int dequePush(deque_t* deq, void* entry) {
	int size = deq->tail - deq->head;
	if (size == INIT_DEQUE_CAPACITY) { /* deque looks full */
		/* may not grow the deque if some interleaving steal occur */
		// std::cout<<getenv("PMI_RANK") <<": Deque full for worker-"<<current_ws()->id << std::endl;
		// HASSERT("DEQUE full, increase deque's size " && 0);
		return 0;
	}
	int n = (deq->tail) % INIT_DEQUE_CAPACITY;
	deq->data[n] = (task_t*) entry;
	deq->tail++;
	return 1;
}

void dequeDestroy(deque_t* deq) {
	free(deq);
}

/*
 * the steal protocol
 */
task_t* dequeSteal(deque_t * deq) {
	int head;
	/* Cannot read deq->data[head] here
	 * Can happen that head=tail=0, then the owner of the deq pushes
	 * a new task when stealer is here in the code, resulting in head=0, tail=1
	 * All other checks down-below will be valid, but the old value of the buffer head
	 * would be returned by the steal rather than the new pushed value.
	 */
	int tail;

	head = deq->head;
	hc_mfence();
	tail = deq->tail;
	if ((tail - head) <= 0) {
		return NULL;
	}

	task_t* t = (task_t*) deq->data[head % INIT_DEQUE_CAPACITY];
	/* compete with other thieves and possibly the owner (if the size == 1) */
	if (hc_cas(&deq->head, head, head + 1)) { /* competing */
		return t;
	}
	return NULL;
}

/*
 * pop the task out of the deque from the tail
 */
task_t* dequePop(deque_t * deq) {
	hc_mfence();
	int tail = deq->tail;
	tail--;
	deq->tail = tail;
	hc_mfence();
	int head = deq->head;

	int size = tail - head;
	if (size < 0) {
		deq->tail = deq->head;
		return NULL;
	}
	task_t* t = (task_t*) deq->data[(tail) % INIT_DEQUE_CAPACITY];

	if (size > 0) {
		return t;
	}

	/* now size == 1, I need to compete with the thieves */
	if (!hc_cas(&deq->head, head, head + 1)) {
		t = NULL;
	}

	/* now the deque is empty */
	deq->tail = deq->head;
	return t;
}

/******************************************************/
/* Semi Concurrent DEQUE                              */
/******************************************************/

void semiConcDequeInit(semiConcDeque_t* semiDeq, void * initValue) {
	deque_t* deq = &semiDeq->deque;
	deq->head = 0;
	deq->tail = 0;
	semiDeq->lock = 0;
}

void semiConcDequeDestroy(semiConcDeque_t* semiDeq) {
	free(semiDeq);
}

void semiConcDequeLockedPush(semiConcDeque_t* semiDeq, void* entry) {
	deque_t* deq = &semiDeq->deque;
	int success = 0;
	while (!success) {
		int size = deq->tail - deq->head;
		if (INIT_DEQUE_CAPACITY == size) {
			HASSERT("DEQUE full, increase deque's size" && 0);
		}

		if (hc_cas(&semiDeq->lock, 0, 1) ) {
			success = 1;
			int n = deq->tail % INIT_DEQUE_CAPACITY;
			deq->data[n] = (task_t*) entry;
			hc_mfence();
			++deq->tail;
			semiDeq->lock= 0;
		}
	}
}

task_t* semiConcDequeNonLockedPop(semiConcDeque_t * semiDeq) {
	deque_t* deq = &semiDeq->deque;
	int head = deq->head;
	int tail = deq->tail;

	if ((tail - head) > 0) {
		task_t* t = (task_t*) deq->data[head % INIT_DEQUE_CAPACITY];
		++deq->head;
		return t;
	}
	return NULL;
}
