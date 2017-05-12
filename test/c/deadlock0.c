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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <assert.h>

#include "hclib.h"

/* DESC: master-helper deadlock scenario
 * This example demonstrates how the current "work-shift" strategy used by HC 
 * workers (in which a blocked worker tries to steal a new task) can easily lead 
 * to deadlock. This example will deadlock when run with 4 workers
 * when using the master-helper strategy with a (help-global without fibers).
 */


#if 1
#define DELAY(t) usleep(t*100000L)
#else
#define DELAY(t) /* no delay */
#endif

extern int get_current_worker();

static inline void echo_worker(const char *taskName) {
    printf("Task %s run by worker %d\n", taskName, get_current_worker());
}

hclib_future_t **future_list;
hclib_promise_t *promise;
int data;

void taskSleep(void *args) {
    echo_worker(args ? "Sleep A" : "Sleep B");
    DELAY(5);
}

void taskA(void *args) {
    echo_worker("A");
    HCLIB_FINISH {
        hclib_async(taskSleep, taskA, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
        DELAY(2);
    }
    printf("%p <- %p\n", promise, &data);
    hclib_promise_put(promise, &data);
}

void taskB(void *args) {
    echo_worker("B");
    HCLIB_FINISH {
        hclib_async(taskSleep, NULL, future_list, NO_PHASER, ANY_PLACE, NO_PROP);
    }
}

void taskC(void *args) {
    echo_worker("C");
    DELAY(1);
    hclib_async(taskB, NULL, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    DELAY(5);
}

void taskMain(void *args) {
    promise = hclib_promise_create();

    future_list = (hclib_future_t **)malloc(2 * sizeof(hclib_future_t *));
    assert(future_list);
    future_list[0] = hclib_get_future_for_promise(promise);
    future_list[1] = NULL;

    hclib_async(&taskA, NULL, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    hclib_async(&taskC, NULL, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    DELAY(10);
}

int main(int argc, char ** argv) {
    hclib_launch(taskMain, NULL);
    printf("Done\n");
}
