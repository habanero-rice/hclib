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
#include <sys/time.h>
#include <signal.h>

#include "hclib.h"

/* DESC: master-helper deadlock scenario
 *
 * This example demonstrates how the current "work-shift" strategy used by HC
 * workers (in which a blocked worker tries to steal a new task) can easily lead
 * to deadlock. This example will deadlock when run with 4 workers
 * when using the master-helper strategy with a (help-global without fibers).
 *
 * Output from the program in the deadlock scenario is as follows:
 *
 *     Task C run by worker 0
 *     Task A run by worker 1
 *     Sleep A run by worker 3
 *     Task B run by worker 1
 *     Deadlocked (as expected).
 *
 * The worker IDs may change, but the key is that the delays in the code
 * virtually guarantee that Task B will be run on top of Task A,
 * which is what leads to the deadlock.
 */


#if 1
#define DELAY(t) usleep(t*100000L)
#else
#define DELAY(t) /* no delay */
#endif

extern int get_current_worker();

static inline void echo_worker(const char *taskName) {
    printf("%s run by worker %d\n", taskName, get_current_worker());
}

hclib_future_t **future_list;
hclib_promise_t *promise;
int data;

void taskSleep(void *from_task_A) {
    echo_worker(from_task_A ? "Sleep A" : "Sleep B");
    DELAY(5);
}

void taskA(void *args) {
    echo_worker("Task A");
    HCLIB_FINISH {
        hclib_async(taskSleep, taskA, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
        DELAY(2);
    }
    echo_worker("Promise put");
    hclib_promise_put(promise, &data);
}

void taskB(void *args) {
    echo_worker("Task B");
    HCLIB_FINISH {
        hclib_async(taskSleep, NULL, future_list, NO_PHASER, ANY_PLACE, NO_PROP);
    }
}

void taskC(void *args) {
    echo_worker("Task C");
    DELAY(1);
    hclib_async(taskB, NULL, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    DELAY(5);
}

int expecting_deadlock(void) {
    return (HCLIB_WORKER_STRATEGY == HCLIB_WORKER_STRATEGY_FIXED)
        && (HCLIB_WORKER_OPTIONS & HCLIB_WORKER_OPTIONS_HELP_GLOBAL);
}

void deadlock_handler(int arg) {
    if (!expecting_deadlock()) {
        printf("FAILURE: unexpected deadlock.\n");
        exit(1);
    }
    else {
        printf("Deadlocked (as expected).\n");
        exit(0);
    }
}

void set_deadlock_alarm(void) {
    struct itimerval it_val = {
        .it_value = { // wait 3 seconds
            .tv_sec  = 3,
            .tv_usec = 0
        },
        .it_interval = { 0 } // no repeat
    };
    HCHECK(signal(SIGALRM, deadlock_handler) == SIG_ERR);
    HCHECK(setitimer(ITIMER_REAL, &it_val, NULL));
}

void taskMain(void *args) {
    promise = hclib_promise_create();

    future_list = (hclib_future_t **)malloc(2 * sizeof(hclib_future_t *));
    assert(future_list);
    future_list[0] = hclib_get_future_for_promise(promise);
    future_list[1] = NULL;

    set_deadlock_alarm();

    hclib_async(&taskA, NULL, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    hclib_async(&taskC, NULL, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    DELAY(10);
}

int main(int argc, char ** argv) {
    setenv("HCLIB_WORKERS", "4", 1);
    hclib_launch(taskMain, NULL);
    printf("Done\n");
    if (expecting_deadlock()) {
        printf("FAILURE: expected deadlock.\n");
        return 1;
    }
    return 0;
}
