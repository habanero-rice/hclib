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

#include "hclib.h"

void producer(void *arg) {
    hclib_promise_t *event = (hclib_promise_t *)arg;
    int *signal = (int *)malloc(sizeof(int));
    assert(signal);
    *signal = 42;

    sleep(1);

    hclib_promise_put(event, signal);
}

void consumer(void *arg) {
    hclib_future_t *event = (hclib_future_t *)arg;
    int *signal = (int *)hclib_future_wait(event);
    assert(*signal == 42);
    printf("signal = %d\n", *signal);
}

void entrypoint(void *arg) {

    hclib_start_finish();

    hclib_promise_t *event = hclib_promise_create();
    hclib_async(consumer, hclib_get_future_for_promise(event), NO_FUTURE, NO_PHASER,
            ANY_PLACE, NO_PROP);
    hclib_async(producer, event, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);

    hclib_end_finish();
}

int main(int argc, char ** argv) {
    hclib_launch(entrypoint, NULL);
    printf("Exiting...\n");
    return 0;
}
