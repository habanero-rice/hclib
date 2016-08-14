/*
 *  RICE University
 *  Habanero Team
 *  
 *  This file is part of HC Test.
 *
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
