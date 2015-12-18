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
    hclib_ddf_t *event = (hclib_ddf_t *)arg;
    int *signal = (int *)malloc(sizeof(int));
    assert(signal);
    *signal = 42;

    sleep(5);

    hclib_ddf_put(event, signal);
}

void consumer(void *arg) {
    hclib_ddf_t *event = (hclib_ddf_t *)arg;
    int *signal = (int *)hclib_ddf_wait(event);
    assert(*signal == 42);
    printf("signal = %d\n", *signal);
}

void entrypoint(void *arg) {

    hclib_start_finish();

    hclib_ddf_t *event = hclib_ddf_create();
    hclib_async(consumer, event, NULL, NULL, NULL, NO_PROP);
    hclib_async(producer, event, NULL, NULL, NULL, NO_PROP);

    hclib_end_finish();
}

int main(int argc, char ** argv) {
    hclib_launch(&argc, argv, entrypoint, NULL);
    printf("Exiting...\n");
    return 0;
}
