/*
 *  RICE University
 *  Habanero Team
 *  
 *  This file is part of HC Test.
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include "hclib.h"

void *async_fct(void *arg) {
    int *count_ptr = (int *)arg;

    printf("Running async with count = %d\n", *count_ptr);
    *count_ptr = *count_ptr + 1;
    return NULL;
}

void entrypoint(void *arg) {

    int n_asyncs = 5;
    int *count = (int *)malloc(sizeof(int));
    assert(count);
    *count = 0;

    hclib_start_finish();
    int i;
    hclib_future_t *prev = NULL;
    for (i = 0; i < n_asyncs; i++) {
        if (prev) {
            prev = hclib_async_future(async_fct, count, prev, NULL);
        } else {
            prev = hclib_async_future(async_fct, count, NULL, NULL);
        }
    }
    hclib_end_finish();

    assert(*count == n_asyncs);
}

int main(int argc, char ** argv) {
    hclib_launch(entrypoint, NULL);
    printf("Exiting...\n");
    return 0;
}
