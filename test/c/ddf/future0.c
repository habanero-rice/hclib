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
    hclib_ddf_t *prev = NULL;
    for (i = 0; i < n_asyncs; i++) {
        if (prev) {
            hclib_ddf_t **ddf_list = (hclib_ddf_t **)malloc(
                    2 * sizeof(hclib_ddf_t *));
            assert(ddf_list);
            ddf_list[0] = prev;
            ddf_list[1] = NULL;
            prev = hclib_async_future(async_fct, count, ddf_list, NULL,
                    NULL, NO_PROP);
        } else {
            prev = hclib_async_future(async_fct, count, NULL, NULL, NULL,
                    NO_PROP);
        }
    }
    hclib_end_finish();

    assert(*count == n_asyncs);
}

int main(int argc, char ** argv) {
    hclib_launch(&argc, argv, entrypoint, NULL);
    printf("Exiting...\n");
    return 0;
}
