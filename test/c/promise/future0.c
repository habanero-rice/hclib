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
            hclib_future_t **future_list = (hclib_future_t **)malloc(
                    2 * sizeof(hclib_future_t *));
            assert(future_list);
            future_list[0] = prev;
            future_list[1] = NULL;
            prev = hclib_async_future(async_fct, count, future_list, NULL,
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
    hclib_launch(entrypoint, NULL);
    printf("Exiting...\n");
    return 0;
}
