/*
 *  RICE University
 *  Habanero Team
 *  
 *  This file is part of HC Test.
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include "hclib_cpp.h"

int main(int argc, char ** argv) {
    hclib::launch(&argc, argv, []() {
        int n_asyncs = 5;
        int *count = (int *)malloc(sizeof(int));
        assert(count);
        *count = 0;

        hclib::finish([=]() {
            int i;
            hclib_promise_t *prev = NULL;
            for (i = 0; i < n_asyncs; i++) {
                if (prev) {
                    hclib_promise_t **promise_list = (hclib_promise_t **)malloc(
                            2 * sizeof(hclib_promise_t *));
                    assert(promise_list);
                    promise_list[0] = prev;
                    promise_list[1] = NULL;
                    prev = hclib::asyncFutureAwait(promise_list, [=]() {
                            printf("Running async with count = %d\n", *count);
                            *count = *count + 1;
                        });
                } else {
                    prev = hclib::asyncFuture([=]() {
                            printf("Running async with count = %d\n", *count);
                            *count = *count + 1;
                        });
                }
            }
        });

        assert(*count == n_asyncs);
    });
    printf("Exiting...\n");
    return 0;
}
