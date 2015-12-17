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
            hclib_ddf_t *prev = NULL;
            for (i = 0; i < n_asyncs; i++) {
                if (prev) {
                    hclib_ddf_t **ddf_list = (hclib_ddf_t **)malloc(
                            2 * sizeof(hclib_ddf_t *));
                    assert(ddf_list);
                    ddf_list[0] = prev;
                    ddf_list[1] = NULL;
                    prev = hclib::asyncFutureAwait(ddf_list, [=]() {
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
