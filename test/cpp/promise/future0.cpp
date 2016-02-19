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
            hclib::future_t *prev = NULL;
            for (i = 0; i < n_asyncs; i++) {
                if (prev) {
                    prev = hclib::async_future_await([=]() {
                            printf("Running async with count = %d\n", *count);
                            *count = *count + 1;
                        }, prev);
                } else {
                    prev = hclib::async_future([=]() {
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
