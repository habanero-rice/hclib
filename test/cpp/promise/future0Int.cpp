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
    hclib::launch([]() {
        constexpr int n_asyncs = 5;
        hclib::future_t<int> *prev = nullptr;

        HCLIB_FINISH {
            for (int i = 0; i < n_asyncs; i++) {
                if (prev) {
                    auto dep = prev;
                    prev = hclib::async_future_await([dep]() {
                            const int count = dep->get();
                            printf("Running async with count = %d\n", count);
                            return count + 1;
                        }, dep);
                } else {
                    prev = hclib::async_future([]() {
                            printf("Running async with count = 0\n");
                            return 1;
                        });
                }
            }
        }

        const int end_count = prev->get();
        assert(end_count == n_asyncs);

    });
    printf("Exiting...\n");
    return 0;
}
