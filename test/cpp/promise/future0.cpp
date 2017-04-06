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

#include "hclib.hpp"

int main(int argc, char ** argv) {
    hclib::launch([]() {
        int n_asyncs = 5;
        int *count = (int *)malloc(sizeof(int));
        assert(count);
        *count = 0;

        hclib::finish([=]() {
            int i;
            hclib::future_t<void> *prev = nullptr;
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
