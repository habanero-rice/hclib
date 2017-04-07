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
