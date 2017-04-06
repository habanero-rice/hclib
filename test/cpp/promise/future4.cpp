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

/**
 * DESC: Test if future->wait() preserves finish scope
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "hclib_cpp.h"

int main(int argc, char **argv) {
    bool *ran = new bool{false};
    assert(ran);

    hclib::launch([=]() {
        hclib::promise_t<void> *promise = new hclib::promise_t<void>();

        for (int i = 0; i < 30; i++) {
            hclib::async([=]() { usleep(10000); });
        }

        HCLIB_FINISH {
            hclib::async([=]() {
                usleep(500000);
                *ran = true;
            });

            hclib::async([=]() {
                usleep(20000);
                promise->put();
            });

            usleep(10000);

            promise->future().wait();
        }
        assert(*ran);
    });

    delete ran;
    printf("OK\n");

    return 0;
}
