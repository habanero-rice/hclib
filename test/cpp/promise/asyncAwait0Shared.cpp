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
#include <assert.h>
#include <memory>

#include "hclib.hpp"

/*
 * Create async await and enable them (by a put) in the
 * reverse order they've been created.
 *
 * NOTE: packing a shared_ptr into a promise_t is not currently supported,
 * so we can't use std::make_shared as one would usually do.
 */
int main(int argc, char ** argv) {
    setbuf(stdout,nullptr);
    int n = 5;
    hclib::promise_t<std::shared_ptr<int>*> **promise_list = new hclib::promise_t<std::shared_ptr<int>*> *[n+1];
    hclib::launch([=]() {
        hclib::finish([=]() {
            int index = 0;
            // Create asyncs
            for (index = 0 ; index <= n; index++) {
                promise_list[index] = new hclib::promise_t<std::shared_ptr<int>*>();
                printf("Populating promise_list at address %p\n",
                        &promise_list[index]);
            }

            for(index=n-1; index>=1; index--) {
                // Build async's arguments
                printf("Creating async %d await on %p will enable %p\n", index,
                        promise_list, &(promise_list[index]));
                hclib::async_await([=]() {
                    printf("Running async %d\n", index);
                    int prev = **promise_list[index-1]->get_future()->get();
                    assert(prev == index-1);
                    printf("Async %d putting in promise %d @ %p\n", index, index,
                            promise_list[index]);
                    promise_list[index]->put(
                            new std::shared_ptr<int>(new int(index)));
                }, promise_list[index-1]->get_future());
            }

            printf("Putting in promise 0\n");
            promise_list[0]->put(new std::shared_ptr<int>(new int(0)));
        });
        // freeing everything up
        for (int index = 0 ; index <= n; index++) {
            delete promise_list[index];
        }
        delete[] promise_list;
    });

    return 0;
}

