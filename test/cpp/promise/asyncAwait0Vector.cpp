/*
 *  RICE University
 *  Habanero Team
 *  
 *  This file is part of HC Test.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    setbuf(stdout,nullptr);
    int n = 5;
    hclib::promise_t<int*> **promise_list = new hclib::promise_t<int*> *[n+1];
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [=]() {
        hclib::finish([=]() {
            int index = 0;
            // Create asyncs
            for (index = 0 ; index <= n; index++) {
                promise_list[index] = new hclib::promise_t<int*>();
                printf("Populating promise_list at address %p\n",
                        &promise_list[index]);
            }
            for (index = n; index >= 1; index--) {
                // Build async's arguments
                printf("Creating async %d await on %p will enable %p\n", index,
                        promise_list, &(promise_list[index]));
                std::vector<hclib_future_t*> fut_vec;
                fut_vec.push_back(promise_list[index-1]->get_future());
                hclib::async_await([=]() {
                    printf("Running async %d\n", index);
                    hclib::future_t<int*> *future = promise_list[index - 1]->get_future();
                    int *input = (int *)future->get();
                    assert(*input == index - 1);

                    printf("Async %d putting in promise %d @ %p\n", index, index,
                            promise_list[index]);
                    promise_list[index]->put(new int(index));
                }, fut_vec);
                //}, std::vector<hclib_future_t*>{promise_list[index-1]->get_future()});
            }

            printf("Putting in promise 0\n");
            promise_list[0]->put(new int(0));
        });
        // freeing everything up
        for (int index = 0 ; index <= n; index++) {
            delete promise_list[index]->future().get();
            delete promise_list[index];
        }
        delete[] promise_list;
    });

    return 0;
}

