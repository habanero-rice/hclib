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

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    int n = 5;
    hclib::promise_t **promise_list = (hclib::promise_t **)malloc(sizeof(hclib::promise_t *) * (n + 1));

    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [=]() {
        hclib::finish([=]() {
            int index = 0;

            // Building 'n' NULL-terminated lists of a single promise each
            for (index = 0 ; index <= n; index++) {
                promise_list[index] = new hclib::promise_t();
            }

            for (index = n - 1; index >= 1; index--) {
                // Build async's arguments
                printf("Creating async %d\n", index);
                hclib::async_await([=]() {
                        hclib::future_t *future = promise_list[index - 1]->get_future();
                        int *input = (int *)future->get();
                        assert(*input == index - 1);

                        hclib::promise_t *promise = promise_list[index];
                        printf("Running async %d\n", index);
                        int *value = (int *)malloc(sizeof(int));
                        *value = index;
                        promise->put(value); }, promise_list[index - 1]->get_future());
            }
            int *value = (int *)malloc(sizeof(int));
            *value = 0;
            printf("Putting in promise 0\n");
            promise_list[0]->put(value);
        });
        // freeing everything up
        for (int index = 0 ; index <= n; index++) {
            free(promise_list[index]->get_future()->get());
            delete promise_list[index];
        }
        free(promise_list);
    });
    return 0;
}
