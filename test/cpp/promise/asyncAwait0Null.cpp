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

void async_fct(void * arg) {
    void ** argv = (void **) arg;
    int index = *((int *) argv[0]);
    hclib::promise_t * promise = (hclib::promise_t *) argv[1];
    printf("Running async %d\n", index/2);
    printf("Async %d putting in promise %d @ %p\n", index/2, index, promise);
    promise->put(NO_DATUM);
    free(argv);
}

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    int n = 5;
    hclib::promise_t ** promise_list = (hclib::promise_t **)malloc(
            sizeof(hclib::promise_t *) * (n + 1));

    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [=]() {
        hclib::finish([=]() {
            int index = 0;
            // Building 'n' NULL-terminated lists of a single promise each

            for (index = 0 ; index <= n; index++) {
                promise_list[index] = new hclib::promise_t();
            }

            for(index=n-1; index>=1; index--) {
                printf("Creating async %d\n", index);
                // Build async's arguments
                hclib::async_await([=]() {
                    hclib::promise_t *promise = promise_list[index];
                    promise->put(NO_DATUM); }, promise_list[index - 1]->get_future());
            }
            printf("Putting in promise 0\n");
            promise_list[0]->put(NO_DATUM);
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
