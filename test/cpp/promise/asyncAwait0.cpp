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
    hclib::promise_t ** promise_list = (hclib::promise_t **)malloc(
            sizeof(hclib::promise_t *) * (2*(n+1)));

    hclib::launch([=]() {
        hclib::finish([=]() {
            int index = 0;

            // Building 'n' NULL-terminated lists of a single promise each
            for (index = 0 ; index <= n; index++) {
                promise_list[index*2] = new hclib::promise_t();
                printf("Creating promise  %p at promise_list\n",
                        &promise_list[index*2]);
                promise_list[index*2+1] = NULL;
            }

            for (index = n - 1; index >= 1; index--) {
                printf("Creating async %d\n", index);
                // Build async's arguments
                printf("Creating async %d await on %p will enable %p\n", index,
                        &(promise_list[(index-1)*2]), &(promise_list[index*2]));
                hclib::async_await([=]() {
                        hclib::promise_t *promise = promise_list[(index * 2)];
                        int my_index = index * 2;
                        printf("Running async %d\n", my_index/2);
                        printf("Async %d putting in promise %d @ %p\n", my_index/2,
                                my_index, promise);
                        int * value = (int *) malloc(sizeof(int)*1);
                        *value = my_index;
                        promise->put(value); }, promise_list[(index - 1)*2]->get_future());
            }
            int * value = (int *) malloc(sizeof(int)*1);
            *value = 2222;
            printf("Putting in promise 0\n");
            promise_list[0]->put(value);
        });
        // freeing everything up
        for (int index = 0 ; index <= n; index++) {
            free(promise_list[index*2]->get_future()->get());
            delete promise_list[index*2];
        }
        free(promise_list);
    });
    return 0;
}
