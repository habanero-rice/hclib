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
    setbuf(stdout,NULL);
    int n = 5;
    hclib::promise_t ** promise_list = (hclib::promise_t **)malloc(
            sizeof(hclib::promise_t *) * (2*(n+1)));
    hclib::launch([=]() {
        hclib::finish([=]() {
            int index = 0;
            // Create asyncs
            // Building 'n' NULL-terminated lists of a single promise each
            for (index = 0 ; index <= n; index++) {
                promise_list[index*2] = new hclib::promise_t();
                printf("Populating promise_list at address %p\n",
                        &promise_list[index*2]);
                promise_list[index*2+1] = NULL;
            }

            for(index=n; index>=1; index--) {
                // Build async's arguments
                printf("Creating async %d await on %p will enable %p\n", index,
                        promise_list, &(promise_list[index*2]));
                hclib::async_await([=]() {
                    printf("Running async %d\n", index);
                    int* prev = (int *) promise_list[(index-1)*2]->get_future()->get();
                    assert(*prev == index-1);
                    printf("Async %d putting in promise %d @ %p\n", index, index*2,
                            promise_list[index*2]);
                    int * value = (int *) malloc(sizeof(int)*1);
                    *value = index;
                    promise_list[index*2]->put(value); }, promise_list[(index - 1) * 2]->get_future());
            }

            int * value = (int *) malloc(sizeof(int));
            *value = 0;
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
