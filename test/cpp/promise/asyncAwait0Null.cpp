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

            for(index=n; index>=1; index--) {
                printf("Creating async %d\n", index);
                // Build async's arguments
                printf("Creating async %d await on %p will enable %p\n", index,
                        &(promise_list[(index-1)*2]), &(promise_list[index*2]));
                hclib::async_await([=]() {
                    hclib::promise_t *promise = promise_list[index * 2];
                    promise->put(NO_DATUM); }, promise_list[(index - 1)*2]->get_future());
            }
            printf("Putting in promise 0\n");
            promise_list[0]->put(NO_DATUM);
        });
        // freeing everything up
        for (int index = 0 ; index <= n; index++) {
            delete promise_list[index*2];
        }
        free(promise_list);
    });
    return 0;
}
