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
    hclib_promise_t * promise = (hclib_promise_t *) argv[1];
    printf("Running async %d\n", index/2);
    printf("Async %d putting in DDF %d @ %p\n", index/2, index, promise);
    hclib_promise_put(promise, NO_DATUM);
    free(argv);
}

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    int n = 5;
    hclib::promise_t ** promise_list = (hclib::promise_t **)malloc(
            sizeof(hclib::promise_t *) * (2*(n+1)));
    hclib::launch(&argc, argv, [=]() {
        hclib::finish([=]() {
            int index = 0;
            // Building 'n' NULL-terminated lists of a single DDF each
            for (index = 0 ; index <= n; index++) {
                promise_list[index*2] = hclib::promise_create();
                printf("Creating promise  %p at promise_list @ %p \n",
                        &promise_list[index*2], hclib::promise_get(promise_list[index*2]));
                promise_list[index*2+1] = NULL;
            }
            for(index=n-1; index>=1; index--) {
                printf("Creating async %d\n", index);
                // Build async's arguments
                printf("Creating async %d await on %p will enable %p\n", index,
                        &(promise_list[(index-1)*2]), &(promise_list[index*2]));
                hclib::asyncAwait(promise_list[(index-1)*2], [=]() {
                    hclib::promise_t *promise = promise_list[index * 2];
                    int index = index * 2;
                    hclib::promise_put(promise, NO_DATUM); });
            }
            printf("Putting in DDF 0\n");
            hclib::promise_put(promise_list[0], NO_DATUM);
        });
        // freeing everything up
        for (int index = 0 ; index <= n; index++) {
            free(hclib::promise_get(promise_list[index*2]));
            hclib::promise_free(promise_list[index*2]);
        }
        free(promise_list);
    });
    return 0;
}
