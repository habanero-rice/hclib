/*
 *  RICE University
 *  Habanero Team
 *  
 *  This file is part of HC Test.
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include "hclib.h"

void async_fct(void * arg) {
    void ** argv = (void **) arg;
    int index = *((int *) argv[0]);
    hclib_promise_t * promise = (hclib_promise_t *) argv[1];
    printf("Running async %d\n", index/2);
    printf("Async %d putting in promise %d @ %p\n", index/2, index, promise);
    hclib_promise_put(promise, NO_DATUM);
    free(argv);
}

void entrypoint(void *arg) {
    hclib_start_finish();
    int n = 5;
    int index = 0;
    // Building 'n' NULL-terminated lists of a single promise each
    hclib_promise_t ** promise_list = (hclib_promise_t **)malloc(
            sizeof(hclib_promise_t *) * (2*(n+1)));
    hclib_future_t **future_list =
        (hclib_future_t **)malloc(sizeof(hclib_future_t *) * (2 * (n + 1)));

    for (index = 0 ; index <= n; index++) {
        promise_list[index * 2] = hclib_promise_create();
        future_list[index * 2] = hclib_get_future_for_promise(promise_list[index * 2]);

        printf("Creating promise  %p at promise_list @ %p \n", &promise_list[index*2],
                hclib_future_get(future_list[index*2]));
        promise_list[index*2+1] = NULL;
        future_list[index * 2 + 1] = NULL;
    }
    for(index=n-1; index>=1; index--) {
        printf("Creating async %d\n", index);
        // Build async's arguments
        void ** argv = malloc(sizeof(void *) * 2);
        argv[0] = malloc(sizeof(int) *1);
        *((int *)argv[0]) = index*2;
        argv[1] = (void *)(promise_list[index*2]);
        printf("Creating async %d await on %p will enable %p\n", index,
                &(promise_list[(index-1)*2]), &(promise_list[index*2]));
        hclib_async(async_fct, argv, &(future_list[(index-1)*2]), NO_PHASER,
                ANY_PLACE, NO_PROP);
    }
    printf("Putting in promise 0\n");
    hclib_promise_put(promise_list[0], NO_DATUM);
    hclib_end_finish();
    // freeing everything up
    for (index = 0 ; index <= n; index++) {
        free(hclib_future_get(future_list[index*2]));
        hclib_promise_free(promise_list[index*2]);
    }
    free(promise_list);
}

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    hclib_launch(entrypoint, NULL);
    printf("Exiting...\n");
    return 0;
}
