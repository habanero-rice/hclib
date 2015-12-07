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

void async_fct(void * arg) {
    void ** argv = (void **) arg;
    int index = *((int *) argv[0]);
    hclib_ddf_t ** ddf_list = (hclib_ddf_t **) argv[1];
    printf("Running async %d\n", index);
    /* Check value set by predecessor */
    int* prev = (int *) hclib_ddf_get(ddf_list[(index-1)*2]);
    assert(*prev == index-1);
    printf("Async %d putting in DDF %d @ %p\n", index, index*2, ddf_list[index*2]);
    int * value = (int *) malloc(sizeof(int)*1);
    *value = index;
    hclib_ddf_put(ddf_list[index*2], value);
    free(argv);
}

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    setbuf(stdout,NULL);
    int n = 5;
    hclib::ddf_t ** ddf_list = (hclib::ddf_t **)malloc(
            sizeof(hclib::ddf_t *) * (2*(n+1)));
    hclib::init(&argc, argv);
    hclib::finish([=]() {
        int index = 0;
        // Create asyncs
        // Building 'n' NULL-terminated lists of a single DDF each
        for (index = 0 ; index <= n; index++) {
            ddf_list[index*2] = hclib::ddf_create();
            printf("Populating ddf_list at address %p\n", &ddf_list[index*2]);
            ddf_list[index*2+1] = NULL;
        }

        for(index=n-1; index>=1; index--) {
            // Build async's arguments
            printf("Creating async %d await on %p will enable %p\n", index, ddf_list, &(ddf_list[index*2]));
            hclib::asyncAwait(ddf_list[(index - 1) * 2], [=]() {
                printf("Running async %d\n", index);
                int* prev = (int *) hclib::ddf_get(ddf_list[(index-1)*2]);
                assert(*prev == index-1);
                printf("Async %d putting in DDF %d @ %p\n", index, index*2, ddf_list[index*2]);
                int * value = (int *) malloc(sizeof(int)*1);
                *value = index;
                hclib::ddf_put(ddf_list[index*2], value); });
        }

        int * value = (int *) malloc(sizeof(int));
        *value = 0;
        printf("Putting in DDF 0\n");
        hclib::ddf_put(ddf_list[0], value);
    });
    // freeing everything up
    for (int index = 0 ; index <= n; index++) {
        free(hclib::ddf_get(ddf_list[index*2]));
        hclib::ddf_free(ddf_list[index*2]);
    }
    free(ddf_list);
    hclib::finalize();

    return 0;
}
