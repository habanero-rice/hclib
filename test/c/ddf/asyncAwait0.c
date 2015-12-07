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
    hclib_ddf_t * ddf = (hclib_ddf_t *) argv[1];
    printf("Running async %d\n", index/2);
    printf("Async %d putting in DDF %d @ %p\n", index/2, index, ddf);
    int * value = (int *) malloc(sizeof(int)*1);
    *value = index;
    hclib_ddf_put(ddf, value);
    free(argv);
}

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    hclib_init(&argc, argv);
    hclib_start_finish();
    int n = 5;
    int index = 0;
    // Building 'n' NULL-terminated lists of a single DDF each
    hclib_ddf_t ** ddf_list = (hclib_ddf_t **)malloc(
            sizeof(hclib_ddf_t *) * (2*(n+1)));
    for (index = 0 ; index <= n; index++) {
        ddf_list[index*2] = hclib_ddf_create();
        printf("Creating ddf  %p at ddf_list @ %p \n", &ddf_list[index*2],
                hclib_ddf_get(ddf_list[index*2]));
        ddf_list[index*2+1] = NULL;
    }
    for(index=n-1; index>=1; index--) {
        printf("Creating async %d\n", index);
        // Build async's arguments
        void ** argv = malloc(sizeof(void *) * 2);
        argv[0] = malloc(sizeof(int) *1);
        *((int *)argv[0]) = index*2;
        argv[1] = (void *)(ddf_list[index*2]);
        printf("Creating async %d await on %p will enable %p\n", index,
                &(ddf_list[(index-1)*2]), &(ddf_list[index*2]));
        hclib_async(async_fct, argv, &(ddf_list[(index-1)*2]), NULL, NO_PROP);
    }
    int * value = (int *) malloc(sizeof(int)*1);
    *value = 2222;
    printf("Putting in DDF 0\n");
    hclib_ddf_put(ddf_list[0], value);
    hclib_end_finish();
    // freeing everything up
    for (index = 0 ; index <= n; index++) {
        free(hclib_ddf_get(ddf_list[index*2]));
        hclib_ddf_free(ddf_list[index*2]);
    }
    free(ddf_list);
    hclib_finalize();
    return 0;
}
