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
    struct ddf_st * ddf = (struct ddf_st *) argv[1];
    printf("Running async %d\n", index/2);
    printf("Async %d putting in DDF %d @ %p\n", index/2, index, ddf);
    ddf_put(ddf, NO_DATUM);
    free(argv);
}

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    hclib_init(&argc, argv);
    start_finish();
    int n = 5;
    int index = 0;
    // Building 'n' NULL-terminated lists of a single DDF each
    struct ddf_st ** ddf_list = (struct ddf_st **) malloc(sizeof(struct ddf_st *) * (2*(n+1)));
    for (index = 0 ; index <= n; index++) {
        ddf_list[index*2] = ddf_create();
        printf("Creating ddf  %p at ddf_list @ %p \n", &ddf_list[index*2], ddf_get(ddf_list[index*2]));
        ddf_list[index*2+1] = NULL;
    }
    for(index=n-1; index>=1; index--) {
        printf("Creating async %d\n", index);
        // Build async's arguments
        void ** argv = malloc(sizeof(void *) * 2);
        argv[0] = malloc(sizeof(int) *1);
        *((int *)argv[0]) = index*2;
        argv[1] = (void *)(ddf_list[index*2]);
        printf("Creating async %d await on %p will enable %p\n", index, &(ddf_list[(index-1)*2]), &(ddf_list[index*2]));
        async(async_fct, argv, &(ddf_list[(index-1)*2]), NULL, NO_PROP);
    }
    printf("Putting in DDF 0\n");
    ddf_put(ddf_list[0], NO_DATUM);
    end_finish();
    // freeing everything up
    for (index = 0 ; index <= n; index++) {
        free(ddf_get(ddf_list[index*2]));
        ddf_free(ddf_list[index*2]);
    }
    free(ddf_list);
    hclib_finalize();
    return 0;
}
