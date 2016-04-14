#include "hclib.h"
# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
typedef struct _pragma26 {
    int (*it_ptr);
    int block_x;
    int block_y;
    int (*max_blocks_x_ptr);
    int (*max_blocks_y_ptr);
    int (*nx_ptr);
    int (*ny_ptr);
    double (*dx_ptr);
    double (*dy_ptr);
    double (*(*f__ptr));
    int (*itold_ptr);
    int (*itnew_ptr);
    double (*(*u__ptr));
    double (*(*unew__ptr));
    int (*block_size_ptr);
 } pragma26;

typedef struct _pragma36 {
    int block_x;
    int block_y;
    int (*nx_ptr);
    int (*ny_ptr);
    double (*dx_ptr);
    double (*dy_ptr);
    double (*(*f__ptr));
    double (*(*u__ptr));
    double (*(*unew__ptr));
    int (*block_size_ptr);
 } pragma36;

static void pragma26_hclib_async(void *____arg);
static void pragma36_hclib_async(void *____arg);
void sweep (int nx, int ny, double dx, double dy, double *f_,
            int itold, int itnew, double *u_, double *unew_, int block_size)
{
    int it;
    int block_x, block_y;

    if (block_size == 0)
        block_size = nx;

    int max_blocks_x = (nx / block_size);
    int max_blocks_y = (ny / block_size);

hclib_start_finish(); {
        for (it = itold + 1; it <= itnew; it++) {
            // Save the current estimate.
            for (block_x = 0; block_x < max_blocks_x; block_x++) {
                for (block_y = 0; block_y < max_blocks_y; block_y++) {
 { 
pragma26 *new_ctx = (pragma26 *)malloc(sizeof(pragma26));
new_ctx->it_ptr = &(it);
new_ctx->block_x = block_x;
new_ctx->block_y = block_y;
new_ctx->max_blocks_x_ptr = &(max_blocks_x);
new_ctx->max_blocks_y_ptr = &(max_blocks_y);
new_ctx->nx_ptr = &(nx);
new_ctx->ny_ptr = &(ny);
new_ctx->dx_ptr = &(dx);
new_ctx->dy_ptr = &(dy);
new_ctx->f__ptr = &(f_);
new_ctx->itold_ptr = &(itold);
new_ctx->itnew_ptr = &(itnew);
new_ctx->u__ptr = &(u_);
new_ctx->unew__ptr = &(unew_);
new_ctx->block_size_ptr = &(block_size);
hclib_async(pragma26_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
                }
            }

 hclib_end_finish(); hclib_start_finish(); ;

            // Compute a new estimate.
            for (block_x = 0; block_x < max_blocks_x; block_x++) {
                for (block_y = 0; block_y < max_blocks_y; block_y++) {
 { 
pragma36 *new_ctx = (pragma36 *)malloc(sizeof(pragma36));
new_ctx->block_x = block_x;
new_ctx->block_y = block_y;
new_ctx->nx_ptr = &(nx);
new_ctx->ny_ptr = &(ny);
new_ctx->dx_ptr = &(dx);
new_ctx->dy_ptr = &(dy);
new_ctx->f__ptr = &(f_);
new_ctx->u__ptr = &(u_);
new_ctx->unew__ptr = &(unew_);
new_ctx->block_size_ptr = &(block_size);
hclib_async(pragma36_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
                }
            }

 hclib_end_finish(); hclib_start_finish(); ;
        }
    } ; hclib_end_finish(); 
} 
static void pragma26_hclib_async(void *____arg) {
    pragma26 *ctx = (pragma26 *)____arg;
    int block_x; block_x = ctx->block_x;
    int block_y; block_y = ctx->block_y;
    hclib_start_finish();
copy_block((*(ctx->nx_ptr)), (*(ctx->ny_ptr)), block_x, block_y, (*(ctx->u__ptr)), (*(ctx->unew__ptr)), (*(ctx->block_size_ptr))) ;     ; hclib_end_finish();

    free(____arg);
}


static void pragma36_hclib_async(void *____arg) {
    pragma36 *ctx = (pragma36 *)____arg;
    int block_x; block_x = ctx->block_x;
    int block_y; block_y = ctx->block_y;
    hclib_start_finish();
compute_estimate(block_x, block_y, (*(ctx->u__ptr)), (*(ctx->unew__ptr)), (*(ctx->f__ptr)), (*(ctx->dx_ptr)), (*(ctx->dy_ptr)),
                                     (*(ctx->nx_ptr)), (*(ctx->ny_ptr)), (*(ctx->block_size_ptr))) ;     ; hclib_end_finish();

    free(____arg);
}


