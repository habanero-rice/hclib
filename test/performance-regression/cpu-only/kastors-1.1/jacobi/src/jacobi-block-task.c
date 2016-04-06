#include "hclib.h"
# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
typedef struct _pragma26 {
    int it;
    int block_x;
    int block_y;
    int max_blocks_x;
    int max_blocks_y;
    int nx;
    int ny;
    double dx;
    double dy;
    double *f_;
    int itold;
    int itnew;
    double *u_;
    double *unew_;
    int block_size;
 } pragma26;

typedef struct _pragma36 {
    int it;
    int block_x;
    int block_y;
    int max_blocks_x;
    int max_blocks_y;
    int nx;
    int ny;
    double dx;
    double dy;
    double *f_;
    int itold;
    int itnew;
    double *u_;
    double *unew_;
    int block_size;
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
pragma26 *ctx = (pragma26 *)malloc(sizeof(pragma26));
ctx->it = it;
ctx->block_x = block_x;
ctx->block_y = block_y;
ctx->max_blocks_x = max_blocks_x;
ctx->max_blocks_y = max_blocks_y;
ctx->nx = nx;
ctx->ny = ny;
ctx->dx = dx;
ctx->dy = dy;
ctx->f_ = f_;
ctx->itold = itold;
ctx->itnew = itnew;
ctx->u_ = u_;
ctx->unew_ = unew_;
ctx->block_size = block_size;
hclib_async(pragma26_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;
                }
            }

 hclib_end_finish(); hclib_start_finish(); ;

            // Compute a new estimate.
            for (block_x = 0; block_x < max_blocks_x; block_x++) {
                for (block_y = 0; block_y < max_blocks_y; block_y++) {
 { 
pragma36 *ctx = (pragma36 *)malloc(sizeof(pragma36));
ctx->it = it;
ctx->block_x = block_x;
ctx->block_y = block_y;
ctx->max_blocks_x = max_blocks_x;
ctx->max_blocks_y = max_blocks_y;
ctx->nx = nx;
ctx->ny = ny;
ctx->dx = dx;
ctx->dy = dy;
ctx->f_ = f_;
ctx->itold = itold;
ctx->itnew = itnew;
ctx->u_ = u_;
ctx->unew_ = unew_;
ctx->block_size = block_size;
hclib_async(pragma36_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;
                }
            }

 hclib_end_finish(); hclib_start_finish(); ;
        }
    } ; hclib_end_finish(); 
} 
static void pragma26_hclib_async(void *____arg) {
    pragma26 *ctx = (pragma26 *)____arg;
    int it; it = ctx->it;
    int block_x; block_x = ctx->block_x;
    int block_y; block_y = ctx->block_y;
    int max_blocks_x; max_blocks_x = ctx->max_blocks_x;
    int max_blocks_y; max_blocks_y = ctx->max_blocks_y;
    int nx; nx = ctx->nx;
    int ny; ny = ctx->ny;
    double dx; dx = ctx->dx;
    double dy; dy = ctx->dy;
    double *f_; f_ = ctx->f_;
    int itold; itold = ctx->itold;
    int itnew; itnew = ctx->itnew;
    double *u_; u_ = ctx->u_;
    double *unew_; unew_ = ctx->unew_;
    int block_size; block_size = ctx->block_size;
    hclib_start_finish();
copy_block(nx, ny, block_x, block_y, u_, unew_, block_size) ;     ; hclib_end_finish();
}


static void pragma36_hclib_async(void *____arg) {
    pragma36 *ctx = (pragma36 *)____arg;
    int it; it = ctx->it;
    int block_x; block_x = ctx->block_x;
    int block_y; block_y = ctx->block_y;
    int max_blocks_x; max_blocks_x = ctx->max_blocks_x;
    int max_blocks_y; max_blocks_y = ctx->max_blocks_y;
    int nx; nx = ctx->nx;
    int ny; ny = ctx->ny;
    double dx; dx = ctx->dx;
    double dy; dy = ctx->dy;
    double *f_; f_ = ctx->f_;
    int itold; itold = ctx->itold;
    int itnew; itnew = ctx->itnew;
    double *u_; u_ = ctx->u_;
    double *unew_; unew_ = ctx->unew_;
    int block_size; block_size = ctx->block_size;
    hclib_start_finish();
compute_estimate(block_x, block_y, u_, unew_, f_, dx, dy,
                                     nx, ny, block_size) ;     ; hclib_end_finish();
}


