#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#endif
# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
typedef struct _pragma26_omp_parallel {
    int it;
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
 } pragma26_omp_parallel;

typedef struct _pragma31_omp_parallel {
    int it;
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
 } pragma31_omp_parallel;

static void pragma26_omp_parallel_hclib_async(void *____arg, const int ___iter0, const int ___iter1);
static void pragma31_omp_parallel_hclib_async(void *____arg, const int ___iter0, const int ___iter1);
void sweep (int nx, int ny, double dx, double dy, double *f_,
        int itold, int itnew, double *u_, double *unew_, int block_size)
{
    int it;
    int block_x, block_y;

    if (block_size == 0)
        block_size = nx;

    int max_blocks_x = (nx / block_size);
    int max_blocks_y = (ny / block_size);
    
    for (it = itold + 1; it <= itnew; it++)
    {
        // Save the current estimate.
 { 
pragma26_omp_parallel *new_ctx = (pragma26_omp_parallel *)malloc(sizeof(pragma26_omp_parallel));
new_ctx->it = it;
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
hclib_loop_domain_t domain[2];
domain[0].low = 0;
domain[0].high = max_blocks_x;
domain[0].stride = 1;
domain[0].tile = -1;
domain[1].low = 0;
domain[1].high = max_blocks_y;
domain[1].stride = 1;
domain[1].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma26_omp_parallel_hclib_async, new_ctx, NULL, 2, domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(new_ctx);
 } ;

 { 
pragma31_omp_parallel *new_ctx = (pragma31_omp_parallel *)malloc(sizeof(pragma31_omp_parallel));
new_ctx->it = it;
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
hclib_loop_domain_t domain[2];
domain[0].low = 0;
domain[0].high = max_blocks_x;
domain[0].stride = 1;
domain[0].tile = -1;
domain[1].low = 0;
domain[1].high = max_blocks_y;
domain[1].stride = 1;
domain[1].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma31_omp_parallel_hclib_async, new_ctx, NULL, 2, domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(new_ctx);
 } ;
    }
} 
static void pragma26_omp_parallel_hclib_async(void *____arg, const int ___iter0, const int ___iter1) {
    pragma26_omp_parallel *ctx = (pragma26_omp_parallel *)____arg;
    int it; it = ctx->it;
    int block_x; block_x = ctx->block_x;
    int block_y; block_y = ctx->block_y;
    hclib_start_finish();
    do {
    block_x = ___iter0;
    block_y = ___iter1;
copy_block((*(ctx->nx_ptr)), (*(ctx->ny_ptr)), block_x, block_y, (*(ctx->u__ptr)), (*(ctx->unew__ptr)), (*(ctx->block_size_ptr))) ;     } while (0);
    ; hclib_end_finish_nonblocking();

}


static void pragma31_omp_parallel_hclib_async(void *____arg, const int ___iter0, const int ___iter1) {
    pragma31_omp_parallel *ctx = (pragma31_omp_parallel *)____arg;
    int it; it = ctx->it;
    int block_x; block_x = ctx->block_x;
    int block_y; block_y = ctx->block_y;
    hclib_start_finish();
    do {
    block_x = ___iter0;
    block_y = ___iter1;
compute_estimate(block_x, block_y, (*(ctx->u__ptr)), (*(ctx->unew__ptr)), (*(ctx->f__ptr)), (*(ctx->dx_ptr)), (*(ctx->dy_ptr)),
                                 (*(ctx->nx_ptr)), (*(ctx->ny_ptr)), (*(ctx->block_size_ptr))) ;     } while (0);
    ; hclib_end_finish_nonblocking();

}


