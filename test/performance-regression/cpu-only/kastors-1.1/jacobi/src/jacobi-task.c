#include "hclib.h"
# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
typedef struct _pragma23 {
    int i;
    int (*it_ptr);
    int j;
    int (*nx_ptr);
    int ny;
    double (*dx_ptr);
    double (*dy_ptr);
    double (*(*f_ptr));
    int (*itold_ptr);
    int (*itnew_ptr);
    double (*(*u_ptr));
    double (*(*unew_ptr));
    int (*block_size_ptr);
 } pragma23;

typedef struct _pragma31 {
    int i;
    int (*it_ptr);
    int j;
    int nx;
    int ny;
    double dx;
    double dy;
    double (*(*f_ptr));
    int (*itold_ptr);
    int (*itnew_ptr);
    double (*(*u_ptr));
    double (*(*unew_ptr));
    int (*block_size_ptr);
 } pragma31;

static void pragma23_hclib_async(void *____arg);
static void pragma31_hclib_async(void *____arg);
void sweep (int nx, int ny, double dx, double dy, double *f,
        int itold, int itnew, double *u, double *unew, int block_size)
{
    int i;
    int it;
    int j;
    // double (*f)[nx][ny] = (double (*)[nx][ny])f_;
    // double (*u)[nx][ny] = (double (*)[nx][ny])u_;
    // double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;

hclib_start_finish(); {
        for (it = itold + 1; it <= itnew; it++) {
            // Save the current estimate.
            for (i = 0; i < nx; i++) {
 { 
pragma23 *new_ctx = (pragma23 *)malloc(sizeof(pragma23));
new_ctx->i = i;
new_ctx->it_ptr = &(it);
new_ctx->j = j;
new_ctx->nx_ptr = &(nx);
new_ctx->ny = ny;
new_ctx->dx_ptr = &(dx);
new_ctx->dy_ptr = &(dy);
new_ctx->f_ptr = &(f);
new_ctx->itold_ptr = &(itold);
new_ctx->itnew_ptr = &(itnew);
new_ctx->u_ptr = &(u);
new_ctx->unew_ptr = &(unew);
new_ctx->block_size_ptr = &(block_size);
hclib_async(pragma23_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
            }
 hclib_end_finish(); hclib_start_finish(); ;
            // Compute a new estimate.
            for (i = 0; i < nx; i++) {
 { 
pragma31 *new_ctx = (pragma31 *)malloc(sizeof(pragma31));
new_ctx->i = i;
new_ctx->it_ptr = &(it);
new_ctx->j = j;
new_ctx->nx = nx;
new_ctx->ny = ny;
new_ctx->dx = dx;
new_ctx->dy = dy;
new_ctx->f_ptr = &(f);
new_ctx->itold_ptr = &(itold);
new_ctx->itnew_ptr = &(itnew);
new_ctx->u_ptr = &(u);
new_ctx->unew_ptr = &(unew);
new_ctx->block_size_ptr = &(block_size);
hclib_async(pragma31_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
            }
 hclib_end_finish(); hclib_start_finish(); ;
        }
    } ; hclib_end_finish(); 
} 
static void pragma23_hclib_async(void *____arg) {
    pragma23 *ctx = (pragma23 *)____arg;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int ny; ny = ctx->ny;
    hclib_start_finish();
for (j = 0; j < ny; j++) {
                    ((*(ctx->u_ptr)))[i * ny + j] = ((*(ctx->unew_ptr)))[i * ny + j];
                } ;     ; hclib_end_finish();

    free(____arg);
}


static void pragma31_hclib_async(void *____arg) {
    pragma31 *ctx = (pragma31 *)____arg;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int nx; nx = ctx->nx;
    int ny; ny = ctx->ny;
    double dx; dx = ctx->dx;
    double dy; dy = ctx->dy;
    hclib_start_finish();
for (j = 0; j < ny; j++) {
                    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
                        ((*(ctx->unew_ptr)))[i * ny + j] = ((*(ctx->f_ptr)))[i * ny + j];
                    } else {
                        ((*(ctx->unew_ptr)))[i * ny + j] = 0.25 * (((*(ctx->u_ptr)))[(i-1) * ny + j] + ((*(ctx->u_ptr)))[i * ny + (j+1)]
                                                + ((*(ctx->u_ptr)))[i * ny + (j-1)] + ((*(ctx->u_ptr)))[(i+1) * ny + j]
                                                + ((*(ctx->f_ptr)))[i * ny + j] * dx * dy);
                    }
                } ;     ; hclib_end_finish();

    free(____arg);
}


