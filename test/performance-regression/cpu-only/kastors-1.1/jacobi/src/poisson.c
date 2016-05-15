#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#endif
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

# include "poisson.h"
# include "main.h"
# include "timer.h"


double r8mat_rms(int nx, int ny, double *a_);
void rhs(int nx, int ny, double *f_, int block_size);
void timestamp(void);
double u_exact(double x, double y);
double uxxyy_exact(double x, double y);

/*
Purpose:

MAIN is the main program for POISSON_OPENMP.

Discussion:

POISSON_OPENMP is a program for solving the Poisson problem.

This program uses OpenMP for parallel execution.

The Poisson equation

- DEL^2 U(X,Y) = F(X,Y)

is solved on the unit square [0,1] x [0,1] using a grid of NX by
NX evenly spaced points.  The first and last points in each direction
are boundary points.

The boundary conditions and F are set so that the exact solution is

U(x,y) = sin ( pi * x * y)

so that

- DEL^2 U(x,y) = pi^2 * ( x^2 + y^2) * sin ( pi * x * y)

The Jacobi iteration is repeatedly applied until convergence is detected.

For convenience in writing the discretized equations, we assume that NX = NY.

Licensing:

This code is distributed under the GNU LGPL license.

Modified:

14 December 2011

Author:

John Burkardt
*/

/******************************************************************************/

typedef struct _pragma126_omp_task {
    int (*matrix_size_ptr);
    int (*block_size_ptr);
    int (*niter_ptr);
    double (*dx_ptr);
    double (*dy_ptr);
    double (*error_ptr);
    int ii;
    int i;
    int jj;
    int j;
    int (*nx_ptr);
    int (*ny_ptr);
    double (*(*f_ptr));
    double (*(*u_ptr));
    double (*(*unew_ptr));
    struct user_parameters (*(*params_ptr));
 } pragma126_omp_task;

static void pragma126_omp_task_hclib_async(void *____arg);
double run(struct user_parameters* params)
{
    int matrix_size = params->matrix_size;
    if (matrix_size <= 0) {
        matrix_size = 512;
        params->matrix_size = matrix_size;
    }
    int block_size = params->blocksize;
    if (block_size <= 0) {
        block_size = 128;
        params->blocksize = block_size;
    }
    int niter = params->titer;
    if (niter <= 0) {
        niter = 4;
        params->titer = niter;
    }
    double dx;
    double dy;
    double error;
    int ii,i;
    int jj,j;
    int nx = matrix_size;
    int ny = matrix_size;
    double *f = (double *)malloc(nx * nx * sizeof(double));
    double *u = (double *)malloc(nx * nx * sizeof(double));
    double *unew = (double *)malloc(nx * ny * sizeof(double));

    /* test if valid */
    if ( (nx % block_size) || (ny % block_size) )
    {
        params->succeed = 0;
        params->string2display = "*****ERROR: blocsize must divide NX and NY";
        return 0;
    }


    /// INITIALISATION
    dx = 1.0 / (double) (nx - 1);
    dy = 1.0 / (double) (ny - 1);


    // Set the right hand side array F.
    rhs(nx, ny, f, block_size);

    /*
       Set the initial solution estimate UNEW.
       We are "allowed" to pick up the boundary conditions exactly.
       */
hclib_start_finish(); for (j = 0; j < ny; j+= block_size) {
            for (i = 0; i < nx; i+= block_size) {
 { 
pragma126_omp_task *new_ctx = (pragma126_omp_task *)malloc(sizeof(pragma126_omp_task));
new_ctx->matrix_size_ptr = &(matrix_size);
new_ctx->block_size_ptr = &(block_size);
new_ctx->niter_ptr = &(niter);
new_ctx->dx_ptr = &(dx);
new_ctx->dy_ptr = &(dy);
new_ctx->error_ptr = &(error);
new_ctx->ii = ii;
new_ctx->i = i;
new_ctx->jj = jj;
new_ctx->j = j;
new_ctx->nx_ptr = &(nx);
new_ctx->ny_ptr = &(ny);
new_ctx->f_ptr = &(f);
new_ctx->u_ptr = &(u);
new_ctx->unew_ptr = &(unew);
new_ctx->params_ptr = &(params);
hclib_async(pragma126_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
            }
        } ; hclib_end_finish(); 

    /// KERNEL INTENSIVE COMPUTATION
    START_TIMER;
    sweep(nx, ny, dx, dy, f, 0, niter, u, unew, block_size);
    END_TIMER;

    if(params->check) {
        double x;
        double y;
        double *udiff = (double *)malloc(nx * ny * sizeof(double));
        /// CHECK OUTPUT
        // Check for convergence.
        for (j = 0; j < ny; j++) {
            y = (double) (j) / (double) (ny - 1);
            for (i = 0; i < nx; i++) {
                x = (double) (i) / (double) (nx - 1);
                (udiff)[i * ny + j] = (unew)[i * ny + j] - u_exact(x, y);
            }
        }
        error = r8mat_rms(nx, ny, udiff);

        double error1;
        // Set the right hand side array F.
        rhs(nx, ny, f, block_size);

        /*
           Set the initial solution estimate UNEW.
           We are "allowed" to pick up the boundary conditions exactly.
           */
        for (j = 0; j < ny; j++) {
            for (i = 0; i < nx; i++) {
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                    (unew)[i * ny + j] = (f)[i * ny + j];
                } else {
                    (unew)[i * ny + j] = 0.0;
                }
            }
        }

        sweep_seq(nx, ny, dx, dy, f, 0, niter, u, unew);

        // Check for convergence.
        for (j = 0; j < ny; j++) {
            y = (double) (j) / (double) (ny - 1);
            for (i = 0; i < nx; i++) {
                x = (double) (i) / (double) (nx - 1);
                (udiff)[i * ny + j] = (unew)[i * ny + j] - u_exact(x, y);
            }
        }
        error1 = r8mat_rms(nx, ny, udiff);
        params->succeed = fabs(error - error1) < 1.0E-6;
        free(udiff);
    }
    free(f);
    free(u);
    free(unew);
    return TIMER;
} 
static void pragma126_omp_task_hclib_async(void *____arg) {
    pragma126_omp_task *ctx = (pragma126_omp_task *)____arg;
    int ii; ii = ctx->ii;
    int i; i = ctx->i;
    int jj; jj = ctx->jj;
    int j; j = ctx->j;
for (jj=j; jj<j+(*(ctx->block_size_ptr)); ++jj) {
                    for (ii=i; ii<i+(*(ctx->block_size_ptr)); ++ii)
                    {
                        if (ii == 0 || ii == (*(ctx->nx_ptr)) - 1 || jj == 0 || jj == (*(ctx->ny_ptr)) - 1) {
                            ((*(ctx->unew_ptr)))[ii * (*(ctx->ny_ptr)) + jj] = ((*(ctx->f_ptr)))[ii * (*(ctx->ny_ptr)) + jj];
                        } else {
                            ((*(ctx->unew_ptr)))[ii * (*(ctx->ny_ptr)) + jj] = 0.0;
                        }
                    }
                } ;     free(____arg);
}



/* R8MAT_RMS returns the RMS norm of a vector stored as a matrix. */
double r8mat_rms(int nx, int ny, double *a_) {
    double (*a)[nx][ny] = (double (*)[nx][ny])a_;
    int i;
    int j;
    double v;

    v = 0.0;

    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            v = v + (*a)[i][j] * (*a)[i][j];
        }
    }
    v = sqrt(v / (double) (nx * ny));

    return v;
}

/* RHS initializes the right hand side "vector". */
typedef struct _pragma235_omp_task {
    int i;
    int ii;
    int j;
    int jj;
    double x;
    double y;
    int nx;
    int ny;
    double (*(*f_ptr));
    int block_size;
 } pragma235_omp_task;

static void pragma235_omp_task_hclib_async(void *____arg);
void rhs(int nx, int ny, double *f, int block_size)
{
    int i,ii;
    int j,jj;
    double x;
    double y;

    // The "boundary" entries of F store the boundary values of the solution.
    // The "interior" entries of F store the right hand sides of the Poisson equation.

hclib_start_finish(); for (j = 0; j < ny; j+=block_size) {
        for (i = 0; i < nx; i+=block_size) {
 { 
pragma235_omp_task *new_ctx = (pragma235_omp_task *)malloc(sizeof(pragma235_omp_task));
new_ctx->i = i;
new_ctx->ii = ii;
new_ctx->j = j;
new_ctx->jj = jj;
new_ctx->x = x;
new_ctx->y = y;
new_ctx->nx = nx;
new_ctx->ny = ny;
new_ctx->f_ptr = &(f);
new_ctx->block_size = block_size;
hclib_async(pragma235_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
        }
    } ; hclib_end_finish(); 
} 
static void pragma235_omp_task_hclib_async(void *____arg) {
    pragma235_omp_task *ctx = (pragma235_omp_task *)____arg;
    int i; i = ctx->i;
    int ii; ii = ctx->ii;
    int j; j = ctx->j;
    int jj; jj = ctx->jj;
    double x; x = ctx->x;
    double y; y = ctx->y;
    int nx; nx = ctx->nx;
    int ny; ny = ctx->ny;
    int block_size; block_size = ctx->block_size;
    hclib_start_finish();
for (jj=j; jj<j+block_size; ++jj)
            {
                y = (double) (jj) / (double) (ny - 1);
                for (ii=i; ii<i+block_size; ++ii)
                {
                    x = (double) (ii) / (double) (nx - 1);
                    if (ii == 0 || ii == nx - 1 || jj == 0 || jj == ny - 1)
                        ((*(ctx->f_ptr)))[ii * ny + jj] = u_exact(x, y);
                    else
                        ((*(ctx->f_ptr)))[ii * ny + jj] = - uxxyy_exact(x, y);
                }
            } ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}



/* Evaluates the exact solution. */
double u_exact(double x, double y) {
    double pi = 3.141592653589793;
    double value;

    value = sin(pi * x * y);

    return value;
}

/* Evaluates (d/dx d/dx + d/dy d/dy) of the exact solution. */
double uxxyy_exact(double x, double y) {
    double pi = 3.141592653589793;
    double value;

    value = - pi * pi * (x * x + y * y) * sin(pi * x * y);

    return value;
}

