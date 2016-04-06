#include "hclib.h"
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

typedef struct _pragma121 {
    int matrix_size;
    int block_size;
    int niter;
    double dx;
    double dy;
    double error;
    int ii;
    int i;
    int jj;
    int j;
    int nx;
    int ny;
    double *f;
    double *u;
    double *unew;
    struct user_parameters *params;
 } pragma121;

static void pragma121_hclib_async(void *____arg);
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
pragma121 *ctx = (pragma121 *)malloc(sizeof(pragma121));
ctx->matrix_size = matrix_size;
ctx->block_size = block_size;
ctx->niter = niter;
ctx->dx = dx;
ctx->dy = dy;
ctx->error = error;
ctx->ii = ii;
ctx->i = i;
ctx->jj = jj;
ctx->j = j;
ctx->nx = nx;
ctx->ny = ny;
ctx->f = f;
ctx->u = u;
ctx->unew = unew;
ctx->params = params;
hclib_async(pragma121_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
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
static void pragma121_hclib_async(void *____arg) {
    pragma121 *ctx = (pragma121 *)____arg;
    int matrix_size; matrix_size = ctx->matrix_size;
    int block_size; block_size = ctx->block_size;
    int niter; niter = ctx->niter;
    double dx; dx = ctx->dx;
    double dy; dy = ctx->dy;
    double error; error = ctx->error;
    int ii; ii = ctx->ii;
    int i; i = ctx->i;
    int jj; jj = ctx->jj;
    int j; j = ctx->j;
    int nx; nx = ctx->nx;
    int ny; ny = ctx->ny;
    double *f; f = ctx->f;
    double *u; u = ctx->u;
    double *unew; unew = ctx->unew;
    struct user_parameters *params; params = ctx->params;
    hclib_start_finish();
for (jj=j; jj<j+block_size; ++jj) {
                    for (ii=i; ii<i+block_size; ++ii)
                    {
                        if (ii == 0 || ii == nx - 1 || jj == 0 || jj == ny - 1) {
                            (unew)[ii * ny + jj] = (f)[ii * ny + jj];
                        } else {
                            (unew)[ii * ny + jj] = 0.0;
                        }
                    }
                } ;     ; hclib_end_finish();
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
typedef struct _pragma230 {
    int i;
    int ii;
    int j;
    int jj;
    double x;
    double y;
    int nx;
    int ny;
    double *f;
    int block_size;
 } pragma230;

static void pragma230_hclib_async(void *____arg);
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
pragma230 *ctx = (pragma230 *)malloc(sizeof(pragma230));
ctx->i = i;
ctx->ii = ii;
ctx->j = j;
ctx->jj = jj;
ctx->x = x;
ctx->y = y;
ctx->nx = nx;
ctx->ny = ny;
ctx->f = f;
ctx->block_size = block_size;
hclib_async(pragma230_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } 
        }
    } ; hclib_end_finish(); 
} 
static void pragma230_hclib_async(void *____arg) {
    pragma230 *ctx = (pragma230 *)____arg;
    int i; i = ctx->i;
    int ii; ii = ctx->ii;
    int j; j = ctx->j;
    int jj; jj = ctx->jj;
    double x; x = ctx->x;
    double y; y = ctx->y;
    int nx; nx = ctx->nx;
    int ny; ny = ctx->ny;
    double *f; f = ctx->f;
    int block_size; block_size = ctx->block_size;
    hclib_start_finish();
for (jj=j; jj<j+block_size; ++jj)
            {
                y = (double) (jj) / (double) (ny - 1);
                for (ii=i; ii<i+block_size; ++ii)
                {
                    x = (double) (ii) / (double) (nx - 1);
                    if (ii == 0 || ii == nx - 1 || jj == 0 || jj == ny - 1)
                        (f)[ii * ny + jj] = u_exact(x, y);
                    else
                        (f)[ii * ny + jj] = - uxxyy_exact(x, y);
                }
            } ;     ; hclib_end_finish();
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

