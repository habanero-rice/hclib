#ifndef KASTORS_POISSON_H
#define KASTORS_POISSON_H
# include <assert.h>

/******************************************************************************/
/*
  Purpose:

  SWEEP carries out one step of the Jacobi iteration.

  Discussion:

  Assuming DX = DY, we can approximate

  - (d/dx d/dx + d/dy d/dy) U(X,Y)

  by

  (U(i-1,j) + U(i+1,j) + U(i,j-1) + U(i,j+1) - 4*U(i,j)) / dx / dy

  The discretization employed below will not be correct in the general
  case where DX and DY are not equal.  It's only a little more complicated
  to allow DX and DY to be different, but we're not going to worry about
  that right now.

  Licensing:

  This code is distributed under the GNU LGPL license.

  Modified:

  14 December 2011

  Author:

  John Burkardt

  Parameters:

  Input, int NX, NY, the X and Y grid dimensions.

  Input, double DX, DY, the spacing between grid points.

  Input, double F[NX][NY], the right hand side data.

  Input, int ITOLD, the iteration index on input.

  Input, int ITNEW, the desired iteration index
  on output.

  Input, double U[NX][NY], the solution estimate on
  iteration ITNEW-1.

  Input/output, double UNEW[NX][NY], on input, the solution
  estimate on iteration ITOLD.  On output, the solution estimate on
  iteration ITNEW.
*/
void sweep(int nx, int ny, double dx, double dy, double *f,
	   int itold, int itnew, double *u, double *unew, int block_size);
void sweep_seq(int nx, int ny, double dx, double dy, double *f,
	   int itold, int itnew, double *u, double *unew);


static inline void copy_block(int nx, int ny, int block_x, int block_y, double *u_, double *unew_, int block_size) {
    int i, j, start_i, start_j;
    double (*u)[nx][ny] = (double (*)[nx][ny])u_;
    double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;
    start_i = block_x * block_size;
    start_j = block_y * block_size;
    for (i = start_i; i < start_i + block_size; i++) {
        for (j = start_j; j < start_j + block_size; j++) {
            assert((i < nx) && (j < ny));
            (*u)[i][j] = (*unew)[i][j];
        }
    }
}

static inline void compute_estimate(int block_x, int block_y, double *u_,
                                    double *unew_, double *f_, double dx,
                                    double dy, int nx, int ny, int block_size) {
    int i, j, start_i, start_j;
    double (*f)[nx][ny] = (double (*)[nx][ny])f_;
    double (*u)[nx][ny] = (double (*)[nx][ny])u_;
    double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;
    start_i = block_x * block_size;
    start_j = block_y * block_size;
    for (i = start_i; i < start_i + block_size; i++) {
        for (j = start_j; j < start_j + block_size; j++) {
            if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
                (*unew)[i][j] = (*f)[i][j];
            } else {
                (*unew)[i][j] = 0.25 * ((*u)[i-1][j] + (*u)[i][j+1]
                                      + (*u)[i][j-1] + (*u)[i+1][j]
                                      + (*f)[i][j] * dx * dy);
            }
        }
    }
}
#endif
