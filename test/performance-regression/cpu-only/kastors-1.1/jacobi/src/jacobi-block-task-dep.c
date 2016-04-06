# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
void sweep (int nx, int ny, double dx, double dy, double *f_,
        int itold, int itnew, double *u_, double *unew_, int block_size)
{
    int it;
#ifdef _OPENMP
    double (*f)[nx][ny] = (double (*)[nx][ny])f_;
    double (*u)[nx][ny] = (double (*)[nx][ny])u_;
    double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;
#endif
    int block_x, block_y;

    if (block_size == 0)
        block_size = nx;

    int max_blocks_x = (nx / block_size);
    int max_blocks_y = (ny / block_size);

#pragma omp parallel \
    shared(u_, unew_, f, max_blocks_x, max_blocks_y, nx, ny, dx, dy, itold, itnew, block_size) \
    private(it, block_x, block_y)
#pragma omp single
    {
        for (it = itold + 1; it <= itnew; it++) {
            // Save the current estimate.
            for (block_x = 0; block_x < max_blocks_x; block_x++) {
                for (block_y = 0; block_y < max_blocks_y; block_y++) {
#pragma omp task shared(u_, unew_, block_size, nx, ny) firstprivate(block_x, block_y) \
                    depend(in: unew[block_x * block_size: block_size][block_y * block_size: block_size]) \
                    depend(out: u[block_x * block_size: block_size][block_y * block_size: block_size])
                    copy_block(nx, ny, block_x, block_y, u_, unew_, block_size);
                }
            }

            // Compute a new estimate.
            for (block_x = 0; block_x < max_blocks_x; block_x++) {
                for (block_y = 0; block_y < max_blocks_y; block_y++) {
                    int xdm1 = block_x == 0 ? 0 : 1;
                    int xdp1 = block_x == max_blocks_x-1 ? 0 : +1;
                    int ydp1 = block_y == max_blocks_y-1 ? 0 : +1;
                    int ydm1 = block_y == 0 ? 0 : 1;
#pragma omp task shared(u_, unew_, f_, dx, dy, nx, ny, block_size) firstprivate(block_x, block_y, xdm1, xdp1, ydp1, ydm1) \
                    depend(out: unew[block_x * block_size: block_size][block_y * block_size: block_size]) \
                    depend(in: f[block_x * block_size: block_size][block_y * block_size: block_size], \
                            u[block_x * block_size: block_size][block_y * block_size: block_size], \
                            u[(block_x - xdm1) * block_size: block_size][block_y * block_size: block_size], \
                            u[block_x * block_size: block_size][(block_y + ydp1)* block_size: block_size], \
                            u[block_x * block_size: block_size][(block_y - ydm1)* block_size: block_size], \
                            u[(block_x + xdp1)* block_size: block_size][block_y * block_size: block_size])
                    compute_estimate(block_x, block_y, u_, unew_, f_, dx, dy,
                                     nx, ny, block_size);
                }
            }
        }
    }
}
