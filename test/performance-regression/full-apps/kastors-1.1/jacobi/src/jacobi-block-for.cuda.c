#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
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
for (block_x = 0; block_x < max_blocks_x; block_x++)
            for (block_y = 0; block_y < max_blocks_y; block_y++)
                copy_block(nx, ny, block_x, block_y, u_, unew_, block_size);

for (block_x = 0; block_x < max_blocks_x; block_x++)
            for (block_y = 0; block_y < max_blocks_y; block_y++)
                compute_estimate(block_x, block_y, u_, unew_, f_, dx, dy,
                                 nx, ny, block_size);
    }
}
