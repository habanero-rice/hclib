#include <sys/time.h>
#include <time.h>
#include <stdio.h>
static unsigned long long current_time_ns() {
#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    unsigned long long s = 1000000000ULL * (unsigned long long)mts.tv_sec;
    return (unsigned long long)mts.tv_nsec + s;
#else
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (((unsigned long long)t.tv_nsec)) + s;
#endif
}
# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
void sweep (int nx, int ny, double dx, double dy, double *f,
        int itold, int itnew, double *u, double *unew, int block_size)
{
    int i;
    int it;
    int j;
    // double (*f)[nx][ny] = (double (*)[nx][ny])f_;
    // double (*u)[nx][ny] = (double (*)[nx][ny])u_;
    // double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;

#pragma omp parallel shared (f, u, unew) private (i, it, j) firstprivate(nx, ny, dx, dy, itold, itnew)
{
#pragma omp single 
{
        for (it = itold + 1; it <= itnew; it++) {
            // Save the current estimate.
            for (i = 0; i < nx; i++) {
#pragma omp task firstprivate(i, ny) private(j) shared(u, unew)
for (j = 0; j < ny; j++) {
                    (u)[i * ny + j] = (unew)[i * ny + j];
                }
            }
#pragma omp taskwait 
;
            // Compute a new estimate.
            for (i = 0; i < nx; i++) {
#pragma omp task firstprivate(i, dx, dy, nx, ny) private(j) shared(u, unew, f)
for (j = 0; j < ny; j++) {
                    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
                        (unew)[i * ny + j] = (f)[i * ny + j];
                    } else {
                        (unew)[i * ny + j] = 0.25 * ((u)[(i-1) * ny + j] + (u)[i * ny + (j+1)]
                                                + (u)[i * ny + (j-1)] + (u)[(i+1) * ny + j]
                                                + (f)[i * ny + j] * dx * dy);
                    }
                }
            }
#pragma omp taskwait 
;
        }
    }
    }
}
