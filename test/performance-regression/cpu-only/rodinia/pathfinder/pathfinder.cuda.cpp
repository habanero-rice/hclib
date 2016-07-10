#include <stdio.h>
__device__ inline int hclib_get_current_worker() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template<class functor_type>
__global__ void wrapper_kernel(unsigned niters, functor_type functor) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < niters) {
        functor(tid);
    }
}
template<class functor_type>
static void kernel_launcher(unsigned niters, functor_type functor) {
    const int threads_per_block = 256;
    const int nblocks = (niters + threads_per_block - 1) / threads_per_block;
    wrapper_kernel<<<nblocks, threads_per_block>>>(niters, functor);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Launch Error - %s\n", cudaGetErrorString(err));
        exit(2);
    }
}
#ifdef __cplusplus
#ifdef __CUDACC__
#endif
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "timer.h"

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

// #define BENCH_PRINT

int rows, cols;
int* data;
// int** wall;
int* result;
#define M_SEED 9

void
init(int argc, char** argv)
{
	if(argc==3){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
	}else{
                printf("Usage: pathfiner width num_of_steps\n");
                exit(0);
        }
	data = new int[rows*cols];
	// wall = new int*[rows];
	// for(int n=0; n<rows; n++)
	// 	wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // wall[i][j] = rand() % 10;
            data[i * cols + j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++) {
        // result[j] = wall[0][j];
        result[j] = data[0 * cols + j];
    }
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",data[i * cols + j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )

inline int MIN(int a, int b) { return ((a)<=(b) ? (a) : (b)); }

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

class pragma114_omp_parallel_hclib_async {
    private:
    int min;
    int* volatile src;
    volatile int cols;
    int* volatile dst;
    int* volatile data;
    volatile int t;

    public:
        pragma114_omp_parallel_hclib_async(int set_min,
                int* set_src,
                int set_cols,
                int* set_dst,
                int* set_data,
                int set_t) {
            min = set_min;
            src = set_src;
            cols = set_cols;
            dst = set_dst;
            data = set_data;
            t = set_t;

        }

        __device__ void operator()(int n) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
          min = src[n];
          if (n > 0) {
              min = src[n - 1] < min ? src[n - 1] : min;
          }
          if (n < cols-1) {
              min = src[n + 1] < min ? src[n + 1] : min;
          }
          dst[n] = data[t+1 * cols + n]+min;
        }
            }
        }
};

void run(int argc, char** argv)
{
    init(argc, argv);

    unsigned long long cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    pin_stats_reset();

for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
 { const int niters = (cols) - (0);
kernel_launcher(niters, pragma114_omp_parallel_hclib_async(min, src, cols, dst, data, t));
 } 
    }

    pin_stats_pause(cycles);
    pin_stats_dump(cycles);

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",dst[i]) ;
    printf("\n") ;
#endif

    delete [] data;
    // delete [] wall;
    delete [] dst;
    delete [] src;
} 

