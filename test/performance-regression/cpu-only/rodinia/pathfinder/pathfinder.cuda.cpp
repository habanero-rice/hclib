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
static void kernel_launcher(const char *kernel_lbl, unsigned niters, functor_type functor) {
    const int threads_per_block = 256;
    const int nblocks = (niters + threads_per_block - 1) / threads_per_block;
    functor.transfer_to_device();
    const unsigned long long start = capp_current_time_ns();
    wrapper_kernel<<<nblocks, threads_per_block>>>(niters, functor);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error while synchronizing kernel - %s\n", cudaGetErrorString(err));
        exit(2);
    }
    const unsigned long long end = capp_current_time_ns();
    fprintf(stderr, "%s %llu ns\n", kernel_lbl, end - start);
    functor.transfer_from_device();
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
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    int min;
    int* volatile src;
    int* volatile h_src;
    volatile int cols;
    int* volatile dst;
    int* volatile h_dst;
    int* volatile data;
    int* volatile h_data;
    volatile int t;

    public:
        pragma114_omp_parallel_hclib_async(int set_min,
                int* set_src,
                int set_cols,
                int* set_dst,
                int* set_data,
                int set_t) {
            min = set_min;
            h_src = set_src;
            cols = set_cols;
            h_dst = set_dst;
            h_data = set_data;
            t = set_t;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        src = NULL;
        dst = NULL;
        data = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 3, h_src, h_dst, h_data);
        device_allocations = (void **)malloc(nallocations * sizeof(void *));
        for (i = 0; i < nallocations; i++) {
            err = cudaMalloc((void **)&device_allocations[i], host_allocation_sizes[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            err = cudaMemcpy((void *)device_allocations[i], (void *)host_allocations[i], host_allocation_sizes[i], cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            if (src == NULL && (char *)h_src >= (char *)host_allocations[i] && ((char *)h_src - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_src - (char *)host_allocations[i]);
                memcpy((void *)(&src), (void *)(&tmp), sizeof(void *));
            }
            if (dst == NULL && (char *)h_dst >= (char *)host_allocations[i] && ((char *)h_dst - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_dst - (char *)host_allocations[i]);
                memcpy((void *)(&dst), (void *)(&tmp), sizeof(void *));
            }
            if (data == NULL && (char *)h_data >= (char *)host_allocations[i] && ((char *)h_data - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_data - (char *)host_allocations[i]);
                memcpy((void *)(&data), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(src || h_src == NULL);
        assert(dst || h_dst == NULL);
        assert(data || h_data == NULL);

    }

    void transfer_from_device() {
        cudaError_t err;
        int i;
        for (i = 0; i < nallocations; i++) {
            err = cudaMemcpy((void *)host_allocations[i], (void *)device_allocations[i], host_allocation_sizes[i], cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
            err = cudaFree(device_allocations[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(3);
        }
        }
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
kernel_launcher("pragma114_omp_parallel", niters, pragma114_omp_parallel_hclib_async(min, src, cols, dst, data, t));
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

