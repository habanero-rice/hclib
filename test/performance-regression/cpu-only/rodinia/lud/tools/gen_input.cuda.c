#include <stdio.h>
__device__ inline int hclib_get_current_worker() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template<class functor_type>
__global__ void wrapper_kernel(unsigned iter_offset, unsigned niters, functor_type functor) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < niters) {
        functor(iter_offset + tid);
    }
}
template<class functor_type>
static void kernel_launcher(const char *kernel_lbl, unsigned iter_offset, unsigned niters, functor_type functor) {
    const int threads_per_block = 256;
    const int nblocks = (niters + threads_per_block - 1) / threads_per_block;
    functor.transfer_to_device();
    const unsigned long long start = capp_current_time_ns();
    wrapper_kernel<<<nblocks, threads_per_block>>>(iter_offset, niters, functor);
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
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef FP_NUMBER
typedef double FP_NUMBER;
#else
typedef float FP_NUMBER;
#endif


#define GET_RAND_FP ((FP_NUMBER)rand()/((FP_NUMBER)(RAND_MAX)+(FP_NUMBER)(1)))
char L_FNAME[32], U_FNAME[32], A_FNAME[32];

class pragma79_omp_parallel_hclib_async {
    private:
        void **host_allocations;
        size_t *host_allocation_sizes;
        unsigned nallocations;
        void **device_allocations;
    int j;
    volatile int MatrixDim;
    FP_NUMBER sum;
    int k;
    FP_NUMBER* volatile L;
    FP_NUMBER* volatile h_L;
    int i;
    FP_NUMBER* volatile U;
    FP_NUMBER* volatile h_U;
    FP_NUMBER* volatile A;
    FP_NUMBER* volatile h_A;

    public:
        pragma79_omp_parallel_hclib_async(int set_j,
                int set_MatrixDim,
                FP_NUMBER set_sum,
                int set_k,
                FP_NUMBER* set_L,
                int set_i,
                FP_NUMBER* set_U,
                FP_NUMBER* set_A) {
            j = set_j;
            MatrixDim = set_MatrixDim;
            sum = set_sum;
            k = set_k;
            h_L = set_L;
            i = set_i;
            h_U = set_U;
            h_A = set_A;

        }

    void transfer_to_device() {
        int i;
        cudaError_t err;

        L = NULL;
        U = NULL;
        A = NULL;

        get_underlying_allocations(&host_allocations, &host_allocation_sizes, &nallocations, 3, h_L, h_U, h_A);
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
            if (L == NULL && (char *)h_L >= (char *)host_allocations[i] && ((char *)h_L - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_L - (char *)host_allocations[i]);
                memcpy((void *)(&L), (void *)(&tmp), sizeof(void *));
            }
            if (U == NULL && (char *)h_U >= (char *)host_allocations[i] && ((char *)h_U - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_U - (char *)host_allocations[i]);
                memcpy((void *)(&U), (void *)(&tmp), sizeof(void *));
            }
            if (A == NULL && (char *)h_A >= (char *)host_allocations[i] && ((char *)h_A - (char *)host_allocations[i]) < host_allocation_sizes[i]) {
                char *tmp = (char *)device_allocations[i] + ((char *)h_A - (char *)host_allocations[i]);
                memcpy((void *)(&A), (void *)(&tmp), sizeof(void *));
            }
        }

        assert(L || h_L == NULL);
        assert(U || h_U == NULL);
        assert(A || h_A == NULL);

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

        __device__ void operator()(int i) {
            for (int __dummy_iter = 0; __dummy_iter < 1; __dummy_iter++) {
                {
        for (j=0; j < MatrixDim; j++){
            sum = 0;
            for(k=0; k < MatrixDim; k++)
                sum += L[i * MatrixDim + k]*U[k * MatrixDim + j];
            A[i * MatrixDim + j] = sum;
        }
    }
            }
        }
};

int main (int argc, char **argv){
    int i,j,k,MatrixDim;
    FP_NUMBER sum, *L, *U, *A;
    FILE *fl,*fu,*fa;

    if ( argc < 2) {
        printf("./gen_input [Matrix_Dimension_size]\n");
        return 1;
    }

    MatrixDim = atoi(argv[1]);
    L = (FP_NUMBER *) malloc(sizeof(FP_NUMBER*)*MatrixDim*MatrixDim);
    U = (FP_NUMBER *) malloc(sizeof(FP_NUMBER*)*MatrixDim*MatrixDim);
    A = (FP_NUMBER *) malloc(sizeof(FP_NUMBER*)*MatrixDim*MatrixDim);

    if ( !L || !U || !A){
        printf("Can not allocate memory\n");
        if (L) free(L);
        if (U) free(U);
        if (A) free(A);
        return 1;
    }

    srand(time(NULL));

    sprintf(L_FNAME, "l-%d.dat", MatrixDim);
    fl = fopen(L_FNAME, "wb");
    if (fl == NULL) {
        printf("Cannot open file %s\n", L_FNAME);
        return 1;
    }

    sprintf(U_FNAME, "u-%d.dat", MatrixDim);
    fu = fopen(U_FNAME, "wb");
    if (fu == NULL) {
        printf("Cannot open file %s\n", U_FNAME);
        return 1;
    }

    sprintf(A_FNAME, "%d.dat", MatrixDim);
    fa = fopen(A_FNAME, "wb");
    if (!fa) {
        printf("Cannot open file %s\n", A_FNAME);
        return 1;
    }

for (i=0; i < MatrixDim; i ++){
        for (j=0; j < MatrixDim; j++){
            if ( i == j) {
                L[i * MatrixDim + j] = 1.0;
                U[i * MatrixDim + j] = GET_RAND_FP;
            } else if (i < j){
                L[i * MatrixDim + j] = 0;
                U[i * MatrixDim + j] = GET_RAND_FP;
            } else { // i > j
                L[i * MatrixDim + j] = GET_RAND_FP;
                U[i * MatrixDim + j] = 0;
            }
        }
    }

 { const int niters = (MatrixDim) - (0);
const int iters_offset = (0);
kernel_launcher("pragma79_omp_parallel", iters_offset, niters, pragma79_omp_parallel_hclib_async(j, MatrixDim, sum, k, L, i, U, A));
 } 

    for (i=0; i < MatrixDim; i ++) {
        for (j=0; j < MatrixDim; j++)
            fprintf(fl, "%f ", L[i * MatrixDim + j]);
        fprintf(fl, "\n");
    }
    fclose(fl);

    for (i=0; i < MatrixDim; i ++) {
        for (j=0; j < MatrixDim; j++)
            fprintf(fu, "%f ", U[i * MatrixDim + j]);
        fprintf(fu, "\n");
    }
    fclose(fu);

    fprintf(fa, "%d\n", MatrixDim);
    for (i=0; i < MatrixDim; i ++) {
        for (j=0; j < MatrixDim; j++)
            fprintf(fa, "%f ", A[i * MatrixDim + j]);
        fprintf(fa, "\n");
    }
    fclose(fa);

    free(L);
    free(U);
    free(A);

    return 0;
} 
