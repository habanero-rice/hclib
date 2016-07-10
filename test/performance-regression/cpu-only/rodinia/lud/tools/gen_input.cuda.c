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

class pragma87_omp_parallel_hclib_async {
    private:
    int j;
    volatile int MatrixDim;
    FP_NUMBER sum;
    int k;
    FP_NUMBER* volatile L;
    int i;
    FP_NUMBER* volatile U;
    FP_NUMBER* volatile A;

    public:
        pragma87_omp_parallel_hclib_async(int set_j,
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
            L = set_L;
            i = set_i;
            U = set_U;
            A = set_A;

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
kernel_launcher(niters, pragma87_omp_parallel_hclib_async(j, MatrixDim, sum, k, L, i, U, A));
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
