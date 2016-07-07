#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
#include <stdio.h>
#include <omp.h>

extern int omp_num_threads;

#define BS 16

#define AA(_i,_j) a[offset*size+_i*size+_j+offset]
#define BB(_i,_j) a[_i*size+_j]

void lud_diagonal_omp (float* a, int size, int offset)
{
    int i, j, k;
    for (i = 0; i < BS; i++) {

        for (j = i; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(i,j) = AA(i,j) - AA(i,k) * AA(k,j);
            }
        }
   
        float temp = 1.f/AA(i,i);
        for (j = i+1; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(j,i) = AA(j,i) - AA(j,k) * AA(k,i);
            }
            AA(j,i) = AA(j,i)*temp;
        }
    }

}

// implements block LU factorization 
typedef struct _pragma61_omp_parallel {
    int (*offset_ptr);
    int chunk_idx;
    int (*chunks_in_inter_row_ptr);
    int (*chunks_per_inter_ptr);
    float (*a);
    int size;
 } pragma61_omp_parallel;

typedef struct _pragma113_omp_parallel {
    int (*offset_ptr);
    int chunk_idx;
    int (*chunks_in_inter_row_ptr);
    int (*chunks_per_inter_ptr);
    float (*a);
    int size;
 } pragma113_omp_parallel;


#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma61_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int chunk_idx) {
        }
};

#else
static void pragma61_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif

#ifdef OMP_TO_HCLIB_ENABLE_GPU

class pragma113_omp_parallel_hclib_async {
    private:

    public:
        __host__ __device__ void operator()(int chunk_idx) {
        }
};

#else
static void pragma113_omp_parallel_hclib_async(void *____arg, const int ___iter0);
#endif
typedef struct _main_entrypoint_ctx {
    float (*a);
    int size;
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    float (*a); a = ctx->a;
    int size; size = ctx->size;
{
    int offset, chunk_idx, size_inter, chunks_in_inter_row, chunks_per_inter;

    for (offset = 0; offset < size - BS ; offset += BS)
    {
        // lu factorization of left-top corner block diagonal matrix 
        //
        lud_diagonal_omp(a, size, offset);
            
        size_inter = size - offset -  BS;
        chunks_in_inter_row  = size_inter/BS;
        
        // calculate perimeter block matrices
        // 
 { 
pragma61_omp_parallel *new_ctx = (pragma61_omp_parallel *)malloc(sizeof(pragma61_omp_parallel));
new_ctx->offset_ptr = &(offset);
new_ctx->chunk_idx = chunk_idx;
new_ctx->chunks_in_inter_row_ptr = &(chunks_in_inter_row);
new_ctx->chunks_per_inter_ptr = &(chunks_per_inter);
new_ctx->a = a;
new_ctx->size = size;
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = chunks_in_inter_row;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((chunks_in_inter_row) - (0), pragma61_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma61_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 
        
        // update interior block matrices
        //
        chunks_per_inter = chunks_in_inter_row*chunks_in_inter_row;

 { 
pragma113_omp_parallel *new_ctx = (pragma113_omp_parallel *)malloc(sizeof(pragma113_omp_parallel));
new_ctx->offset_ptr = &(offset);
new_ctx->chunk_idx = chunk_idx;
new_ctx->chunks_in_inter_row_ptr = &(chunks_in_inter_row);
new_ctx->chunks_per_inter_ptr = &(chunks_per_inter);
new_ctx->a = a;
new_ctx->size = size;
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = chunks_per_inter;
domain[0].stride = 1;
domain[0].tile = -1;
#ifdef OMP_TO_HCLIB_ENABLE_GPU
hclib::future_t *fut = hclib::forasync_cuda((chunks_per_inter) - (0), pragma113_omp_parallel_hclib_async(), hclib::get_closest_gpu_locale(), NULL);
fut->wait();
#else
hclib_future_t *fut = hclib_forasync_future((void *)pragma113_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
#endif
free(new_ctx);
 } 
    }

    lud_diagonal_omp(a, size, offset);
    } ;     free(____arg);
}

void lud_omp(float *a, int size)
{
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->a = a;
new_ctx->size = size;
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);

}  

#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma61_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma61_omp_parallel *ctx = (pragma61_omp_parallel *)____arg;
    int chunk_idx; chunk_idx = ctx->chunk_idx;
    float (*a); a = ctx->a;
    int size; size = ctx->size;
    do {
    chunk_idx = ___iter0;
{
            int i, j, k, i_global, j_global, i_here, j_here;
            float sum;           
            float temp[BS*BS] __attribute__ ((aligned (64)));

            for (i = 0; i < BS; i++) {
for (j =0; j < BS; j++){
                    temp[i*BS + j] = a[size*(i + (*(ctx->offset_ptr))) + (*(ctx->offset_ptr)) + j ];
                }
            }
            i_global = (*(ctx->offset_ptr));
            j_global = (*(ctx->offset_ptr));
            
            // processing top perimeter
            //
            j_global += BS * (chunk_idx+1);
            for (j = 0; j < BS; j++) {
                for (i = 0; i < BS; i++) {
                    sum = 0.f;
                    for (k=0; k < i; k++) {
                        sum += temp[BS*i +k] * BB((i_global+k),(j_global+j));
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    BB(i_here, j_here) = BB(i_here,j_here) - sum;
                }
            }

            // processing left perimeter
            //
            j_global = (*(ctx->offset_ptr));
            i_global += BS * (chunk_idx + 1);
            for (i = 0; i < BS; i++) {
                for (j = 0; j < BS; j++) {
                    sum = 0.f;
                    for (k=0; k < j; k++) {
                        sum += BB((i_global+i),(j_global+k)) * temp[BS*k + j];
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    a[size*i_here + j_here] = ( a[size*i_here+j_here] - sum ) / a[size*((*(ctx->offset_ptr))+j) + (*(ctx->offset_ptr))+j];
                }
            }

        } ;     } while (0);
}

#endif


#ifndef OMP_TO_HCLIB_ENABLE_GPU

static void pragma113_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma113_omp_parallel *ctx = (pragma113_omp_parallel *)____arg;
    int chunk_idx; chunk_idx = ctx->chunk_idx;
    float (*a); a = ctx->a;
    int size; size = ctx->size;
    do {
    chunk_idx = ___iter0;
{
            int i, j, k, i_global, j_global;
            float temp_top[BS*BS] __attribute__ ((aligned (64)));
            float temp_left[BS*BS] __attribute__ ((aligned (64)));
            float sum[BS] __attribute__ ((aligned (64))) = {0.f};
            
            i_global = (*(ctx->offset_ptr)) + BS * (1 +  chunk_idx/(*(ctx->chunks_in_inter_row_ptr)));
            j_global = (*(ctx->offset_ptr)) + BS * (1 + chunk_idx%(*(ctx->chunks_in_inter_row_ptr)));

            for (i = 0; i < BS; i++) {
for (j =0; j < BS; j++){
                    temp_top[i*BS + j]  = a[size*(i + (*(ctx->offset_ptr))) + j + j_global ];
                    temp_left[i*BS + j] = a[size*(i + i_global) + (*(ctx->offset_ptr)) + j];
                }
            }

            for (i = 0; i < BS; i++)
            {
                for (k=0; k < BS; k++) {
for (j = 0; j < BS; j++) {
                        sum[j] += temp_left[BS*i + k] * temp_top[BS*k + j];
                    }
                }
for (j = 0; j < BS; j++) {
                    BB((i+i_global),(j+j_global)) -= sum[j];
                    sum[j] = 0.f;
                }
            }
        } ;     } while (0);
}

#endif

