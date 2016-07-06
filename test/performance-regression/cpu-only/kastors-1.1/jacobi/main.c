#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#  include <omp.h>

#include "main.h"

#define min(a, b) ((a<b)?a:b)
#define max(a, b) ((a>b)?a:b)

void parse(int argc, char* argv[], struct user_parameters* params)
{
    int i;
    for(i=1; i<argc; i++) {
        if(!strcmp(argv[i], "-c"))
            params->check = 1;
        else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("----------------------------------------------\n");
            printf("-                KaStORS                     -\n");
            printf("-   Kaapi Starpu OpenMP Runtime task Suite   -\n");
            printf("----------------------------------------------\n");
            printf("-h, --help : Show help information\n");
            printf("-c : Ask to check result\n");
            printf("-i : Number of iterations\n");
#ifdef TITER
            printf("-r : Number ot timestep iteration\n");
#endif
#ifdef MSIZE
            printf("-n : Matrix size\n");
#endif
#ifdef SMSIZE
            printf("-m : SubMatrix size\n");
#endif
#ifdef BSIZE
            printf("-b : Block size\n");
#endif
#ifdef IBSIZE
            printf("-ib : Internal Block size\n");
#endif
#ifdef CUTOFF_SIZE
            printf("-s : Cutoff (Size of the matrix)\n");
#endif
#ifdef CUTOFF_DEPTH
            printf("-d : Cutoff (depth)\n");
#endif
            exit(EXIT_SUCCESS);
        } else if(!strcmp(argv[i], "-i")) {
            if (++i < argc)
                params->niter = atoi(argv[i]);
            else {
                fprintf(stderr, "-i requires a number\n");
                exit(EXIT_FAILURE);
            }
#ifdef TITER
        } else if(!strcmp(argv[i], "-r")) {
            if (++i < argc)
                params->titer = atoi(argv[i]);
            else {
                fprintf(stderr, "-r requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef MSIZE
        } else if(!strcmp(argv[i], "-n")) {
            if (++i < argc)
                params->matrix_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-n requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef SMSIZE
        } else if(!strcmp(argv[i], "-m")) {
            if (++i < argc)
                params->submatrix_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-m requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef BSIZE
        } else if(!strcmp(argv[i], "-b")) {
            if (++i < argc)
                params->blocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-b requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef IBSIZE
        } else if(!strcmp(argv[i], "-ib")) {
            if (++i < argc)
                params->iblocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-ib requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef CUTOFF_SIZE
        } else if(!strcmp(argv[i], "-s")) {
            if (++i < argc)
                params->cutoff_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-s requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef CUTOFF_DEPTH
        } else if(!strcmp(argv[i], "-d")) {
            if (++i < argc)
                params->cutoff_depth = atoi(argv[i]);
            else {
                fprintf(stderr, "-d requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
        } else
            fprintf(stderr, "Unknown parameter : %s\n", argv[i]);
    }
}

int comp (const void * elem1, const void * elem2) 
{
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

typedef struct _main_entrypoint_ctx {
    int num_threads;
    struct user_parameters params;
    int argc;
    char (*(*argv));
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int num_threads; num_threads = ctx->num_threads;
    struct user_parameters params; params = ctx->params;
    int argc; argc = ctx->argc;
    char (*(*argv)); argv = ctx->argv;
{
    // warmup
    run(&params);

    double mean = 0.0;
    double meansqr = 0.0;
    double min_ = DBL_MAX;
    double max_ = -1;
    double* all_times = (double*)malloc(sizeof(double) * params.niter); 

    for (int i=0; i<params.niter; ++i)
    {
      double cur_time = run(&params);
      all_times[i] = cur_time;
      mean += cur_time;
      min_ = min(min_, cur_time);
      max_ = max(max_, cur_time);
      meansqr += cur_time * cur_time;
      }
    mean /= params.niter;
    meansqr /= params.niter;
    double stddev = sqrt(meansqr - mean * mean);

    qsort(all_times, params.niter, sizeof(double), comp);
    double median = all_times[params.niter / 2];

    free(all_times);

    printf("Program : %s\n", argv[0]);
#ifdef MSIZE
    printf("Size : %d\n", params.matrix_size);
#endif
#ifdef SMSIZE
    printf("Submatrix size : %d\n", params.submatrix_size);
#endif
#ifdef BSIZE
    printf("Blocksize : %d\n", params.blocksize);
#endif
#ifdef IBSIZE
    printf("Internal Blocksize : %d\n", params.iblocksize);
#endif
#ifdef TITER
    printf("Iteration time : %d\n", params.titer);
#endif
    printf("Iterations : %d\n", params.niter);
#ifdef CUTOFF_SIZE
    printf("Cutoff Size : %d\n", params.cutoff_size);
#endif
#ifdef CUTOFF_DEPTH
    printf("Cutoff depth : %d\n", params.cutoff_depth);
#endif
    printf("Threads : %d\n", num_threads);
#ifdef GFLOPS
    printf("Gflops:: ");
#else
    printf("Time(sec):: ");
#endif
    printf("avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median : %lf\n",
           mean, stddev, min_, max_, median);
    if(params.check)
        printf("Check : %s\n", (params.succeed)?
                ((params.succeed > 1)?"not implemented":"success")
                :"fail");
    if (params.string2display !=0)
      printf("%s", params.string2display);
    printf("\n");
    } ;     free(____arg);
}

int main(int argc, char* argv[])
{
    int num_threads = 1;
    struct user_parameters params;
    memset(&params, 0, sizeof(params));

    /* default value */
    params.niter = 1;

    parse(argc, argv, &params);

// get Number of thread if OpenMP is activated

main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->num_threads = num_threads;
new_ctx->params = params;
new_ctx->argc = argc;
new_ctx->argv = argv;
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);


    return 0;
} 
