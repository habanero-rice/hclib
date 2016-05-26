#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
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
int** wall;
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
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
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

typedef struct _pragma108_omp_parallel {
    int (*t_ptr);
    unsigned long long (*cycles_ptr);
    int (*(*src_ptr));
    int (*(*dst_ptr));
    int (*(*temp_ptr));
    int min;
    int (*argc_ptr);
    char (*(*(*argv_ptr)));
 } pragma108_omp_parallel;

static void pragma108_omp_parallel_hclib_async(void *____arg, const int ___iter0);
typedef struct _main_entrypoint_ctx {
    unsigned long long cycles;
    int (*src);
    int (*dst);
    int (*temp);
    int min;
    int argc;
    char (*(*argv));
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    unsigned long long cycles; cycles = ctx->cycles;
    int (*src); src = ctx->src;
    int (*dst); dst = ctx->dst;
    int (*temp); temp = ctx->temp;
    int min; min = ctx->min;
    int argc; argc = ctx->argc;
    char (*(*argv)); argv = ctx->argv;
for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
 { 
pragma108_omp_parallel *new_ctx = (pragma108_omp_parallel *)malloc(sizeof(pragma108_omp_parallel));
new_ctx->t_ptr = &(t);
new_ctx->cycles_ptr = &(cycles);
new_ctx->src_ptr = &(src);
new_ctx->dst_ptr = &(dst);
new_ctx->temp_ptr = &(temp);
new_ctx->min = min;
new_ctx->argc_ptr = &(argc);
new_ctx->argv_ptr = &(argv);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = cols;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma108_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
free(new_ctx);
 } 
    } ;     free(____arg);
}

void run(int argc, char** argv)
{
    init(argc, argv);

    unsigned long long cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    pin_stats_reset();

main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->cycles = cycles;
new_ctx->src = src;
new_ctx->dst = dst;
new_ctx->temp = temp;
new_ctx->min = min;
new_ctx->argc = argc;
new_ctx->argv = argv;
hclib_launch(main_entrypoint, new_ctx);


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
    delete [] wall;
    delete [] dst;
    delete [] src;
}  
static void pragma108_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma108_omp_parallel *ctx = (pragma108_omp_parallel *)____arg;
    int min; min = ctx->min;
    do {
    int n;     n = ___iter0;
{
          min = (*(ctx->src_ptr))[n];
          if (n > 0) {
              min = (*(ctx->src_ptr))[n - 1] < min ? (*(ctx->src_ptr))[n - 1] : min;
          }
          if (n < cols-1) {
              min = (*(ctx->src_ptr))[n + 1] < min ? (*(ctx->src_ptr))[n + 1] : min;
          }
          (*(ctx->dst_ptr))[n] = wall[(*(ctx->t_ptr))+1][n]+min;
        } ;     } while (0);
}



