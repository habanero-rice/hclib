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

void run(int argc, char** argv)
{
    init(argc, argv);

    unsigned long long cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    pin_stats_reset();

const unsigned long long full_program_start = current_time_ns();
for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
 { const unsigned long long parallel_for_start = current_time_ns();
#pragma omp parallel for private(min)
for(int n = 0; n < cols; n++){
          min = src[n];
          if (n > 0) {
              min = src[n - 1] < min ? src[n - 1] : min;
          }
          if (n < cols-1) {
              min = src[n + 1] < min ? src[n + 1] : min;
          }
          dst[n] = data[t+1 * cols + n]+min;
        } ; 
const unsigned long long parallel_for_end = current_time_ns();
printf("pragma105_omp_parallel %llu ns", parallel_for_end - parallel_for_start); } 

    } ; 
const unsigned long long full_program_end = current_time_ns();
printf("full_program %llu ns", full_program_end - full_program_start);


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

