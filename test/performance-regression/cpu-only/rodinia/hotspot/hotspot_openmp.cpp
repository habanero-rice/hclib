#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

// Returns the current system time in microseconds 
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;

}

using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE	256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
#define OPEN
//#define NUM_THREAD 4

typedef float FLOAT;

/* chip parameters	*/
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* ambient temperature, assuming no package at all	*/
const FLOAT amb_temp = 80.0;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
typedef struct _pragma69_omp_parallel {
    float delta;
    int r;
    int c;
    int chunk;
    int num_chunk;
    int chunks_in_row;
    int (*chunks_in_col_ptr);
    float (*(*result_ptr));
    float (*(*temp_ptr));
    float (*(*power_ptr));
    int row;
    int col;
    float (*Cap_1_ptr);
    float (*Rx_1_ptr);
    float (*Ry_1_ptr);
    float (*Rz_1_ptr);
    float (*step_ptr);
 } pragma69_omp_parallel;

static void pragma69_omp_parallel_hclib_async(void *____arg, const int ___iter0);
void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int row, int col,
					  FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1, 
					  FLOAT step)
{
    FLOAT delta;
    int r, c;
    int chunk;
    int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col/BLOCK_SIZE_C;
    int chunks_in_col = row/BLOCK_SIZE_R;

	// omp_set_num_threads(num_omp_threads);
 { 
pragma69_omp_parallel *new_ctx = (pragma69_omp_parallel *)malloc(sizeof(pragma69_omp_parallel));
new_ctx->delta = delta;
new_ctx->r = r;
new_ctx->c = c;
new_ctx->chunk = chunk;
new_ctx->num_chunk = num_chunk;
new_ctx->chunks_in_row = chunks_in_row;
new_ctx->chunks_in_col_ptr = &(chunks_in_col);
new_ctx->result_ptr = &(result);
new_ctx->temp_ptr = &(temp);
new_ctx->power_ptr = &(power);
new_ctx->row = row;
new_ctx->col = col;
new_ctx->Cap_1_ptr = &(Cap_1);
new_ctx->Rx_1_ptr = &(Rx_1);
new_ctx->Ry_1_ptr = &(Ry_1);
new_ctx->Rz_1_ptr = &(Rz_1);
new_ctx->step_ptr = &(step);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = num_chunk;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma69_omp_parallel_hclib_async, new_ctx, 1, domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(new_ctx);
 } 
} 
static void pragma69_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma69_omp_parallel *ctx = (pragma69_omp_parallel *)____arg;
    float delta; delta = ctx->delta;
    int r; r = ctx->r;
    int c; c = ctx->c;
    int chunk; chunk = ctx->chunk;
    int num_chunk; num_chunk = ctx->num_chunk;
    int chunks_in_row; chunks_in_row = ctx->chunks_in_row;
    int row; row = ctx->row;
    int col; col = ctx->col;
    do {
    chunk = ___iter0;
{
        int r_start = BLOCK_SIZE_R*(chunk/(*(ctx->chunks_in_col_ptr)));
        int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
        int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
        int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
       
        if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
        {
            for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
                for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
                    /* Corner 1 */
                    if ( (r == 0) && (c == 0) ) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[0] +
                            ((*(ctx->temp_ptr))[1] - (*(ctx->temp_ptr))[0]) * (*(ctx->Rx_1_ptr)) +
                            ((*(ctx->temp_ptr))[col] - (*(ctx->temp_ptr))[0]) * (*(ctx->Ry_1_ptr)) +
                            (amb_temp - (*(ctx->temp_ptr))[0]) * (*(ctx->Rz_1_ptr)));
                    }	/* Corner 2 */
                    else if ((r == 0) && (c == col-1)) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[c] +
                            ((*(ctx->temp_ptr))[c-1] - (*(ctx->temp_ptr))[c]) * (*(ctx->Rx_1_ptr)) +
                            ((*(ctx->temp_ptr))[c+col] - (*(ctx->temp_ptr))[c]) * (*(ctx->Ry_1_ptr)) +
                        (   amb_temp - (*(ctx->temp_ptr))[c]) * (*(ctx->Rz_1_ptr)));
                    }	/* Corner 3 */
                    else if ((r == row-1) && (c == col-1)) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[r*col+c] + 
                            ((*(ctx->temp_ptr))[r*col+c-1] - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rx_1_ptr)) + 
                            ((*(ctx->temp_ptr))[(r-1)*col+c] - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Ry_1_ptr)) + 
                        (   amb_temp - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rz_1_ptr)));					
                    }	/* Corner 4	*/
                    else if ((r == row-1) && (c == 0)) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[r*col] + 
                            ((*(ctx->temp_ptr))[r*col+1] - (*(ctx->temp_ptr))[r*col]) * (*(ctx->Rx_1_ptr)) + 
                            ((*(ctx->temp_ptr))[(r-1)*col] - (*(ctx->temp_ptr))[r*col]) * (*(ctx->Ry_1_ptr)) + 
                            (amb_temp - (*(ctx->temp_ptr))[r*col]) * (*(ctx->Rz_1_ptr)));
                    }	/* Edge 1 */
                    else if (r == 0) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[c] + 
                            ((*(ctx->temp_ptr))[c+1] + (*(ctx->temp_ptr))[c-1] - 2.0*(*(ctx->temp_ptr))[c]) * (*(ctx->Rx_1_ptr)) + 
                            ((*(ctx->temp_ptr))[col+c] - (*(ctx->temp_ptr))[c]) * (*(ctx->Ry_1_ptr)) + 
                            (amb_temp - (*(ctx->temp_ptr))[c]) * (*(ctx->Rz_1_ptr)));
                    }	/* Edge 2 */
                    else if (c == col-1) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[r*col+c] + 
                            ((*(ctx->temp_ptr))[(r+1)*col+c] + (*(ctx->temp_ptr))[(r-1)*col+c] - 2.0*(*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Ry_1_ptr)) + 
                            ((*(ctx->temp_ptr))[r*col+c-1] - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rx_1_ptr)) + 
                            (amb_temp - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rz_1_ptr)));
                    }	/* Edge 3 */
                    else if (r == row-1) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[r*col+c] + 
                            ((*(ctx->temp_ptr))[r*col+c+1] + (*(ctx->temp_ptr))[r*col+c-1] - 2.0*(*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rx_1_ptr)) + 
                            ((*(ctx->temp_ptr))[(r-1)*col+c] - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Ry_1_ptr)) + 
                            (amb_temp - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rz_1_ptr)));
                    }	/* Edge 4 */
                    else if (c == 0) {
                        delta = ((*(ctx->Cap_1_ptr))) * ((*(ctx->power_ptr))[r*col] + 
                            ((*(ctx->temp_ptr))[(r+1)*col] + (*(ctx->temp_ptr))[(r-1)*col] - 2.0*(*(ctx->temp_ptr))[r*col]) * (*(ctx->Ry_1_ptr)) + 
                            ((*(ctx->temp_ptr))[r*col+1] - (*(ctx->temp_ptr))[r*col]) * (*(ctx->Rx_1_ptr)) + 
                            (amb_temp - (*(ctx->temp_ptr))[r*col]) * (*(ctx->Rz_1_ptr)));
                    }
                    (*(ctx->result_ptr))[r*col+c] =(*(ctx->temp_ptr))[r*col+c]+ delta;
                }
            }
            continue;
        }

        for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
            /* Update Temperatures */
                (*(ctx->result_ptr))[r*col+c] =(*(ctx->temp_ptr))[r*col+c]+ 
                     ( (*(ctx->Cap_1_ptr)) * ((*(ctx->power_ptr))[r*col+c] + 
                    ((*(ctx->temp_ptr))[(r+1)*col+c] + (*(ctx->temp_ptr))[(r-1)*col+c] - 2.f*(*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Ry_1_ptr)) + 
                    ((*(ctx->temp_ptr))[r*col+c+1] + (*(ctx->temp_ptr))[r*col+c-1] - 2.f*(*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rx_1_ptr)) + 
                    (amb_temp - (*(ctx->temp_ptr))[r*col+c]) * (*(ctx->Rz_1_ptr))));
            }
        }
    } ;     } while (0);
}



/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int row, int col) 
{
	#ifdef VERBOSE
	int i = 0;
	#endif

	FLOAT grid_height = chip_height / row;
	FLOAT grid_width = chip_width / col;

	FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

	FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1=1.f/Rx;
    FLOAT Ry_1=1.f/Ry;
    FLOAT Rz_1=1.f/Rz;
    FLOAT Cap_1 = step/Cap;
	#ifdef VERBOSE
	fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
	fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
	#endif

        {
            FLOAT* r = result;
            FLOAT* t = temp;
            for (int i = 0; i < num_iterations ; i++)
            {
                #ifdef VERBOSE
                fprintf(stdout, "iteration %d\n", i++);
                #endif
                single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
                FLOAT* tmp = t;
                t = r;
                r = tmp;
            }	
        }
	#ifdef VERBOSE
	fprintf(stdout, "iteration %d\n", i++);
	#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file) {

    int i,j, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
        printf( "The file was not opened\n" );


    for (i=0; i < grid_rows; i++) 
        for (j=0; j < grid_cols; j++)
        {

            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str,fp);
            index++;
        }

    fclose(fp);	
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file)
{
  	int i, index;
	FILE *fp;
	char str[STR_SIZE];
	FLOAT val;

	fp = fopen (file, "r");
	if (!fp)
		fatal ("file could not be opened for reading");

	for (i=0; i < grid_rows * grid_cols; i++) {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		if ((sscanf(str, "%f", &val) != 1) )
			fatal("invalid file format");
		vect[i] = val;
	}

	fclose(fp);	
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
	fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<no. of threads>   - number of threads\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
        fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

typedef struct _main_entrypoint_ctx {
    int grid_rows;
    int grid_cols;
    int sim_time;
    int i;
    float (*temp);
    float (*power);
    float (*result);
    char (*tfile);
    char (*pfile);
    char (*ofile);
    long long start_time;
    int argc;
    char (*(*argv));
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int grid_rows; grid_rows = ctx->grid_rows;
    int grid_cols; grid_cols = ctx->grid_cols;
    int sim_time; sim_time = ctx->sim_time;
    int i; i = ctx->i;
    float (*temp); temp = ctx->temp;
    float (*power); power = ctx->power;
    float (*result); result = ctx->result;
    char (*tfile); tfile = ctx->tfile;
    char (*pfile); pfile = ctx->pfile;
    char (*ofile); ofile = ctx->ofile;
    long long start_time; start_time = ctx->start_time;
    int argc; argc = ctx->argc;
    char (*(*argv)); argv = ctx->argv;
compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols) ;     free(____arg);
}

int main(int argc, char **argv)
{
	int grid_rows, grid_cols, sim_time, i;
	FLOAT *temp, *power, *result;
	char *tfile, *pfile, *ofile;
	
	/* check validity of inputs	*/
	if (argc != 8)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 ||
		(grid_cols = atoi(argv[2])) <= 0 ||
		(sim_time = atoi(argv[3])) <= 0 || 
		(num_omp_threads = atoi(argv[4])) <= 0
		)
		usage(argc, argv);

	/* allocate memory for the temperature and power arrays	*/
	temp = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	power = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	result = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	if(!temp || !power)
		fatal("unable to allocate memory");

	/* read initial temperatures and input power	*/
	tfile = argv[5];
	pfile = argv[6];
    ofile = argv[7];

	read_input(temp, grid_rows, grid_cols, tfile);
	read_input(power, grid_rows, grid_cols, pfile);

	printf("Start computing the transient temperature\n");
	
    long long start_time = get_time();

main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->grid_rows = grid_rows;
new_ctx->grid_cols = grid_cols;
new_ctx->sim_time = sim_time;
new_ctx->i = i;
new_ctx->temp = temp;
new_ctx->power = power;
new_ctx->result = result;
new_ctx->tfile = tfile;
new_ctx->pfile = pfile;
new_ctx->ofile = ofile;
new_ctx->start_time = start_time;
new_ctx->argc = argc;
new_ctx->argv = argv;
hclib_launch(main_entrypoint, new_ctx);
;

    long long end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));

    writeoutput((1&sim_time) ? result : temp, grid_rows, grid_cols, ofile);

	/* output results	*/
#ifdef VERBOSE
	fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
	for(i=0; i < grid_rows * grid_cols; i++)
	fprintf(stdout, "%d\t%g\n", i, temp[i]);
#endif
	/* cleanup	*/
	free(temp);
	free(power);

	return 0;
} 
/* vim: set ts=4 sw=4  sts=4 et si ai: */
