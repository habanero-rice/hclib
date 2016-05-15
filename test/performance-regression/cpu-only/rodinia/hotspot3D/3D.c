#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#endif
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h> 
#include <math.h> 
#include <sys/time.h>
#include <string.h>

#define STR_SIZE (256)
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5


/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016; float chip_width = 0.016; 
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void fatal(char *s)
{
    fprintf(stderr, "Error: %s\n", s);
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {
    int i,j,k;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if( (fp  = fopen(file, "r" )) ==0 )
      fatal( "The file was not opened" );


    for (i=0; i <= grid_rows-1; i++) 
      for (j=0; j <= grid_cols-1; j++)
        for (k=0; k <= layers-1; k++)
          {
            if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
            if (feof(fp))
              fatal("not enough lines in file");
            if ((sscanf(str, "%f", &val) != 1))
              fatal("invalid file format");
            vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
          }

    fclose(fp);	

}


void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {

    int i,j,k, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
      printf( "The file was not opened\n" );

    for (i=0; i < grid_rows; i++) 
      for (j=0; j < grid_cols; j++)
        for (k=0; k < layers; k++)
          {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j+k*grid_rows*grid_cols]);
            fputs(str,fp);
            index++;
          }

    fclose(fp);	
}



void computeTempCPU(float *pIn, float* tIn, float *tOut, 
        int nx, int ny, int nz, float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    int c,w,e,n,s,b,t;
    int x,y,z;
    int i = 0;
    do{
        for(z = 0; z < nz; z++)
            for(y = 0; y < ny; y++)
                for(x = 0; x < nx; x++)
                {
                    c = x + y * nx + z * nx * ny;

                    w = (x == 0) ? c      : c - 1;
                    e = (x == nx - 1) ? c : c + 1;
                    n = (y == 0) ? c      : c - nx;
                    s = (y == ny - 1) ? c : c + nx;
                    b = (z == 0) ? c      : c - nx * ny;
                    t = (z == nz - 1) ? c : c + nx * ny;


                    tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t]*ct + tIn[b]*cb + (dt/Cap) * pIn[c] + ct*amb_temp;
                }
        float *temp = tIn;
        tIn = tOut;
        tOut = temp; 
        i++;
    }
    while(i < numiter);

}

float accuracy(float *arr1, float *arr2, int len)
{
    float err = 0.0; 
    int i;
    for(i = 0; i < len; i++)
    {
        err += (arr1[i]-arr2[i]) * (arr1[i]-arr2[i]);
    }

    return (float)sqrt(err/len);


}
typedef struct _pragma164_omp_parallel {
    int z;
    int (*count_ptr);
    float (*(*tIn_t_ptr));
    float (*(*tOut_t_ptr));
    float (*ce_ptr);
    float (*cw_ptr);
    float (*cn_ptr);
    float (*cs_ptr);
    float (*ct_ptr);
    float (*cb_ptr);
    float (*cc_ptr);
    float (*stepDivCap_ptr);
    float (*(*pIn_ptr));
    float (*(*tIn_ptr));
    float (*(*tOut_ptr));
    int (*nx_ptr);
    int (*ny_ptr);
    int (*nz_ptr);
    float (*Cap_ptr);
    float (*Rx_ptr);
    float (*Ry_ptr);
    float (*Rz_ptr);
    float (*dt_ptr);
    int (*numiter_ptr);
 } pragma164_omp_parallel;

static void pragma164_omp_parallel_hclib_async(void *____arg, const int ___iter0);
void computeTempOMP(float *pIn, float* tIn, float *tOut, 
        int nx, int ny, int nz, float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{  

    float ce, cw, cn, cs, ct, cb, cc;

    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);


    {
        int count = 0;
        float *tIn_t = tIn;
        float *tOut_t = tOut;

        do {
            int z; 
 { 
pragma164_omp_parallel *new_ctx = (pragma164_omp_parallel *)malloc(sizeof(pragma164_omp_parallel));
new_ctx->z = z;
new_ctx->count_ptr = &(count);
new_ctx->tIn_t_ptr = &(tIn_t);
new_ctx->tOut_t_ptr = &(tOut_t);
new_ctx->ce_ptr = &(ce);
new_ctx->cw_ptr = &(cw);
new_ctx->cn_ptr = &(cn);
new_ctx->cs_ptr = &(cs);
new_ctx->ct_ptr = &(ct);
new_ctx->cb_ptr = &(cb);
new_ctx->cc_ptr = &(cc);
new_ctx->stepDivCap_ptr = &(stepDivCap);
new_ctx->pIn_ptr = &(pIn);
new_ctx->tIn_ptr = &(tIn);
new_ctx->tOut_ptr = &(tOut);
new_ctx->nx_ptr = &(nx);
new_ctx->ny_ptr = &(ny);
new_ctx->nz_ptr = &(nz);
new_ctx->Cap_ptr = &(Cap);
new_ctx->Rx_ptr = &(Rx);
new_ctx->Ry_ptr = &(Ry);
new_ctx->Rz_ptr = &(Rz);
new_ctx->dt_ptr = &(dt);
new_ctx->numiter_ptr = &(numiter);
hclib_loop_domain_t domain[1];
domain[0].low = 0;
domain[0].high = nz;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma164_omp_parallel_hclib_async, new_ctx, NULL, 1, domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(new_ctx);
 } 
            float *t = tIn_t;
            tIn_t = tOut_t;
            tOut_t = t; 
            count++;
        } while (count < numiter);
    } 
    return; 
} 
static void pragma164_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma164_omp_parallel *ctx = (pragma164_omp_parallel *)____arg;
    int z; z = ctx->z;
    do {
    z = ___iter0;
{
                int y;
                for (y = 0; y < (*(ctx->ny_ptr)); y++) {
                    int x;
                    for (x = 0; x < (*(ctx->nx_ptr)); x++) {
                        int c, w, e, n, s, b, t;
                        c =  x + y * (*(ctx->nx_ptr)) + z * (*(ctx->nx_ptr)) * (*(ctx->ny_ptr));
                        w = (x == 0)    ? c : c - 1;
                        e = (x == (*(ctx->nx_ptr))-1) ? c : c + 1;
                        n = (y == 0)    ? c : c - (*(ctx->nx_ptr));
                        s = (y == (*(ctx->ny_ptr))-1) ? c : c + (*(ctx->nx_ptr));
                        b = (z == 0)    ? c : c - (*(ctx->nx_ptr)) * (*(ctx->ny_ptr));
                        t = (z == (*(ctx->nz_ptr))-1) ? c : c + (*(ctx->nx_ptr)) * (*(ctx->ny_ptr));
                        (*(ctx->tOut_t_ptr))[c] = (*(ctx->cc_ptr)) * (*(ctx->tIn_t_ptr))[c] + (*(ctx->cw_ptr)) * (*(ctx->tIn_t_ptr))[w] + (*(ctx->ce_ptr)) * (*(ctx->tIn_t_ptr))[e]
                            + (*(ctx->cs_ptr)) * (*(ctx->tIn_t_ptr))[s] + (*(ctx->cn_ptr)) * (*(ctx->tIn_t_ptr))[n] + (*(ctx->cb_ptr)) * (*(ctx->tIn_t_ptr))[b] + (*(ctx->ct_ptr)) * (*(ctx->tIn_t_ptr))[t]+((*(ctx->dt_ptr))/(*(ctx->Cap_ptr))) * (*(ctx->pIn_ptr))[c] + (*(ctx->ct_ptr))*amb_temp;
                    }
                }
            } ;     } while (0);
}

 

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
    fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
    fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

    fprintf(stderr, "\t<iteration> - number of iterations\n");
    fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
    fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<outputFile - output file\n");
    exit(1);
}



typedef struct _main_entrypoint_ctx {
    char (*pfile);
    char (*tfile);
    char (*ofile);
    int iterations;
    int numCols;
    int numRows;
    int layers;
    float dx;
    float dy;
    float dz;
    float Cap;
    float Rx;
    float Ry;
    float Rz;
    float max_slope;
    float dt;
    float (*powerIn);
    float (*tempOut);
    float (*tempIn);
    float (*tempCopy);
    int size;
    float (*answer);
    int argc;
    char (*(*argv));
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    char (*pfile); pfile = ctx->pfile;
    char (*tfile); tfile = ctx->tfile;
    char (*ofile); ofile = ctx->ofile;
    int iterations; iterations = ctx->iterations;
    int numCols; numCols = ctx->numCols;
    int numRows; numRows = ctx->numRows;
    int layers; layers = ctx->layers;
    float dx; dx = ctx->dx;
    float dy; dy = ctx->dy;
    float dz; dz = ctx->dz;
    float Cap; Cap = ctx->Cap;
    float Rx; Rx = ctx->Rx;
    float Ry; Ry = ctx->Ry;
    float Rz; Rz = ctx->Rz;
    float max_slope; max_slope = ctx->max_slope;
    float dt; dt = ctx->dt;
    float (*powerIn); powerIn = ctx->powerIn;
    float (*tempOut); tempOut = ctx->tempOut;
    float (*tempIn); tempIn = ctx->tempIn;
    float (*tempCopy); tempCopy = ctx->tempCopy;
    int size; size = ctx->size;
    float (*answer); answer = ctx->answer;
    int argc; argc = ctx->argc;
    char (*(*argv)); argv = ctx->argv;
{
    struct timeval start, stop;
    float time;
    gettimeofday(&start,NULL);
    computeTempOMP(powerIn, tempIn, tempOut, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);
    gettimeofday(&stop,NULL);
    time = (stop.tv_usec-start.tv_usec)*1.0e-6 + stop.tv_sec - start.tv_sec;
    computeTempCPU(powerIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);

    float acc = accuracy(tempOut,answer,numRows*numCols*layers);
    printf("Time: %.3f (s)\n",time);
    printf("Accuracy: %e\n",acc);
    } ;     free(____arg);
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        usage(argc,argv);
    }

    char *pfile, *tfile, *ofile;// *testFile;
    int iterations = atoi(argv[3]);

    pfile = argv[4];
    tfile = argv[5];
    ofile = argv[6];
    //testFile = argv[7];
    int numCols = atoi(argv[1]);
    int numRows = atoi(argv[1]);
    int layers = atoi(argv[2]);

    /* calculating parameters*/

    float dx = chip_height/numRows;
    float dy = chip_width/numCols;
    float dz = t_chip/layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    // cout << Rx << " " << Ry << " " << Rz << endl;
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;


    float *powerIn, *tempOut, *tempIn, *tempCopy;// *pCopy;
    //    float *d_powerIn, *d_tempIn, *d_tempOut;
    int size = numCols * numRows * layers;

    powerIn = (float*)calloc(size, sizeof(float));
    tempCopy = (float*)malloc(size * sizeof(float));
    tempIn = (float*)calloc(size,sizeof(float));
    tempOut = (float*)calloc(size, sizeof(float));
    //pCopy = (float*)calloc(size,sizeof(float));
    float* answer = (float*)calloc(size, sizeof(float));

    // outCopy = (float*)calloc(size, sizeof(float));
    readinput(powerIn,numRows, numCols, layers,pfile);
    readinput(tempIn, numRows, numCols, layers, tfile);

    memcpy(tempCopy,tempIn, size * sizeof(float));

main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->pfile = pfile;
new_ctx->tfile = tfile;
new_ctx->ofile = ofile;
new_ctx->iterations = iterations;
new_ctx->numCols = numCols;
new_ctx->numRows = numRows;
new_ctx->layers = layers;
new_ctx->dx = dx;
new_ctx->dy = dy;
new_ctx->dz = dz;
new_ctx->Cap = Cap;
new_ctx->Rx = Rx;
new_ctx->Ry = Ry;
new_ctx->Rz = Rz;
new_ctx->max_slope = max_slope;
new_ctx->dt = dt;
new_ctx->powerIn = powerIn;
new_ctx->tempOut = tempOut;
new_ctx->tempIn = tempIn;
new_ctx->tempCopy = tempCopy;
new_ctx->size = size;
new_ctx->answer = answer;
new_ctx->argc = argc;
new_ctx->argv = argv;
hclib_launch(main_entrypoint, new_ctx);

    writeoutput(tempOut,numRows, numCols, layers, ofile);
    free(tempIn);
    free(tempOut); free(powerIn);
    return 0;
} 	


