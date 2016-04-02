#include "hclib.h"
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
typedef struct _pragma158 {
    float *pIn;
    float *tIn;
    float *tOut;
    int nx;
    int ny;
    int nz;
    float Cap;
    float Rx;
    float Ry;
    float Rz;
    float dt;
    int numiter;
    float ce;
    float cw;
    float cn;
    float cs;
    float ct;
    float cb;
    float cc;
    float stepDivCap;
    int count;
    float *tIn_t;
    float *tOut_t;
    int z;
 } pragma158;

static void pragma158_hclib_async(void *____arg, const int ___iter);
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
pragma158 *ctx = (pragma158 *)malloc(sizeof(pragma158));
ctx->pIn = pIn;
ctx->tIn = tIn;
ctx->tOut = tOut;
ctx->nx = nx;
ctx->ny = ny;
ctx->nz = nz;
ctx->Cap = Cap;
ctx->Rx = Rx;
ctx->Ry = Ry;
ctx->Rz = Rz;
ctx->dt = dt;
ctx->numiter = numiter;
ctx->ce = ce;
ctx->cw = cw;
ctx->cn = cn;
ctx->cs = cs;
ctx->ct = ct;
ctx->cb = cb;
ctx->cc = cc;
ctx->stepDivCap = stepDivCap;
ctx->count = count;
ctx->tIn_t = tIn_t;
ctx->tOut_t = tOut_t;
ctx->z = z;
hclib_loop_domain_t domain;
domain.low = 0;
domain.high = nz;
domain.stride = 1;
domain.tile = 1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma158_hclib_async, ctx, NULL, 1, &domain, FORASYNC_MODE_RECURSIVE);
hclib_future_wait(fut);
free(ctx);
 } 
            float *t = tIn_t;
            tIn_t = tOut_t;
            tOut_t = t; 
            count++;
        } while (count < numiter);
    } 
    return; 
} static void pragma158_hclib_async(void *____arg, const int ___iter) {
    pragma158 *ctx = (pragma158 *)____arg;
    float *pIn; pIn = ctx->pIn;
    float *tIn; tIn = ctx->tIn;
    float *tOut; tOut = ctx->tOut;
    int nx; nx = ctx->nx;
    int ny; ny = ctx->ny;
    int nz; nz = ctx->nz;
    float Cap; Cap = ctx->Cap;
    float Rx; Rx = ctx->Rx;
    float Ry; Ry = ctx->Ry;
    float Rz; Rz = ctx->Rz;
    float dt; dt = ctx->dt;
    int numiter; numiter = ctx->numiter;
    float ce; ce = ctx->ce;
    float cw; cw = ctx->cw;
    float cn; cn = ctx->cn;
    float cs; cs = ctx->cs;
    float ct; ct = ctx->ct;
    float cb; cb = ctx->cb;
    float cc; cc = ctx->cc;
    float stepDivCap; stepDivCap = ctx->stepDivCap;
    int count; count = ctx->count;
    float *tIn_t; tIn_t = ctx->tIn_t;
    float *tOut_t; tOut_t = ctx->tOut_t;
    int z; z = ctx->z;
    hclib_start_finish();
    do {
    z = ___iter;
{
                int y;
                for (y = 0; y < ny; y++) {
                    int x;
                    for (x = 0; x < nx; x++) {
                        int c, w, e, n, s, b, t;
                        c =  x + y * nx + z * nx * ny;
                        w = (x == 0)    ? c : c - 1;
                        e = (x == nx-1) ? c : c + 1;
                        n = (y == 0)    ? c : c - nx;
                        s = (y == ny-1) ? c : c + nx;
                        b = (z == 0)    ? c : c - nx * ny;
                        t = (z == nz-1) ? c : c + nx * ny;
                        tOut_t[c] = cc * tIn_t[c] + cw * tIn_t[w] + ce * tIn_t[e]
                            + cs * tIn_t[s] + cn * tIn_t[n] + cb * tIn_t[b] + ct * tIn_t[t]+(dt/Cap) * pIn[c] + ct*amb_temp;
                    }
                }
            } ;     } while (0);
    ; hclib_end_finish();
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
    int argc;
    char **argv;
    char *pfile;
    char *tfile;
    char *ofile;
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
    float *powerIn;
    float *tempOut;
    float *tempIn;
    float *tempCopy;
    int size;
    float *answer;
 } main_entrypoint_ctx;

static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int argc; argc = ctx->argc;
    char **argv; argv = ctx->argv;
    char *pfile; pfile = ctx->pfile;
    char *tfile; tfile = ctx->tfile;
    char *ofile; ofile = ctx->ofile;
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
    float *powerIn; powerIn = ctx->powerIn;
    float *tempOut; tempOut = ctx->tempOut;
    float *tempIn; tempIn = ctx->tempIn;
    float *tempCopy; tempCopy = ctx->tempCopy;
    int size; size = ctx->size;
    float *answer; answer = ctx->answer;
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
    } ; }

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

main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
ctx->argc = argc;
ctx->argv = argv;
ctx->pfile = pfile;
ctx->tfile = tfile;
ctx->ofile = ofile;
ctx->iterations = iterations;
ctx->numCols = numCols;
ctx->numRows = numRows;
ctx->layers = layers;
ctx->dx = dx;
ctx->dy = dy;
ctx->dz = dz;
ctx->Cap = Cap;
ctx->Rx = Rx;
ctx->Ry = Ry;
ctx->Rz = Rz;
ctx->max_slope = max_slope;
ctx->dt = dt;
ctx->powerIn = powerIn;
ctx->tempOut = tempOut;
ctx->tempIn = tempIn;
ctx->tempCopy = tempCopy;
ctx->size = size;
ctx->answer = answer;
hclib_launch(main_entrypoint, ctx);
free(ctx);

    writeoutput(tempOut,numRows, numCols, layers, ofile);
    free(tempIn);
    free(tempOut); free(powerIn);
    return 0;
} 	


