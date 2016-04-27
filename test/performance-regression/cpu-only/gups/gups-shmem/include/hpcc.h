/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; fill-column: 79; coding: iso-latin-1-unix -*- */

#ifndef HPCC_H
#define HPCC_H 1

/* HPL includes:
stdio.h
stdlib.h
string.h
stdarg.h
vararg.h (if necessary)
*/

#include <hpccver.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>

#define SHMEMFFT_TIMING_COUNT 8

/* Define 64-bit types and corresponding format strings for printf() and constants */
#ifdef LONG_IS_64BITS
typedef unsigned long u64Int;
typedef long s64Int;
#define FSTR64 "%ld"
#define FSTRU64 "%lu"
#define ZERO64B 0L
#else
typedef unsigned long long u64Int;
typedef long long s64Int;
#define FSTR64 "%lld"
#define FSTRU64 "%llu"
#define ZERO64B 0LL
#endif

typedef struct {
  double GBs, time, residual;
  int n, nb, nprow, npcol;
} PTRANS_RuntimeData;

/* parameters of execution */
typedef struct {

  /* HPCC section */
  char inFname[256 + 1], outFname[256 + 1];
  double SHMEMGUPs, StarGUPs, SingleGUPs,
    SHMEMRandomAccess_ErrorsFraction, SHMEMRandomAccess_time, SHMEMRandomAccess_CheckTime,
    SHMEMRandomAccess_TimeBound,
    StarStreamCopyGBs, StarStreamScaleGBs,
    StarStreamAddGBs, StarStreamTriadGBs, SingleStreamCopyGBs, SingleStreamScaleGBs,
    SingleStreamAddGBs, SingleStreamTriadGBs, StarDGEMMGflops, SingleDGEMMGflops;
  double StarFFTGflops, SingleFFTGflops, SHMEMFFTGflops, SHMEMFFT_maxErr;
  double MaxPingPongLatency, RandomlyOrderedRingLatency, MinPingPongBandwidth,
    NaturallyOrderedRingBandwidth, RandomlyOrderedRingBandwidth,
    MinPingPongLatency, AvgPingPongLatency, MaxPingPongBandwidth, AvgPingPongBandwidth,
    NaturallyOrderedRingLatency;
  int DGEMM_N;
  int StreamThreads, StreamVectorSize;
  int FFT_N;
  int SHMEMFFT_Procs;
  int SHMEMRandomAccess_Algorithm;

  int Failure; /* over all failure of the benchmark */

  double SHMEMFFTtimingsForward[SHMEMFFT_TIMING_COUNT], SHMEMFFTtimingsBackward[SHMEMFFT_TIMING_COUNT];

  size_t HPLMaxProcMem;
  int HPLMaxProc, HPLMinProc;
  int RunHPL, RunStarDGEMM, RunSingleDGEMM, RunPTRANS, RunStarStream,
    RunSingleStream, RunSHMEMRandomAccess, RunStarRandomAccess,
    RunSingleRandomAccess, RunLatencyBandwidth, RunStarFFT, RunSingleFFT, RunSHMEMFFT;

  int FFTEnblk, FFTEnp, FFTEl2size;
  s64Int RandomAccess_N, SHMEMRandomAccess_ExeUpdates, SHMEMRandomAccess_N, SHMEMRandomAccess_Errors, SHMEMFFT_N;
} HPCC_Params;
/*
This is what needs to be done to add a new benchmark:
-  Add the benchmark code to the directory structure (headers, makefiles)
-  Add benchmark output data to the HPCC_Params structure.
-  Initialize the HPCC_Params structure data in HPCC_Init().
-  Add a call to the benchmark function in main().
-  Make sure that all the processes fill out the structure with the same data.
-  Print the output of the benchmark in HPCC_Finalize().
-  For tests that have "Star" and "Single" variants (DGEMM, RandomAccess, STREAM) the function
that performs the test returns a value (0 or 1) that indicates runtime failure and also returns
benchamark failure (due to wrong optimization that causes numerical error) by setting
params->Failure.
*/

extern int HPCC_Init(HPCC_Params *params);
extern int HPCC_Finalize(HPCC_Params *params);
extern int HPCC_LocalVectorSize(HPCC_Params *params, int vecCnt, size_t size, int pow2);
extern double HPCC_dweps();
extern float HPCC_sweps();
extern int HPCC_StarDGEMM(HPCC_Params *params);
extern int HPCC_SingleDGEMM(HPCC_Params *params);
extern int PTRANS(HPCC_Params *params);
extern int HPCC_SHMEMRandomAccess(HPCC_Params *params);
extern int HPCC_SingleRandomAccess(HPCC_Params *params);
extern int HPCC_StarRandomAccess(HPCC_Params *params);
extern int HPCC_SingleStream(HPCC_Params *params);
extern int HPCC_StarStream(HPCC_Params *params);
extern int HPCC_StarFFT(HPCC_Params *params);
extern int HPCC_SingleFFT(HPCC_Params *params);
extern int HPCC_SHMEMFFT(HPCC_Params *params);

extern int HPCC_TestFFT(HPCC_Params *params, int doIO, double *UGflops, int *Un, int *Ufailure);
extern int HPCC_TestDGEMM(HPCC_Params *params, int doIO, double *UGflops, int *Un, int *Ufailure);
extern int MaxMem(int nprocs, int imrow, int imcol, int nmat, int *mval, int *nval, int nbmat,
  int *mbval, int *nbval, int ngrids, int *npval, int *nqval, long *maxMem);
extern int HPCC_RandomAccess(HPCC_Params *params, int doIO, double *GUPs, int *failure);
extern int HPCC_Stream(HPCC_Params *params, int doIO, double *copyGBs, double *scaleGBs,
  double *addGBs, double *triadGBs, int *failure);
extern void main_bench_lat_bw(HPCC_Params *params);

extern int pdtrans(char *trans, int *m, int *n, int * mb, int *nb, double *a, int *lda,
  double *beta, double *c__, int *ldc, int *imrow, int *imcol, double *work, int *iwork);
extern int pdmatgen(int *ictxt, char *aform, char *diag, int *m, int *n, int *mb, int *nb, double*a,
  int *lda, int *iarow, int *iacol, int *iseed, int *iroff, int *irnum, int *icoff, int *icnum,
  int * myrow, int *mycol, int *nprow, int *npcol, double alpha);
extern int pdmatcmp(int *ictxt, int *m_, int *n_, double *a, int *lda_, double *aCopy, int *ldc_,
  double *error);
extern int pxerbla(int *ictxt, char *srname, int *info);
extern int slcombine_(int *ictxt, char *scope, char *op, char * timetype, int *n, int *ibeg,
  double *times);
extern int icopy_(int *n, int *sx, int *incx, int * sy, int *incy);
extern int numroc_(int *, int *, int *, int *, int *);
extern int slboot_(void);
extern int sltimer_(int *i__);
extern int ilcm_(int *, int *);
extern int iceil_(int *, int *);
extern double pdrand();
extern int setran_(int *, int *, int *);
extern int jumpit_(int *, int *, int *, int *);
extern int xjumpm_(int *, int *, int *, int *, int *, int *, int *);
/* ---------------------------------------------------------------------- */

#define DPRN(i,v) do{printf(__FILE__ "(%d)@%d:" #v "=%g\n",__LINE__,i,(double)(v));fflush(stdout);}while(0)

#define BEGIN_IO(r,fn,f) if(0==r){f=fopen(fn,"a");if(!f)fprintf(f=stderr,"Problem with appending to file '%s'\n",fn)
#define END_IO(r,f) fflush(f); if (f!=stdout && f!=stderr) fclose(f);} f=(FILE*)(NULL)

#include <hpccmema.h>

#define XMALLOC(t,s) ((t*)malloc(sizeof(t)*(s))) 
#define XCALLOC(t,s) ((t*)calloc((s),sizeof(t)))

#endif
