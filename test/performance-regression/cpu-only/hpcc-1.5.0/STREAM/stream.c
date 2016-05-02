/*-----------------------------------------------------------------------*/
/* Program: STREAM                                                       */
/* Revision: $Id: stream_mpi.c,v 1.7 2014/10/22 00:13:21 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2013: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*           "tuned STREAM benchmark results"                            */
/*           "based on a variant of the STREAM benchmark code"           */
/*         Other comparable, clear, and reasonable labelling is          */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/

#include <hpcc.h>

#include <float.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define TUNED 1
#define VERBOSE 1

/* INSTRUCTIONS:
 *
 * 1) STREAM requires different amounts of memory to run on different
 *           systems, depending on both the system cache size(s) and the
 *           granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *           to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. I don't worry about the difference
 *           between 10^6 and 2^20, so in practice the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M elements.
 *
 *      Version 5.10 increases the default array size from 2 million
 *          elements to 10 million elements in response to the increasing
 *          size of L3 caches.  The new default size is large enough for caches
 *          up to 20 MB.
 *      Version 5.10 changes the loop index variables from "register int"
 *          to "ssize_t", which allows array indices >2^32 (4 billion)
 *          on properly configured 64-bit systems.  Additional compiler options
 *          (such as "-mcmodel=medium") may be required for large memory runs.
 *
 *      Array size can be set at compile time without modifying the source
 *          code for the (many) compilers that support preprocessor definitions
 *          on the compile line.  E.g.,
 *                gcc -O -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream.100M
 *          will override the default size of 10M with a new size of 100M elements
 *          per array.
 */

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */

static int array_elements;
# define N 2000000
# define NTIMES 10

/*
// Make the scalar coefficient modifiable at compile time.
// The old value of 3.0 cause floating-point overflows after a relatively small
// number of iterations.  The new default of 0.42 allows over 2000 iterations for
// 32-bit IEEE arithmetic and over 18000 iterations for 64-bit IEEE arithmetic.
// The growth in the solution can be eliminated (almost) completely by setting
// the scalar value to 0.41421445, but this also means that the error checking
// code no longer triggers an error if the code does not actually execute the
// correct number of iterations!
*/
#ifndef SCALAR
#define SCALAR 0.42
#endif

/*
// ----------------------- !!! NOTE CHANGE IN DEFINITION !!! ------------------
// The OFFSET preprocessor variable is not used in this version of the benchmark.
// The user must change the code at or after the "posix_memalign" array allocations
//    to change the relative alignment of the pointers.
// ----------------------- !!! NOTE CHANGE IN DEFINITION !!! ------------------
*/
# define OFFSET 0

/*
 * 3) Compile the code with optimization.  Many compilers generate
 *    unreasonably bad code before the optimizer tightens things up.
 *    If the results are unreasonably good, on the other hand, the
 *    optimizer might be too smart for me!
 *
 *    For a simple single-core version, try compiling with:
 *            cc -O stream.c -o stream
 *    This is known to work on many, many systems....
 *
 *    To use multiple cores, you need to tell the compiler to obey the OpenMP
 *    directives in the code.  This varies by compiler, but a common example is
 *            gcc -O -fopenmp stream.c -o stream_omp
 *    The environment variable OMP_NUM_THREADS allows runtime control of the
 *    number of threads/cores used when the resulting "stream_omp" program
 *    is executed.
 *
 *     To run with single-precision variables and arithmetic, simply add
 *         -DSTREAM_TYPE=float
 *     to the compile line.
 *     Note that this changes the minimum array sizes required --- see (1) above.
 *
 *     The preprocessor directive "TUNED" does not do much -- it simply causes the
 *     code to call separate functions to execute each kernel.  Trivial versions
 *     of these functions are provided, but they are *not* tuned -- they just
 *     provide predefined interfaces to be replaced with tuned code.
 *
 *
 * 4) Optional: Mail the results to mccalpin@cs.virginia.edu
 *    Be sure to include info that will help me understand:
 *  a) the computer hardware configuration (e.g., processor model, memory type)
 *  b) the compiler name/version and compilation flags
 *  c) any run-time information (such as OMP_NUM_THREADS)
 *  d) all of the output from the test case.
 *
 * Thanks!
 *
 *-----------------------------------------------------------------------*/

# define HLINE "-------------------------------------------------------------\n"

/* Some compilers require an extra keyword to recognize the "restrict" qualifier. */
static double * a;
static double * b;
static double * c;

static double avgtime[4] = {0}, maxtime[4] = {0},
  mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char *label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double bytes[4] = {
    2 * sizeof(double),
    2 * sizeof(double),
    3 * sizeof(double),
    3 * sizeof(double)
    };

#ifdef TUNED
extern void tuned_STREAM_Copy(void);
extern void tuned_STREAM_Scale(double scalar);
extern void tuned_STREAM_Add(void);
extern void tuned_STREAM_Triad(double scalar);
#endif

static void
checkSTREAMresults(FILE *outFile, int doIO, double *AvgErrByRank, int numranks, int *failure) {
  double aj,bj,cj,scalar;
  double aSumErr,bSumErr,cSumErr;
  double aAvgErr,bAvgErr,cAvgErr;
  double epsilon;
  int j, k, ierr, err;

  /* Repeat the computation of aj, bj, cj */
  /* reproduce initialization */
  aj = 1.0;
  bj = 2.0;
  cj = 0.0;
  /* a[] is modified during timing check */
  aj = 2.0E0 * aj;
  /* now execute timing loop */
  scalar = SCALAR;
  for (k=0; k<NTIMES; k++) {
    cj = aj;
    bj = scalar*cj;
    cj = aj+bj;
    aj = bj+scalar*cj;
  }

  /* Compute the average of the average errors contributed by each MPI rank */
  aSumErr = 0.0;
  bSumErr = 0.0;
  cSumErr = 0.0;
  for (k=0; k<numranks; k++) {
          aSumErr += AvgErrByRank[3*k + 0];
          bSumErr += AvgErrByRank[3*k + 1];
          cSumErr += AvgErrByRank[3*k + 2];
  }
  aAvgErr = aSumErr / (double) numranks;
  bAvgErr = bSumErr / (double) numranks;
  cAvgErr = cSumErr / (double) numranks;

  if (sizeof(double) == 4) {
          epsilon = 1.e-6;
  }
  else if (sizeof(double) == 8) {
          epsilon = 1.e-13;
  }
  else if (sizeof(double) == 10) {
          epsilon = 1.e-23;
  }
  else {
          if (doIO) fprintf( outFile, "WEIRD: sizeof(double) = %lu\n",sizeof(double));
          epsilon = 1.e-6;
  }

  *failure = 1;

  err = 0;

  if (fabs(aAvgErr/aj) > epsilon) {
    err++;
    if (doIO) {
      fprintf( outFile, "Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
      fprintf( outFile, "     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,fabs(aAvgErr)/aj);
    }
    ierr = 0;
    for (j=0; j<array_elements; j++) {
      if (fabs(a[j]/aj-1.0) > epsilon) {
        ierr++;
      }
    }
    if (ierr > 0)
      if (doIO)
        fprintf( outFile, "     For array a[], %d errors were found.\n",ierr);
  }
  if (fabs(bAvgErr/bj) > epsilon) {
    err++;
    if (doIO) {
      fprintf( outFile, "Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
      fprintf( outFile, "     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,fabs(bAvgErr)/bj);
      fprintf( outFile, "     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
    }
    ierr = 0;
    for (j=0; j<array_elements; j++) {
      if (fabs(b[j]/bj-1.0) > epsilon) {
        ierr++;
      }
    }
    if (ierr > 0)
      if (doIO)
        fprintf( outFile, "     For array b[], %d errors were found.\n",ierr);
  }
  if (fabs(cAvgErr/cj) > epsilon) {
    err++;
    if (doIO) {
      fprintf( outFile, "Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
      fprintf( outFile, "     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,fabs(cAvgErr)/cj);
      fprintf( outFile, "     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
    }
    ierr = 0;
    for (j=0; j<array_elements; j++) {
      if (fabs(c[j]/cj-1.0) > epsilon) {
        ierr++;
      }
    }
    if (ierr > 0)
      if (doIO)
        fprintf( outFile, "     For array c[], %d errors were found.\n",ierr);
  }
  if (err == 0) {
    *failure = 0;
    if (doIO)
      fprintf( outFile, "Solution Validates: avg error less than %e on all three arrays\n",epsilon);
  }
}

# define M 20

static int
checktick() {
    int  i, minDelta, Delta;
    double t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
      t1 = MPI_Wtime();
      while( ((t2=MPI_Wtime()) - t1) < 1.0E-6 )
        ;
      timesfound[i] = t1 = t2;
    }

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
      Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
      minDelta = Mmin(minDelta, Mmax(Delta,0));
    }

   return(minDelta);
}
#undef M


/*
For the MPI code I separate the computation of errors from the error
reporting output functions (which are handled by MPI rank 0).
*/
void computeSTREAMerrors(double *aAvgErr, double *bAvgErr, double *cAvgErr)
{
  double aj,bj,cj,scalar;
  double aSumErr,bSumErr,cSumErr;
  int j;
  int k;

  /* reproduce initialization */
  aj = 1.0;
  bj = 2.0;
  cj = 0.0;
  /* a[] is modified during timing check */
  aj = 2.0E0 * aj;
  /* now execute timing loop */
  scalar = SCALAR;
  for (k=0; k<NTIMES; k++)
  {
      cj = aj;
      bj = scalar*cj;
      cj = aj+bj;
      aj = bj+scalar*cj;
  }

  /* accumulate deltas between observed and expected results */
  aSumErr = 0.0;
  bSumErr = 0.0;
  cSumErr = 0.0;
  for (j=0; j<array_elements; j++) {
    aSumErr += fabs(a[j] - aj);
    bSumErr += fabs(b[j] - bj);
    cSumErr += fabs(c[j] - cj);
  }
  *aAvgErr = aSumErr / (double) array_elements;
  *bAvgErr = bSumErr / (double) array_elements;
  *cAvgErr = cSumErr / (double) array_elements;
}


int
HPCC_Stream(HPCC_Params *params, int doIO, MPI_Comm comm, int world_rank,
  double *copyGBs, double *scaleGBs, double *addGBs, double *triadGBs, int *failure) {
    int quantum,  BytesPerWord, numranks, myrank;
    int j, k;
    double  scalar, t, t0, t1, times[4][NTIMES], times_copy[4][NTIMES];
    FILE *outFile;
    double GiBs = 1024.0 * 1024.0 * 1024.0, curGBs;
    double AvgError[3] = {0.0,0.0,0.0};
    double *AvgErrByRank;


    if (doIO) {
      outFile = fopen( params->outFname, "a" );
      if (! outFile) {
        outFile = stderr;
        fprintf( outFile, "Cannot open output file.\n" );
        return 1;
      }
    }

    t0 = MPI_Wtime();

    MPI_Comm_size( comm, &numranks );
    MPI_Comm_rank( comm, &myrank );

    array_elements = HPCC_LocalVectorSize( params, 3, sizeof(double), 0 ); /* Need 3 vectors */
    params->StreamVectorSize = array_elements;

    a = HPCC_XMALLOC( double, array_elements );
    b = HPCC_XMALLOC( double, array_elements );
    c = HPCC_XMALLOC( double, array_elements );

    if (!a || !b || !c) {
      if (c) HPCC_free(c);
      if (b) HPCC_free(b);
      if (a) HPCC_free(a);
      if (doIO) {
        fprintf( outFile, "Failed to allocate memory (%d).\n", array_elements );
        fflush( outFile );
        fclose( outFile );
      }
      /* FIXME: must be made global */
      return 1;
    }

    /* --- SETUP --- determine precision and check timing --- */

    if (doIO) {
    fprintf( outFile, HLINE);
    BytesPerWord = sizeof(double);
    fprintf( outFile, "This system uses %d bytes per DOUBLE PRECISION word.\n",
             BytesPerWord);

    fprintf( outFile, HLINE);
    fprintf( outFile, "Array size = %d, Offset = %d\n" , array_elements, OFFSET);
    fprintf( outFile, "Total memory required = %.4f GiB.\n",
             (3.0 * BytesPerWord) * ( (double) array_elements / GiBs));
    fprintf( outFile, "Each test is run %d times.\n", NTIMES );
    fprintf( outFile, " The *best* time for each kernel (excluding the first iteration)\n" );
    fprintf( outFile, " will be used to compute the reported bandwidth.\n");
    fprintf( outFile, "The SCALAR value used for this run is %f\n", SCALAR );

    }

#ifdef _OPENMP
    if (doIO) fprintf( outFile, HLINE);
#pragma omp parallel private(k)
    {
#pragma omp single nowait
      {
        k = omp_get_num_threads();
        if (doIO) fprintf( outFile, "Number of Threads requested = %i\n",k);
        params->StreamThreads = k;
      }
    }
#endif

    /* --- SETUP --- initialize arrays and estimate precision of timer --- */

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (j=0; j<array_elements; j++) {
      a[j] = 1.0;
      b[j] = 2.0;
      c[j] = 0.0;
    }

    /* Rank 0 needs to allocate arrays to hold error data and timing data from
       all ranks for analysis and output.
       Allocate and instantiate the arrays here -- after the primary arrays
       have been instantiated -- so there is no possibility of having these
       auxiliary arrays mess up the NUMA placement of the primary arrays. */

    /* There are 3 average error values for each rank (using double). */
    AvgErrByRank = HPCC_XMALLOC( double, 3 * numranks );

    /* There are 4*NTIMES timing values for each rank (always doubles) */
    if (AvgErrByRank == NULL) {
      if (doIO)
        fprintf( outFile, "Ooops -- allocation of arrays to collect timing data on MPI rank %d failed\n", world_rank);
      MPI_Abort(comm, 3); /* FIXME: handle failure more gracefully */
    }

    /* FIXME: replace with loop to use floating-point data */
    memset(AvgErrByRank,0,3*sizeof(double)*numranks);

    if (doIO) fprintf( outFile, HLINE);

    if  ( (quantum = checktick()) >= 1) {
      if (doIO) fprintf( outFile, "Your clock granularity/precision appears to be "
                         "%d microseconds.\n", quantum);
    } else {
      if (doIO) fprintf( outFile, "Your clock granularity appears to be "
                         "less than one microsecond.\n");
    }

    /* Get initial timing estimate to compare to timer granularity.
       All ranks need to run this code since it changes the values in array `a' */
    t = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (j = 0; j < array_elements; j++)
      a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (MPI_Wtime() - t);

    if (doIO) {
    fprintf( outFile, "Each test below will take on the order"
             " of %d microseconds.\n", (int) t  );
    fprintf( outFile, "   (= %d clock ticks)\n", (int) (t/quantum) );
    fprintf( outFile, "Increase the size of the arrays if this shows that\n");
    fprintf( outFile, "you are not getting at least 20 clock ticks per test.\n");

    fprintf( outFile, HLINE);

    fprintf( outFile, "WARNING -- The above is only a rough guideline.\n");
    fprintf( outFile, "For best results, please be sure you know the\n");
    fprintf( outFile, "precision of your system timer.\n");
    fprintf( outFile, HLINE);

    t1 = MPI_Wtime();
    fprintf( outFile, "VERBOSE: total setup time for rank 0 = %f seconds\n",t1-t0);
    fprintf( outFile, HLINE);
    }

    /* --- MAIN LOOP --- repeat test cases NTIMES times --- */

    /* This code has more barriers and timing calls than are actually needed, but
       this should not cause a problem for arrays that are large enough to satisfy
       the STREAM run rules. */

    scalar = SCALAR;
    for (k=0; k<NTIMES; k++) {
        /* kernel 1: Copy */
        MPI_Barrier( comm );
        times[0][k] = MPI_Wtime();
#ifdef TUNED
        tuned_STREAM_Copy();
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (j=0; j<array_elements; j++)
          c[j] = a[j];
#endif
        MPI_Barrier( comm );
        times[0][k] = MPI_Wtime() - times[0][k];

        /* kernel 2: Scale */
        MPI_Barrier( comm );
        times[1][k] = MPI_Wtime();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (j=0; j<array_elements; j++)
          b[j] = scalar*c[j];
#endif
        MPI_Barrier( comm );
        times[1][k] = MPI_Wtime() - times[1][k];

        /* kernel 3: Add */
        MPI_Barrier( comm );
        times[2][k] = MPI_Wtime();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (j=0; j<array_elements; j++)
          c[j] = a[j]+b[j];
#endif
        MPI_Barrier( comm );
        times[2][k] = MPI_Wtime() - times[2][k];

        /* kernel 4: Triad */
        MPI_Barrier( comm );
        times[3][k] = MPI_Wtime();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (j=0; j<array_elements; j++)
          a[j] = b[j]+scalar*c[j];
#endif
        MPI_Barrier( comm );
        times[3][k] = MPI_Wtime() - times[3][k];
    }

    t0 = MPI_Wtime();

    /* --- SUMMARY --- */

    /* Because of the MPI_Barrier() calls, the timings from any thread are equally valid.
       The best estimate of the maximum performance is the minimum of the "outside the barrier"
       timings across all the MPI ranks. */

    memcpy(times_copy, times, sizeof times_copy );

    /* for each iteration and each kernel, collect the minimum time across all MPI ranks */
    MPI_Allreduce( times_copy, times, 4*NTIMES, MPI_DOUBLE, MPI_MIN, comm );

    /* Back to the original code, but now using the minimum global timing across all ranks */
    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
    {
      for (j=0; j<4; j++)
      {
        avgtime[j] = avgtime[j] + times[j][k];
        mintime[j] = Mmin(mintime[j], times[j][k]);
        maxtime[j] = Mmax(maxtime[j], times[j][k]);
      }
    }

    if (doIO)
      fprintf( outFile, "Function      Rate (GB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
      avgtime[j] /= (double)(NTIMES - 1); /* note -- skip first iteration */

      /* make sure no division by zero */
      curGBs = (mintime[j] > 0.0 ? 1.0 / mintime[j] : -1.0);
      curGBs *= 1e-9 * bytes[j] * array_elements;
      if (doIO)
        fprintf( outFile, "%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
                 curGBs,
                 avgtime[j],
                 mintime[j],
                 maxtime[j]);
      switch (j) {
        case 0: *copyGBs = curGBs; break;
        case 1: *scaleGBs = curGBs; break;
        case 2: *addGBs = curGBs; break;
        case 3: *triadGBs = curGBs; break;
      }
    }
    if (doIO)
      fprintf( outFile, HLINE);

    /* --- Every Rank Checks its Results --- */
    computeSTREAMerrors(&AvgError[0], &AvgError[1], &AvgError[2]);
    /* --- Collect the Average Errors for Each Array on Rank 0 --- */
    MPI_Gather(AvgError, 3, MPI_DOUBLE, AvgErrByRank, 3, MPI_DOUBLE, 0, comm);

    /* -- Combined averaged errors and report on Rank 0 only --- */
    if (myrank == 0) {
      checkSTREAMresults( outFile, doIO, AvgErrByRank, numranks, failure );
      if (doIO) fprintf( outFile, HLINE);
    }

    HPCC_free(AvgErrByRank);

    HPCC_free(c);
    HPCC_free(b);
    HPCC_free(a);

    if (doIO) {
      fflush( outFile );
      fclose( outFile );
    }

    return 0;
}

void tuned_STREAM_Copy()
{
  int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (j=0; j<array_elements; j++)
            c[j] = a[j];
}

void tuned_STREAM_Scale(double scalar)
{
  int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (j=0; j<array_elements; j++)
    b[j] = scalar*c[j];
}

void tuned_STREAM_Add()
{
  int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (j=0; j<array_elements; j++)
    c[j] = a[j]+b[j];
}

void tuned_STREAM_Triad(double scalar)
{
  int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (j=0; j<array_elements; j++)
    a[j] = b[j]+scalar*c[j];
}
