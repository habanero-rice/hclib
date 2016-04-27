/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; -*- */
#include<time.h>
#include<sys/time.h>

/* Random number generator */
#ifdef LONG_IS_64BITS
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
#else
#define POLY 0x0000000000000007ULL
#define PERIOD 1317624576693539401LL
#endif

/* For timing */

static double RTSEC()
{
 struct timeval tp;
 gettimeofday (&tp, NULL);
 return tp.tv_sec + tp.tv_usec/(double)1.0e6;
}

extern s64Int starts (u64Int);

#define WANT_MPI2_TEST 0


#define HPCC_TRUE 1
#define HPCC_FALSE 0
#define HPCC_DONE 0

#define FINISHED_TAG 1
#define UPDATE_TAG   2
#define USE_NONBLOCKING_SEND 1

#define MAX_TOTAL_PENDING_UPDATES 1024
#define LOCAL_BUFFER_SIZE MAX_TOTAL_PENDING_UPDATES

#define USE_MULTIPLE_RECV 1

#ifdef USE_MULTIPLE_RECV
#define MAX_RECV					16
#else
#define MAX_RECV					1
#endif

extern u64Int *HPCC_Table;

extern u64Int LocalSendBuffer[LOCAL_BUFFER_SIZE];
extern u64Int LocalRecvBuffer[MAX_RECV*LOCAL_BUFFER_SIZE];

extern void
Power2NodesRandomAccessUpdate(u64Int logTableSize,
                                 u64Int TableSize,
                                 u64Int LocalTableSize,
                                 u64Int MinLocalTableSize,
                                 u64Int GlobalStartMyProc,
                                 u64Int Top,
                                 int logNumProcs,
                                 int NumProcs,
                                 int Remainder,
                                 int MyProc,
                                 s64Int ProcNumUpdates);

