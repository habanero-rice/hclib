/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; -*- */
/* time_bound.c
 *
 * Estimates the largest number of updates that
 * will keep the benchmark's execution time under
 * a specified time bound.
 *
 * The number of updates is estimated by performing
 * a fraction (currently 1% as defined in time_bound.h)
 * of the default number of updates and
 * measuring the execution time. The maximum
 * number of updates then is estimated based on
 * the average execution time per update and the time
 * bound.
 */


#include <hpcc.h>
#include "RandomAccess.h"
#include "time_bound.h"
#include "buckets.h"


void HPCC_Power2NodesTimeLCG(HPCC_RandomAccess_tabparams_t tparams, double timeBound, u64Int *EstimatedNumIter) {
  s64Int i, j;
  int proc_count;

  s64Int SendCnt;
  u64Int Ran;
  s64Int WhichPe;
  u64Int LocalOffset;
  int logLocalTableSize = tparams.logTableSize - tparams.logNumProcs;
  int NumberReceiving = tparams.NumProcs - 1;
#ifdef USE_MULTIPLE_RECV
  int index, NumRecvs;
  MPI_Request inreq[MAX_RECV]  = { MPI_REQUEST_NULL };
  MPI_Request outreq = MPI_REQUEST_NULL;
#else
  MPI_Request inreq, outreq = MPI_REQUEST_NULL;
#endif
  int bufferBase;
  u64Int inmsg;
  MPI_Status status;
  int have_done;

  int pe;
  int pendingUpdates;
  int maxPendingUpdates;
  int localBufferSize;
  int peUpdates;
  int recvUpdates;
  Bucket_Ptr Buckets;

  double ra_LoopRealTime;
  double iterTime;

  /* Initialize main table */
  for (i=0; i<tparams.LocalTableSize; i++)
    HPCC_Table[i] = i + tparams.GlobalStartMyProc;

  /* Perform updates to main table.  The scalar equivalent is:
   *
   *     u64Int Ran;
   *     Ran = 1;
   *     for (i=0; i<NUPDATE; i++) {
   *       Ran = LCG_MUL64 * Ran + LCG_ADD64;
   *       Table[Ran >> (64 - LOG2_TABSIZE)] ^= Ran;
   *     }
   */

  ra_LoopRealTime = -RTSEC();

  pendingUpdates = 0;
  maxPendingUpdates = MAX_TOTAL_PENDING_UPDATES;
  localBufferSize = LOCAL_BUFFER_SIZE;
  Buckets = HPCC_InitBuckets(tparams.NumProcs, maxPendingUpdates);

  SendCnt = 4 * tparams.LocalTableSize/RA_SAMPLE_FACTOR;
  Ran = HPCC_starts_LCG(4 * tparams.GlobalStartMyProc);

  i = 0;
#ifdef USE_MULTIPLE_RECV
  NumRecvs = (tparams.NumProcs > 4) ? (Mmin(4,MAX_RECV)) : 1;
  for (j = 0; j < NumRecvs; j++)
    MPI_Irecv(&LocalRecvBuffer[j*LOCAL_BUFFER_SIZE], localBufferSize,
              tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
              &inreq[j]);
#else
  MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
          MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif

  while (i < SendCnt) {

     /* receive messages */
     do {
#ifdef USE_MULTIPLE_RECV
       MPI_Testany(NumRecvs, inreq, &index, &have_done, &status);
#else
       MPI_Test(&inreq, &have_done, &status);
#endif
       if (have_done) {
         if (status.MPI_TAG == UPDATE_TAG) {
           MPI_Get_count(&status, tparams.dtype64, &recvUpdates);
#ifdef USE_MULTIPLE_RECV
           bufferBase = index*LOCAL_BUFFER_SIZE;
#else
           bufferBase = 0;
#endif
           for (j=0; j < recvUpdates; j ++) {
             inmsg = LocalRecvBuffer[bufferBase+j];
             HPCC_Table[(inmsg  >> (64 - tparams.logTableSize)) & (tparams.LocalTableSize - 1)] ^= inmsg;
           }
         } else if (status.MPI_TAG == FINISHED_TAG) {
           /* we got a done message.  Thanks for playing... */
           NumberReceiving--;
         } else {
           MPI_Abort( MPI_COMM_WORLD, -1 );
         }
#ifdef USE_MULTIPLE_RECV
         MPI_Irecv(&LocalRecvBuffer[index*LOCAL_BUFFER_SIZE], localBufferSize,
                   tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                   &inreq[index]);
#else
         MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
                   MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif
       }
     } while (have_done && NumberReceiving > 0);


     if (pendingUpdates < maxPendingUpdates) {
       Ran = LCG_MUL64 * Ran + LCG_ADD64;
       WhichPe = (Ran >> (64 - tparams.logTableSize + logLocalTableSize)) & (tparams.NumProcs - 1);
       if (WhichPe == tparams.MyProc) {
         LocalOffset = (Ran >> (64 - tparams.logTableSize)) - tparams.GlobalStartMyProc;
         HPCC_Table[LocalOffset] ^= Ran;
       }
       else {
         HPCC_InsertUpdate(Ran, WhichPe, Buckets);
         pendingUpdates++;
       }
       i++;
     }

     else {
       MPI_Test(&outreq, &have_done, MPI_STATUS_IGNORE);
       if (have_done) {
         outreq = MPI_REQUEST_NULL;
         pe = HPCC_GetUpdates (Buckets, LocalSendBuffer, localBufferSize, &peUpdates);
         MPI_Isend(&LocalSendBuffer, peUpdates, tparams.dtype64, (int)pe, UPDATE_TAG,
                   MPI_COMM_WORLD, &outreq);
         pendingUpdates -= peUpdates;
       }

     }
   }

   /* send updates in buckets */
   while (pendingUpdates > 0) {

     /* receive messages */
     do {
#ifdef USE_MULTIPLE_RECV
       MPI_Testany(NumRecvs, inreq, &index, &have_done, &status);
#else
       MPI_Test(&inreq, &have_done, &status);
#endif
       if (have_done) {
         if (status.MPI_TAG == UPDATE_TAG) {
           MPI_Get_count(&status, tparams.dtype64, &recvUpdates);
#ifdef USE_MULTIPLE_RECV
           bufferBase = index * LOCAL_BUFFER_SIZE;
#else
           bufferBase = 0;
#endif
           for (j=0; j < recvUpdates; j ++) {
             inmsg = LocalRecvBuffer[bufferBase+j];
             HPCC_Table[(inmsg >> (64 - tparams.logTableSize)) & (tparams.LocalTableSize-1)] ^= inmsg;
           }
         } else if (status.MPI_TAG == FINISHED_TAG) {
           /* we got a done message.  Thanks for playing... */
           NumberReceiving--;
         } else {
           MPI_Abort( MPI_COMM_WORLD, -1 );
         }
#ifdef USE_MULTIPLE_RECV
         MPI_Irecv(&LocalRecvBuffer[index*LOCAL_BUFFER_SIZE], localBufferSize,
                   tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                   &inreq[index]);
#else
         MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
                   MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif
       }
     } while (have_done && NumberReceiving > 0);


     MPI_Test(&outreq, &have_done, MPI_STATUS_IGNORE);
     if (have_done) {
       outreq = MPI_REQUEST_NULL;
       pe = HPCC_GetUpdates(Buckets, LocalSendBuffer, localBufferSize, &peUpdates);
       MPI_Isend(&LocalSendBuffer, peUpdates, tparams.dtype64, (int)pe, UPDATE_TAG,
                 MPI_COMM_WORLD, &outreq);
       pendingUpdates -= peUpdates;
     }
   }


   /* send our done messages */
   for (proc_count = 0 ; proc_count < tparams.NumProcs ; ++proc_count) {
     if (proc_count == tparams.MyProc) { tparams.finish_req[tparams.MyProc] = MPI_REQUEST_NULL; continue; }
     /* send garbage - who cares, no one will look at it */
     MPI_Isend(&Ran, 0, tparams.dtype64, proc_count, FINISHED_TAG,
               MPI_COMM_WORLD, tparams.finish_req + proc_count);
   }


   /* Finish everyone else up... */
   while (NumberReceiving > 0) {
#ifdef USE_MULTIPLE_RECV
     MPI_Waitany(NumRecvs, inreq, &index, &status);
#else
     MPI_Wait(&inreq, &status);
#endif
     if (status.MPI_TAG == UPDATE_TAG) {
       MPI_Get_count(&status, tparams.dtype64, &recvUpdates);
#ifdef USE_MULTIPLE_RECV
       bufferBase = index * LOCAL_BUFFER_SIZE;
#else
       bufferBase = 0;
#endif
       for (j=0; j <recvUpdates; j ++) {
         inmsg = LocalRecvBuffer[bufferBase+j];
         HPCC_Table[(inmsg >> (64 - tparams.logTableSize)) & (tparams.LocalTableSize - 1)] ^= inmsg;
       }
     } else if (status.MPI_TAG == FINISHED_TAG) {
       /* we got a done message.  Thanks for playing... */
       NumberReceiving--;
     } else {
       MPI_Abort( MPI_COMM_WORLD, -1 );
     }
#ifdef USE_MULTIPLE_RECV
     MPI_Irecv(&LocalRecvBuffer[index*LOCAL_BUFFER_SIZE], localBufferSize,
               tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
               &inreq[index]);
#else
     MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
               MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif
   }

   ra_LoopRealTime += RTSEC();

   /* estimate largest number of iterations that satisfy time bound */
   iterTime = (double)(ra_LoopRealTime/SendCnt);
   *EstimatedNumIter = timeBound/iterTime;
#ifdef DEBUG_TIME_BOUND
   fprintf (stdout, "MyProc: %4d SampledNumIter: %8d ", tparams.MyProc, SendCnt);
   fprintf (stdout, "LoopRealTime: %.8f IterTime: %.8f EstimatedNumIter: %8d\n",
       ra_LoopRealTime, iterTime, *EstimatedNumIter);
#endif

   MPI_Waitall( tparams.NumProcs, tparams.finish_req, tparams.finish_statuses);

   /* Be nice and clean up after ourselves */
   HPCC_FreeBuckets(Buckets, tparams.NumProcs);
#ifdef USE_MULTIPLE_RECV
  for (j = 0; j < NumRecvs; j++) {
    MPI_Cancel(&inreq[j]);
    MPI_Wait(&inreq[j], MPI_STATUS_IGNORE);
  }
#else
  MPI_Cancel(&inreq);
  MPI_Wait(&inreq, MPI_STATUS_IGNORE);
#endif
  MPI_Wait(&outreq, MPI_STATUS_IGNORE);

/* end multiprocessor code */
}



void HPCC_AnyNodesTimeLCG(HPCC_RandomAccess_tabparams_t tparams, double timeBound, u64Int *EstimatedNumIter) {
  s64Int i, j;
  int proc_count;

  s64Int SendCnt;
  u64Int Ran;
  s64Int WhichPe;
  u64Int GlobalOffset, LocalOffset;
  int NumberReceiving = tparams.NumProcs - 1;
#ifdef USE_MULTIPLE_RECV
  int index, NumRecvs;
  MPI_Request inreq[MAX_RECV]  = { MPI_REQUEST_NULL };
  MPI_Request outreq = MPI_REQUEST_NULL;
#else
  MPI_Request inreq, outreq = MPI_REQUEST_NULL;
#endif
  u64Int inmsg;
  int bufferBase;
  MPI_Status status;
  int have_done;

  int pe;
  int pendingUpdates;
  int maxPendingUpdates;
  int localBufferSize;
  int peUpdates;
  int recvUpdates;
  Bucket_Ptr Buckets;

  double ra_LoopRealTime;
  double iterTime;

  /* Initialize main table */
  for (i=0; i<tparams.LocalTableSize; i++)
    HPCC_Table[i] = i + tparams.GlobalStartMyProc;

  /* Perform updates to main table.  The scalar equivalent is:
   *
   *     u64Int Ran;
   *     Ran = 1;
   *     for (i=0; i<NUPDATE; i++) {
   *       Ran = LCG_MUL64 * Ran + LCG_ADD64;
   *       Table[Ran >> (64 - LOG2_TABSIZE)] ^= Ran;
   *     }
   */

  ra_LoopRealTime = -RTSEC();

  pendingUpdates = 0;
  maxPendingUpdates = MAX_TOTAL_PENDING_UPDATES;
  localBufferSize = LOCAL_BUFFER_SIZE;
  Buckets = HPCC_InitBuckets(tparams.NumProcs, maxPendingUpdates);

  SendCnt = 4 * tparams.LocalTableSize/RA_SAMPLE_FACTOR;
  Ran = HPCC_starts_LCG(4 * tparams.GlobalStartMyProc);

  i = 0;
#ifdef USE_MULTIPLE_RECV
  NumRecvs = (tparams.NumProcs > 4) ? (Mmin(4,MAX_RECV)) : 1;
  for (j = 0; j < NumRecvs; j++)
    MPI_Irecv(&LocalRecvBuffer[j*LOCAL_BUFFER_SIZE], localBufferSize,
              tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
              &inreq[j]);
#else
  MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
            MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif

  while (i < SendCnt) {

    /* receive messages */
    do {
#ifdef USE_MULTIPLE_RECV
      MPI_Testany(NumRecvs, inreq, &index, &have_done, &status);
#else
      MPI_Test(&inreq, &have_done, &status);
#endif
      if (have_done) {
        if (status.MPI_TAG == UPDATE_TAG) {
          MPI_Get_count(&status, tparams.dtype64, &recvUpdates);
#ifdef USE_MULTIPLE_RECV
          bufferBase = index*LOCAL_BUFFER_SIZE;
#else
          bufferBase = 0;
#endif
          for (j=0; j < recvUpdates; j ++) {
            inmsg = LocalRecvBuffer[bufferBase+j];
            LocalOffset = (inmsg >> (64 - tparams.logTableSize)) - tparams.GlobalStartMyProc;
            HPCC_Table[LocalOffset] ^= inmsg;
          }
        } else if (status.MPI_TAG == FINISHED_TAG) {
          /* we got a done message.  Thanks for playing... */
          NumberReceiving--;
        } else {
          MPI_Abort( MPI_COMM_WORLD, -1 );
        }
#ifdef USE_MULTIPLE_RECV
        MPI_Irecv(&LocalRecvBuffer[index*LOCAL_BUFFER_SIZE], localBufferSize,
                  tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                  &inreq[index]);
#else
        MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
                  MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif
      }
    } while (have_done && NumberReceiving > 0);


    if (pendingUpdates < maxPendingUpdates) {
      Ran = LCG_MUL64 * Ran + LCG_ADD64;
      GlobalOffset = Ran >> (64 - tparams.logTableSize);
      if ( GlobalOffset < tparams.Top)
        WhichPe = ( GlobalOffset / (tparams.MinLocalTableSize + 1) );
      else
        WhichPe = ( (GlobalOffset - tparams.Remainder) / tparams.MinLocalTableSize );

      if (WhichPe == tparams.MyProc) {
        LocalOffset = (Ran >> (64 - tparams.logTableSize)) - tparams.GlobalStartMyProc;
        HPCC_Table[LocalOffset] ^= Ran;
      }
      else {
        HPCC_InsertUpdate(Ran, WhichPe, Buckets);
        pendingUpdates++;
      }
      i++;
    }

    else {
      MPI_Test(&outreq, &have_done, MPI_STATUS_IGNORE);
      if (have_done) {
        outreq = MPI_REQUEST_NULL;
        pe = HPCC_GetUpdates (Buckets, LocalSendBuffer, localBufferSize, &peUpdates);
        MPI_Isend(&LocalSendBuffer, peUpdates, tparams.dtype64, (int)pe, UPDATE_TAG,
                  MPI_COMM_WORLD, &outreq);
        pendingUpdates -= peUpdates;
      }
    }

  }


  /* send updates in buckets */
  while (pendingUpdates > 0) {

    /* receive messages */
    do {
#ifdef USE_MULTIPLE_RECV
      MPI_Testany(NumRecvs, inreq, &index, &have_done, &status);
#else
      MPI_Test(&inreq, &have_done, &status);
#endif
      if (have_done) {
        if (status.MPI_TAG == UPDATE_TAG) {
          MPI_Get_count(&status, tparams.dtype64, &recvUpdates);
#ifdef USE_MULTIPLE_RECV
          bufferBase = index*LOCAL_BUFFER_SIZE;
#else
          bufferBase = 0;
#endif
          for (j=0; j < recvUpdates; j ++) {
            inmsg = LocalRecvBuffer[bufferBase+j];
            LocalOffset = (inmsg >> (64 - tparams.logTableSize)) - tparams.GlobalStartMyProc;
            HPCC_Table[LocalOffset] ^= inmsg;
          }
        } else if (status.MPI_TAG == FINISHED_TAG) {
          /* we got a done message.  Thanks for playing... */
          NumberReceiving--;
        } else {
          MPI_Abort( MPI_COMM_WORLD, -1 );
        }
#ifdef USE_MULTIPLE_RECV
        MPI_Irecv(&LocalRecvBuffer[index*LOCAL_BUFFER_SIZE], localBufferSize,
                  tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                  &inreq[index]);
#else
        MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
                  MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif
      }
    } while (have_done && NumberReceiving > 0);


    MPI_Test(&outreq, &have_done, MPI_STATUS_IGNORE);
    if (have_done) {
      outreq = MPI_REQUEST_NULL;
      pe = HPCC_GetUpdates (Buckets, LocalSendBuffer, localBufferSize, &peUpdates);
      MPI_Isend(&LocalSendBuffer, peUpdates, tparams.dtype64, (int)pe, UPDATE_TAG,
                MPI_COMM_WORLD, &outreq);
      pendingUpdates -= peUpdates;
    }

  }

  /* send our done messages */
  for (proc_count = 0 ; proc_count < tparams.NumProcs ; ++proc_count) {
    if (proc_count == tparams.MyProc) { tparams.finish_req[tparams.MyProc] = MPI_REQUEST_NULL; continue; }
    /* send garbage - who cares, no one will look at it */
    MPI_Isend(&Ran, 0, tparams.dtype64, proc_count, FINISHED_TAG,
              MPI_COMM_WORLD, tparams.finish_req + proc_count);
  }

  /* Finish everyone else up... */
  while (NumberReceiving > 0) {
#ifdef USE_MULTIPLE_RECV
    MPI_Waitany(NumRecvs, inreq, &index, &status);
#else
    MPI_Wait(&inreq, &status);
#endif
    if (status.MPI_TAG == UPDATE_TAG) {
      MPI_Get_count(&status, tparams.dtype64, &recvUpdates);
#ifdef USE_MULTIPLE_RECV
      bufferBase = index * LOCAL_BUFFER_SIZE;
#else
      bufferBase = 0;
#endif
      for (j=0; j <recvUpdates; j ++) {
        inmsg = LocalRecvBuffer[bufferBase+j];
        LocalOffset = (inmsg >> (64 - tparams.logTableSize)) - tparams.GlobalStartMyProc;
        HPCC_Table[LocalOffset] ^= inmsg;
      }
    } else if (status.MPI_TAG == FINISHED_TAG) {
      /* we got a done message.  Thanks for playing... */
      NumberReceiving--;
    } else {
      MPI_Abort( MPI_COMM_WORLD, -1 );
    }
#ifdef USE_MULTIPLE_RECV
    MPI_Irecv(&LocalRecvBuffer[index*LOCAL_BUFFER_SIZE], localBufferSize,
              tparams.dtype64, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
              &inreq[index]);
#else
    MPI_Irecv(&LocalRecvBuffer, localBufferSize, tparams.dtype64,
              MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &inreq);
#endif
  }

  ra_LoopRealTime += RTSEC();
  iterTime = (double) (ra_LoopRealTime/SendCnt);
  *EstimatedNumIter = (int)(timeBound/iterTime);

#ifdef DEBUG_TIME_BOUND
  fprintf (stdout, "MyProc: %4d SampledNumIter: %8d ", tparams.MyProc, SendCnt);
  fprintf (stdout, "LoopRealTime: %.8f IterTime: %.8f EstimatedNumIter: %8d\n",
     ra_LoopRealTime, iterTime, *EstimatedNumIter);
#endif

  MPI_Waitall( tparams.NumProcs, tparams.finish_req, tparams.finish_statuses);

  /* Be nice and clean up after ourselves */
  HPCC_FreeBuckets(Buckets, tparams.NumProcs);
#ifdef USE_MULTIPLE_RECV
  for (j = 0; j < NumRecvs; j++) {
    MPI_Cancel(&inreq[j]);
    MPI_Wait(&inreq[j], MPI_STATUS_IGNORE);
  }
#else
  MPI_Cancel(&inreq);
  MPI_Wait(&inreq, MPI_STATUS_IGNORE);
#endif
  MPI_Wait(&outreq, MPI_STATUS_IGNORE);

  /* end multiprocessor code */
}
