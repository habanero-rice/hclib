/*
 * OpenSHMEM version:
 *
 * Copyright (c) 2011 - 2015
 *   University of Houston System and UT-Battelle, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * o Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * o Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * o Neither the name of the University of Houston System,
 *   UT-Battelle, LLC. nor the names of its contributors may be used to
 *   endorse or promote products derived from this software without specific
 *   prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#include <hpcc.h>
#include "RandomAccess.h"
#include <stdio.h>
#include <shmem.h>

/* Verification phase: local buckets to sort into */
#define BUCKET_SIZE 1024
#define SLOT_CNT 1
#define FIRST_SLOT 2

void
HPCC_Power2NodesSHMEMRandomAccessCheck(u64Int logTableSize,
                                u64Int TableSize,
                                u64Int LocalTableSize,
                                u64Int GlobalStartMyProc,
                                int logNumProcs,
                                int NumProcs,
                                int MyProc,
				                u64Int ProcNumUpdates,
                                s64Int *NumErrors)
{

  u64Int Ran;
  u64Int RanTmp;
  s64Int NextSlot;
  s64Int WhichPe;
  s64Int PeBucketBase;
  s64Int SendCnt;
  s64Int errors;
  int i;
  int j;
  s64Int *PeCheckDone;
  int LocalAllDone =  HPCC_FALSE;
  static int sAbort, rAbort;

  u64Int *LocalBuckets;     /* buckets used in verification phase */
  u64Int *GlobalBuckets;    /* buckets used in verification phase */

  static long int ipSync[_SHMEM_BCAST_SYNC_SIZE];
  static int ipWrk[_SHMEM_REDUCE_SYNC_SIZE];
  static long int lpSync[_SHMEM_BCAST_SYNC_SIZE];
  static long lpWrk[_SHMEM_REDUCE_SYNC_SIZE];
  const int slot_size = BUCKET_SIZE+FIRST_SLOT;

  for (i = 0; i < _SHMEM_BCAST_SYNC_SIZE; i += 1){
        ipSync[i] = _SHMEM_SYNC_VALUE;
        lpSync[i] = _SHMEM_SYNC_VALUE;
  }

  LocalBuckets = shmem_malloc(sizeof(u64Int)*NumProcs*(slot_size));
  sAbort = 0; if (! LocalBuckets) sAbort = 1;
  shmem_barrier_all(); 
  shmem_int_sum_to_all(&rAbort, &sAbort, 1, 0, 0, NumProcs,ipWrk,ipSync );
  shmem_barrier_all(); 

  if (rAbort > 0) {
    if (MyProc == 0) fprintf(stderr, "Failed to allocate memory for local buckets.\n");
    goto failed_localbuckets;
  }
  GlobalBuckets = shmem_malloc(sizeof(u64Int)*NumProcs*(slot_size));

  sAbort = 0; if (! GlobalBuckets) sAbort = 1;

  shmem_barrier_all();
  shmem_int_sum_to_all(&rAbort, &sAbort, 1, 0, 0, NumProcs,ipWrk,ipSync );
  shmem_barrier_all(); 

  if (rAbort > 0) {
    if (MyProc == 0) fprintf(stderr, "Failed to allocate memory for global buckets.\n");
    goto failed_globalbuckets;
  }


  SendCnt = ProcNumUpdates; /*  SendCnt = 4 * LocalTableSize; */
  Ran = starts (4 * GlobalStartMyProc);

  PeCheckDone = XMALLOC ( s64Int, NumProcs);

  for (i=0; i<NumProcs; i++)
    PeCheckDone[i] = HPCC_FALSE;

  while(LocalAllDone == HPCC_FALSE){
    if (SendCnt > 0) {
      /* Initalize local buckets */
      for (i=0; i<NumProcs; i++){
        PeBucketBase = i * (slot_size);
        LocalBuckets[PeBucketBase+SLOT_CNT] = FIRST_SLOT;
        LocalBuckets[PeBucketBase+HPCC_DONE] = HPCC_FALSE;
      }

      /* Fill local buckets until one is full or out of data */
      NextSlot = FIRST_SLOT;
      while(NextSlot != (BUCKET_SIZE+FIRST_SLOT) && SendCnt>0 ) {
        Ran = (Ran << 1) ^ ((s64Int) Ran < ZERO64B ? POLY : ZERO64B);
        WhichPe = (Ran >> (logTableSize - logNumProcs)) & (NumProcs - 1);
        PeBucketBase = WhichPe * (slot_size);
        NextSlot = LocalBuckets[PeBucketBase+SLOT_CNT];
        LocalBuckets[PeBucketBase+NextSlot] = Ran;
        LocalBuckets[PeBucketBase+SLOT_CNT] = ++NextSlot;
        SendCnt--;
      }

      if (SendCnt == 0)
        for (i=0; i<NumProcs; i++)
          LocalBuckets[i*(slot_size)+HPCC_DONE] = HPCC_TRUE;

    } /* End of sending loop */

    shmem_barrier_all();

    LocalAllDone = HPCC_TRUE;

    /* Now move all the buckets to the appropriate pe */

    for (i=0 ; i<NumProcs ; i++)
         shmem_longlong_put(&GlobalBuckets[slot_size*MyProc],&LocalBuckets[slot_size*i],
                             slot_size,i);

    shmem_barrier_all(); 

    for (i = 0; i < NumProcs; i ++) {
      if(PeCheckDone[i] == HPCC_FALSE) {
        PeBucketBase = i * (BUCKET_SIZE+FIRST_SLOT);
        PeCheckDone[i] = GlobalBuckets[PeBucketBase+HPCC_DONE];
        for (j = FIRST_SLOT; j < GlobalBuckets[PeBucketBase+SLOT_CNT]; j ++) {
          RanTmp = GlobalBuckets[PeBucketBase+j];
          HPCC_Table[RanTmp & (LocalTableSize-1)] ^= RanTmp;
        }
        LocalAllDone &= PeCheckDone[i];
      }
    }
  }

  errors = 0;
  for (i=0; i<LocalTableSize; i++){
    if (HPCC_Table[i] != i + GlobalStartMyProc)
      errors++;
  }
  *NumErrors = errors;

  free( PeCheckDone );
  shmem_free( GlobalBuckets );

  failed_globalbuckets:
  shmem_free( LocalBuckets );

  failed_localbuckets:
  return;
}
