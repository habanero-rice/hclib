#include "hclib.h"
/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/**********************************************************************************************/

/*
 * Copyright (c) 1996 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to use, copy, modify, and distribute the Software without
 * restriction, provided the Software, including any modified copies made
 * under this license, is not distributed for a fee, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE MASSACHUSETTS INSTITUTE OF TECHNOLOGY BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * /WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the Massachusetts
 * Institute of Technology shall not be used in advertising or otherwise
 * to promote the sale, use or other dealings in this Software without
 * prior written authorization from the Massachusetts Institute of
 * Technology.
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "app-desc.h"
#include "bots.h"
#include "strassen.h"

/***********************************************************************
 * Naive sequential algorithm, for comparison purposes
 **********************************************************************/
void matrixmul(int n, REAL *A, int an, REAL *B, int bn, REAL *C, int cn)
{
   int i, j, k;
   REAL s;

   for (i = 0; i < n; ++i)
   { 
      for (j = 0; j < n; ++j)
      {
         s = 0.0;
         for (k = 0; k < n; ++k) s += ELEM(A, an, i, k) * ELEM(B, bn, k, j);
         ELEM(C, cn, i, j) = s;
      }
   }
}
/*****************************************************************************
**
** FastNaiveMatrixMultiply
**
** For small to medium sized matrices A, B, and C of size
** MatrixSize * MatrixSize this function performs the operation
** C = A x B efficiently.
**
** Note MatrixSize must be divisible by 8.
**
** INPUT:
**    C = (*C WRITE) Address of top left element of matrix C.
**    A = (*A IS READ ONLY) Address of top left element of matrix A.
**    B = (*B IS READ ONLY) Address of top left element of matrix B.
**    MatrixSize = Size of matrices (for n*n matrix, MatrixSize = n)
**    RowWidthA = Number of elements in memory between A[x,y] and A[x,y+1]
**    RowWidthB = Number of elements in memory between B[x,y] and B[x,y+1]
**    RowWidthC = Number of elements in memory between C[x,y] and C[x,y+1]
**
** OUTPUT:
**    C = (*C WRITE) Matrix C contains A x B. (Initial value of *C undefined.)
**
*****************************************************************************/
void FastNaiveMatrixMultiply(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
     unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB)
{ 
  /* Assumes size of real is 8 bytes */
  PTR RowWidthBInBytes = RowWidthB  << 3;
  PTR RowWidthAInBytes = RowWidthA << 3;
  PTR MatrixWidthInBytes = MatrixSize << 3;
  PTR RowIncrementC = ( RowWidthC - MatrixSize) << 3;
  unsigned Horizontal, Vertical;
  
  REAL *ARowStart = A;
  for (Vertical = 0; Vertical < MatrixSize; Vertical++) {
    for (Horizontal = 0; Horizontal < MatrixSize; Horizontal += 8) {
      REAL *BColumnStart = B + Horizontal;
      REAL FirstARowValue = *ARowStart++;

      REAL Sum0 = FirstARowValue * (*BColumnStart);
      REAL Sum1 = FirstARowValue * (*(BColumnStart+1));
      REAL Sum2 = FirstARowValue * (*(BColumnStart+2));
      REAL Sum3 = FirstARowValue * (*(BColumnStart+3));
      REAL Sum4 = FirstARowValue * (*(BColumnStart+4));
      REAL Sum5 = FirstARowValue * (*(BColumnStart+5));
      REAL Sum6 = FirstARowValue * (*(BColumnStart+6));
      REAL Sum7 = FirstARowValue * (*(BColumnStart+7));	

      unsigned Products;
      for (Products = 1; Products < MatrixSize; Products++) {
	REAL ARowValue = *ARowStart++;
	BColumnStart = (REAL*) (((PTR) BColumnStart) + RowWidthBInBytes);
	Sum0 += ARowValue * (*BColumnStart);
	Sum1 += ARowValue * (*(BColumnStart+1));
	Sum2 += ARowValue * (*(BColumnStart+2));
	Sum3 += ARowValue * (*(BColumnStart+3));
	Sum4 += ARowValue * (*(BColumnStart+4));
	Sum5 += ARowValue * (*(BColumnStart+5));
	Sum6 += ARowValue * (*(BColumnStart+6));
	Sum7 += ARowValue * (*(BColumnStart+7));	
      }
      ARowStart = (REAL*) ( ((PTR) ARowStart) - MatrixWidthInBytes);

      *(C) = Sum0;
      *(C+1) = Sum1;
      *(C+2) = Sum2;
      *(C+3) = Sum3;
      *(C+4) = Sum4;
      *(C+5) = Sum5;
      *(C+6) = Sum6;
      *(C+7) = Sum7;
      C+=8;
    }
    ARowStart = (REAL*) ( ((PTR) ARowStart) + RowWidthAInBytes );
    C = (REAL*) ( ((PTR) C) + RowIncrementC );
  }
}
/*****************************************************************************
**
** FastAdditiveNaiveMatrixMultiply
**
** For small to medium sized matrices A, B, and C of size
** MatrixSize * MatrixSize this function performs the operation
** C += A x B efficiently.
**
** Note MatrixSize must be divisible by 8.
**
** INPUT:
**    C = (*C READ/WRITE) Address of top left element of matrix C.
**    A = (*A IS READ ONLY) Address of top left element of matrix A.
**    B = (*B IS READ ONLY) Address of top left element of matrix B.
**    MatrixSize = Size of matrices (for n*n matrix, MatrixSize = n)
**    RowWidthA = Number of elements in memory between A[x,y] and A[x,y+1]
**    RowWidthB = Number of elements in memory between B[x,y] and B[x,y+1]
**    RowWidthC = Number of elements in memory between C[x,y] and C[x,y+1]
**
** OUTPUT:
**    C = (*C READ/WRITE) Matrix C contains C + A x B.
**
*****************************************************************************/
void FastAdditiveNaiveMatrixMultiply(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
     unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB)
{ 
  /* Assumes size of real is 8 bytes */
  PTR RowWidthBInBytes = RowWidthB  << 3;
  PTR RowWidthAInBytes = RowWidthA << 3;
  PTR MatrixWidthInBytes = MatrixSize << 3;
  PTR RowIncrementC = ( RowWidthC - MatrixSize) << 3;
  unsigned Horizontal, Vertical;
  
  REAL *ARowStart = A;
  for (Vertical = 0; Vertical < MatrixSize; Vertical++) {
    for (Horizontal = 0; Horizontal < MatrixSize; Horizontal += 8) {
      REAL *BColumnStart = B + Horizontal;

      REAL Sum0 = *C;
      REAL Sum1 = *(C+1);
      REAL Sum2 = *(C+2);
      REAL Sum3 = *(C+3);
      REAL Sum4 = *(C+4);
      REAL Sum5 = *(C+5);
      REAL Sum6 = *(C+6);
      REAL Sum7 = *(C+7);	

      unsigned Products;
      for (Products = 0; Products < MatrixSize; Products++) {
	REAL ARowValue = *ARowStart++;

	Sum0 += ARowValue * (*BColumnStart);
	Sum1 += ARowValue * (*(BColumnStart+1));
	Sum2 += ARowValue * (*(BColumnStart+2));
	Sum3 += ARowValue * (*(BColumnStart+3));
	Sum4 += ARowValue * (*(BColumnStart+4));
	Sum5 += ARowValue * (*(BColumnStart+5));
	Sum6 += ARowValue * (*(BColumnStart+6));
	Sum7 += ARowValue * (*(BColumnStart+7));

	BColumnStart = (REAL*) (((PTR) BColumnStart) + RowWidthBInBytes);

      }
      ARowStart = (REAL*) ( ((PTR) ARowStart) - MatrixWidthInBytes);

      *(C) = Sum0;
      *(C+1) = Sum1;
      *(C+2) = Sum2;
      *(C+3) = Sum3;
      *(C+4) = Sum4;
      *(C+5) = Sum5;
      *(C+6) = Sum6;
      *(C+7) = Sum7;
      C+=8;
    }

    ARowStart = (REAL*) ( ((PTR) ARowStart) + RowWidthAInBytes );
    C = (REAL*) ( ((PTR) C) + RowIncrementC );
  }
}
/*****************************************************************************
**
** MultiplyByDivideAndConquer
**
** For medium to medium-large (would you like fries with that) sized
** matrices A, B, and C of size MatrixSize * MatrixSize this function
** efficiently performs the operation
**    C  = A x B (if AdditiveMode == 0)
**    C += A x B (if AdditiveMode != 0)
**
** Note MatrixSize must be divisible by 16.
**
** INPUT:
**    C = (*C READ/WRITE) Address of top left element of matrix C.
**    A = (*A IS READ ONLY) Address of top left element of matrix A.
**    B = (*B IS READ ONLY) Address of top left element of matrix B.
**    MatrixSize = Size of matrices (for n*n matrix, MatrixSize = n)
**    RowWidthA = Number of elements in memory between A[x,y] and A[x,y+1]
**    RowWidthB = Number of elements in memory between B[x,y] and B[x,y+1]
**    RowWidthC = Number of elements in memory between C[x,y] and C[x,y+1]
**    AdditiveMode = 0 if we want C = A x B, otherwise we'll do C += A x B
**
** OUTPUT:
**    C (+)= A x B. (+ if AdditiveMode != 0)
**
*****************************************************************************/
void MultiplyByDivideAndConquer(REAL *C, REAL *A, REAL *B,
				     unsigned MatrixSize,
				     unsigned RowWidthC,
				     unsigned RowWidthA,
				     unsigned RowWidthB,
				     int AdditiveMode
				    )
{
  REAL  *A01, *A10, *A11, *B01, *B10, *B11, *C01, *C10, *C11;
  unsigned QuadrantSize = MatrixSize >> 1;

  /* partition the matrix */
  A01 = A + QuadrantSize;
  A10 = A + RowWidthA * QuadrantSize;
  A11 = A10 + QuadrantSize;

  B01 = B + QuadrantSize;
  B10 = B + RowWidthB * QuadrantSize;
  B11 = B10 + QuadrantSize;

  C01 = C + QuadrantSize;
  C10 = C + RowWidthC * QuadrantSize;
  C11 = C10 + QuadrantSize;

  if (QuadrantSize > SizeAtWhichNaiveAlgorithmIsMoreEfficient) {

    MultiplyByDivideAndConquer(C, A, B, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C01, A, B01, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C11, A10, B01, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C10, A10, B, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C, A01, B10, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     1);

    MultiplyByDivideAndConquer(C01, A01, B11, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     1);

    MultiplyByDivideAndConquer(C11, A11, B11, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     1);

    MultiplyByDivideAndConquer(C10, A11, B10, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     1);
    
  } else {

    if (AdditiveMode) {
      FastAdditiveNaiveMatrixMultiply(C, A, B, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastAdditiveNaiveMatrixMultiply(C01, A, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastAdditiveNaiveMatrixMultiply(C11, A10, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastAdditiveNaiveMatrixMultiply(C10, A10, B, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
    } else {
      
      FastNaiveMatrixMultiply(C, A, B, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastNaiveMatrixMultiply(C01, A, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastNaiveMatrixMultiply(C11, A10, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastNaiveMatrixMultiply(C10, A10, B, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
    }

    FastAdditiveNaiveMatrixMultiply(C, A01, B10, QuadrantSize,
				    RowWidthC, RowWidthA, RowWidthB);
    
    FastAdditiveNaiveMatrixMultiply(C01, A01, B11, QuadrantSize,
				    RowWidthC, RowWidthA, RowWidthB);
    
    FastAdditiveNaiveMatrixMultiply(C11, A11, B11, QuadrantSize,
				    RowWidthC, RowWidthA, RowWidthB);
    
    FastAdditiveNaiveMatrixMultiply(C10, A11, B10, QuadrantSize,
				    RowWidthC, RowWidthA, RowWidthB);
  }
  return;
}
/*****************************************************************************
**
** OptimizedStrassenMultiply
**
** For large matrices A, B, and C of size MatrixSize * MatrixSize this
** function performs the operation C = A x B efficiently.
**
** INPUT:
**    C = (*C WRITE) Address of top left element of matrix C.
**    A = (*A IS READ ONLY) Address of top left element of matrix A.
**    B = (*B IS READ ONLY) Address of top left element of matrix B.
**    MatrixSize = Size of matrices (for n*n matrix, MatrixSize = n)
**    RowWidthA = Number of elements in memory between A[x,y] and A[x,y+1]
**    RowWidthB = Number of elements in memory between B[x,y] and B[x,y+1]
**    RowWidthC = Number of elements in memory between C[x,y] and C[x,y+1]
**
** OUTPUT:
**    C = (*C WRITE) Matrix C contains A x B. (Initial value of *C undefined.)
**
*****************************************************************************/
void OptimizedStrassenMultiply_seq(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
     unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB, int Depth)
{
  unsigned QuadrantSize = MatrixSize >> 1; /* MatixSize / 2 */
  unsigned QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize
                                 + 32;
  unsigned Column, Row;
  
  /************************************************************************
  ** For each matrix A, B, and C, we'll want pointers to each quandrant
  ** in the matrix. These quandrants will be addressed as follows:
  **  --        --
  **  | A11  A12 |
  **  |          |
  **  | A21  A22 |
  **  --        --
  ************************************************************************/
  REAL /* *A11, *B11, *C11, */ *A12, *B12, *C12,
       *A21, *B21, *C21, *A22, *B22, *C22;

  REAL *S1,*S2,*S3,*S4,*S5,*S6,*S7,*S8,*M2,*M5,*T1sMULT;
  const int NumberOfVariables = 11;

  PTR TempMatrixOffset = 0;
  PTR MatrixOffsetA = 0;
  PTR MatrixOffsetB = 0;

  char *Heap;
  void *StartHeap;

  /* Distance between the end of a matrix row and the start of the next row */
  PTR RowIncrementA = ( RowWidthA - QuadrantSize ) << 3;
  PTR RowIncrementB = ( RowWidthB - QuadrantSize ) << 3;
  PTR RowIncrementC = ( RowWidthC - QuadrantSize ) << 3;

  if (MatrixSize <= bots_app_cutoff_value) {
    MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
    return;
  }

  /* Initialize quandrant matrices */
  A12 = A + QuadrantSize;
  B12 = B + QuadrantSize;
  C12 = C + QuadrantSize;
  A21 = A + (RowWidthA * QuadrantSize);
  B21 = B + (RowWidthB * QuadrantSize);
  C21 = C + (RowWidthC * QuadrantSize);
  A22 = A21 + QuadrantSize;
  B22 = B21 + QuadrantSize;
  C22 = C21 + QuadrantSize;

  /* Allocate Heap Space Here */
  StartHeap = Heap = (char *)malloc(QuadrantSizeInBytes * NumberOfVariables);
  /* ensure that heap is on cache boundary */
  if ( ((PTR) Heap) & 31)
     Heap = (char*) ( ((PTR) Heap) + 32 - ( ((PTR) Heap) & 31) );
  
  /* Distribute the heap space over the variables */
  S1 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S3 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S4 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S6 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S7 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S8 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  M2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  M5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  T1sMULT = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  
  /***************************************************************************
  ** Step through all columns row by row (vertically)
  ** (jumps in memory by RowWidth => bad locality)
  ** (but we want the best locality on the innermost loop)
  ***************************************************************************/
  for (Row = 0; Row < QuadrantSize; Row++) {
    
    /*************************************************************************
    ** Step through each row horizontally (addressing elements in each column)
    ** (jumps linearly througn memory => good locality)
    *************************************************************************/
    for (Column = 0; Column < QuadrantSize; Column++) {
      
      /***********************************************************
      ** Within this loop, the following holds for MatrixOffset:
      ** MatrixOffset = (Row * RowWidth) + Column
      ** (note: that the unit of the offset is number of reals)
      ***********************************************************/
      /* Element of Global Matrix, such as A, B, C */
      #define E(Matrix)   (* (REAL*) ( ((PTR) Matrix) + TempMatrixOffset ) )
      #define EA(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetA ) )
      #define EB(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetB ) )

      /* FIXME - may pay to expand these out - got higher speed-ups below */
      /* S4 = A12 - ( S2 = ( S1 = A21 + A22 ) - A11 ) */
      E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A) );

      /* S8 = (S6 = B22 - ( S5 = B12 - B11 ) ) - B21 */
      E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B) ) ) - EB(B21);

      /* S3 = A11 - A21 */
      E(S3) = EA(A) - EA(A21);
      
      /* S7 = B22 - B12 */
      E(S7) = EB(B22) - EB(B12);

      TempMatrixOffset += sizeof(REAL);
      MatrixOffsetA += sizeof(REAL);
      MatrixOffsetB += sizeof(REAL);
    } /* end row loop*/

    MatrixOffsetA += RowIncrementA;
    MatrixOffsetB += RowIncrementB;
  } /* end column loop */

  /* M2 = A11 x B11 */
  OptimizedStrassenMultiply_seq(M2, A, B, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);

  /* M5 = S1 * S5 */
  OptimizedStrassenMultiply_seq(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of T1 = S2 x S6 + M2 */
  OptimizedStrassenMultiply_seq(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of T2 = T1 + S3 x S7 */
  OptimizedStrassenMultiply_seq(C22, S3, S7, QuadrantSize, RowWidthC /*FIXME*/, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of C11 = M2 + A12 * B21 */
  OptimizedStrassenMultiply_seq(C, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
  
  /* Step 1 of C12 = S4 x B22 + T1 + M5 */
  OptimizedStrassenMultiply_seq(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);

  /* Step 1 of C21 = T2 - A22 * S8 */
  OptimizedStrassenMultiply_seq(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);

  /***************************************************************************
  ** Step through all columns row by row (vertically)
  ** (jumps in memory by RowWidth => bad locality)
  ** (but we want the best locality on the innermost loop)
  ***************************************************************************/
  for (Row = 0; Row < QuadrantSize; Row++) {
    /*************************************************************************
    ** Step through each row horizontally (addressing elements in each column)
    ** (jumps linearly througn memory => good locality)
    *************************************************************************/
    for (Column = 0; Column < QuadrantSize; Column += 4) {
      REAL LocalM5_0 = *(M5);
      REAL LocalM5_1 = *(M5+1);
      REAL LocalM5_2 = *(M5+2);
      REAL LocalM5_3 = *(M5+3);
      REAL LocalM2_0 = *(M2);
      REAL LocalM2_1 = *(M2+1);
      REAL LocalM2_2 = *(M2+2);
      REAL LocalM2_3 = *(M2+3);
      REAL T1_0 = *(T1sMULT) + LocalM2_0;
      REAL T1_1 = *(T1sMULT+1) + LocalM2_1;
      REAL T1_2 = *(T1sMULT+2) + LocalM2_2;
      REAL T1_3 = *(T1sMULT+3) + LocalM2_3;
      REAL T2_0 = *(C22) + T1_0;
      REAL T2_1 = *(C22+1) + T1_1;
      REAL T2_2 = *(C22+2) + T1_2;
      REAL T2_3 = *(C22+3) + T1_3;
      (*(C))   += LocalM2_0;
      (*(C+1)) += LocalM2_1;
      (*(C+2)) += LocalM2_2;
      (*(C+3)) += LocalM2_3;
      (*(C12))   += LocalM5_0 + T1_0;
      (*(C12+1)) += LocalM5_1 + T1_1;
      (*(C12+2)) += LocalM5_2 + T1_2;
      (*(C12+3)) += LocalM5_3 + T1_3;
      (*(C22))   = LocalM5_0 + T2_0;
      (*(C22+1)) = LocalM5_1 + T2_1;
      (*(C22+2)) = LocalM5_2 + T2_2;
      (*(C22+3)) = LocalM5_3 + T2_3;
      (*(C21  )) = (- *(C21  )) + T2_0;
      (*(C21+1)) = (- *(C21+1)) + T2_1;
      (*(C21+2)) = (- *(C21+2)) + T2_2;
      (*(C21+3)) = (- *(C21+3)) + T2_3;
      M5 += 4;
      M2 += 4;
      T1sMULT += 4;
      C += 4;
      C12 += 4;
      C21 += 4;
      C22 += 4;
    }
    C = (REAL*) ( ((PTR) C ) + RowIncrementC);
    C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
    C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
    C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
  }
  free(StartHeap);
}
typedef struct _pragma679 {
    unsigned int (*QuadrantSize_ptr);
    unsigned int (*QuadrantSizeInBytes_ptr);
    unsigned int (*Column_ptr);
    unsigned int (*Row_ptr);
    REAL (*(*A12_ptr));
    REAL (*(*B12_ptr));
    REAL (*(*C12_ptr));
    REAL (*(*A21_ptr));
    REAL (*(*B21_ptr));
    REAL (*(*C21_ptr));
    REAL (*(*A22_ptr));
    REAL (*(*B22_ptr));
    REAL (*(*C22_ptr));
    REAL (*(*S1_ptr));
    REAL (*(*S2_ptr));
    REAL (*(*S3_ptr));
    REAL (*(*S4_ptr));
    REAL (*(*S5_ptr));
    REAL (*(*S6_ptr));
    REAL (*(*S7_ptr));
    REAL (*(*S8_ptr));
    REAL (*(*M2_ptr));
    REAL (*(*M5_ptr));
    REAL (*(*T1sMULT_ptr));
    const int (*NumberOfVariables_ptr);
    PTR (*TempMatrixOffset_ptr);
    PTR (*MatrixOffsetA_ptr);
    PTR (*MatrixOffsetB_ptr);
    char (*(*Heap_ptr));
    void (*(*StartHeap_ptr));
    PTR (*RowIncrementA_ptr);
    PTR (*RowIncrementB_ptr);
    PTR (*RowIncrementC_ptr);
    REAL (*(*C_ptr));
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    unsigned int (*MatrixSize_ptr);
    unsigned int (*RowWidthC_ptr);
    unsigned int (*RowWidthA_ptr);
    unsigned int (*RowWidthB_ptr);
    int (*Depth_ptr);
 } pragma679;

typedef struct _pragma683 {
    unsigned int (*QuadrantSize_ptr);
    unsigned int (*QuadrantSizeInBytes_ptr);
    unsigned int (*Column_ptr);
    unsigned int (*Row_ptr);
    REAL (*(*A12_ptr));
    REAL (*(*B12_ptr));
    REAL (*(*C12_ptr));
    REAL (*(*A21_ptr));
    REAL (*(*B21_ptr));
    REAL (*(*C21_ptr));
    REAL (*(*A22_ptr));
    REAL (*(*B22_ptr));
    REAL (*(*C22_ptr));
    REAL (*(*S1_ptr));
    REAL (*(*S2_ptr));
    REAL (*(*S3_ptr));
    REAL (*(*S4_ptr));
    REAL (*(*S5_ptr));
    REAL (*(*S6_ptr));
    REAL (*(*S7_ptr));
    REAL (*(*S8_ptr));
    REAL (*(*M2_ptr));
    REAL (*(*M5_ptr));
    REAL (*(*T1sMULT_ptr));
    const int (*NumberOfVariables_ptr);
    PTR (*TempMatrixOffset_ptr);
    PTR (*MatrixOffsetA_ptr);
    PTR (*MatrixOffsetB_ptr);
    char (*(*Heap_ptr));
    void (*(*StartHeap_ptr));
    PTR (*RowIncrementA_ptr);
    PTR (*RowIncrementB_ptr);
    PTR (*RowIncrementC_ptr);
    REAL (*(*C_ptr));
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    unsigned int (*MatrixSize_ptr);
    unsigned int (*RowWidthC_ptr);
    unsigned int (*RowWidthA_ptr);
    unsigned int (*RowWidthB_ptr);
    int (*Depth_ptr);
 } pragma683;

typedef struct _pragma687 {
    unsigned int (*QuadrantSize_ptr);
    unsigned int (*QuadrantSizeInBytes_ptr);
    unsigned int (*Column_ptr);
    unsigned int (*Row_ptr);
    REAL (*(*A12_ptr));
    REAL (*(*B12_ptr));
    REAL (*(*C12_ptr));
    REAL (*(*A21_ptr));
    REAL (*(*B21_ptr));
    REAL (*(*C21_ptr));
    REAL (*(*A22_ptr));
    REAL (*(*B22_ptr));
    REAL (*(*C22_ptr));
    REAL (*(*S1_ptr));
    REAL (*(*S2_ptr));
    REAL (*(*S3_ptr));
    REAL (*(*S4_ptr));
    REAL (*(*S5_ptr));
    REAL (*(*S6_ptr));
    REAL (*(*S7_ptr));
    REAL (*(*S8_ptr));
    REAL (*(*M2_ptr));
    REAL (*(*M5_ptr));
    REAL (*(*T1sMULT_ptr));
    const int (*NumberOfVariables_ptr);
    PTR (*TempMatrixOffset_ptr);
    PTR (*MatrixOffsetA_ptr);
    PTR (*MatrixOffsetB_ptr);
    char (*(*Heap_ptr));
    void (*(*StartHeap_ptr));
    PTR (*RowIncrementA_ptr);
    PTR (*RowIncrementB_ptr);
    PTR (*RowIncrementC_ptr);
    REAL (*(*C_ptr));
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    unsigned int (*MatrixSize_ptr);
    unsigned int (*RowWidthC_ptr);
    unsigned int (*RowWidthA_ptr);
    unsigned int (*RowWidthB_ptr);
    int (*Depth_ptr);
 } pragma687;

typedef struct _pragma691 {
    unsigned int (*QuadrantSize_ptr);
    unsigned int (*QuadrantSizeInBytes_ptr);
    unsigned int (*Column_ptr);
    unsigned int (*Row_ptr);
    REAL (*(*A12_ptr));
    REAL (*(*B12_ptr));
    REAL (*(*C12_ptr));
    REAL (*(*A21_ptr));
    REAL (*(*B21_ptr));
    REAL (*(*C21_ptr));
    REAL (*(*A22_ptr));
    REAL (*(*B22_ptr));
    REAL (*(*C22_ptr));
    REAL (*(*S1_ptr));
    REAL (*(*S2_ptr));
    REAL (*(*S3_ptr));
    REAL (*(*S4_ptr));
    REAL (*(*S5_ptr));
    REAL (*(*S6_ptr));
    REAL (*(*S7_ptr));
    REAL (*(*S8_ptr));
    REAL (*(*M2_ptr));
    REAL (*(*M5_ptr));
    REAL (*(*T1sMULT_ptr));
    const int (*NumberOfVariables_ptr);
    PTR (*TempMatrixOffset_ptr);
    PTR (*MatrixOffsetA_ptr);
    PTR (*MatrixOffsetB_ptr);
    char (*(*Heap_ptr));
    void (*(*StartHeap_ptr));
    PTR (*RowIncrementA_ptr);
    PTR (*RowIncrementB_ptr);
    PTR (*RowIncrementC_ptr);
    REAL (*(*C_ptr));
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    unsigned int (*MatrixSize_ptr);
    unsigned int (*RowWidthC_ptr);
    unsigned int (*RowWidthA_ptr);
    unsigned int (*RowWidthB_ptr);
    int (*Depth_ptr);
 } pragma691;

typedef struct _pragma695 {
    unsigned int (*QuadrantSize_ptr);
    unsigned int (*QuadrantSizeInBytes_ptr);
    unsigned int (*Column_ptr);
    unsigned int (*Row_ptr);
    REAL (*(*A12_ptr));
    REAL (*(*B12_ptr));
    REAL (*(*C12_ptr));
    REAL (*(*A21_ptr));
    REAL (*(*B21_ptr));
    REAL (*(*C21_ptr));
    REAL (*(*A22_ptr));
    REAL (*(*B22_ptr));
    REAL (*(*C22_ptr));
    REAL (*(*S1_ptr));
    REAL (*(*S2_ptr));
    REAL (*(*S3_ptr));
    REAL (*(*S4_ptr));
    REAL (*(*S5_ptr));
    REAL (*(*S6_ptr));
    REAL (*(*S7_ptr));
    REAL (*(*S8_ptr));
    REAL (*(*M2_ptr));
    REAL (*(*M5_ptr));
    REAL (*(*T1sMULT_ptr));
    const int (*NumberOfVariables_ptr);
    PTR (*TempMatrixOffset_ptr);
    PTR (*MatrixOffsetA_ptr);
    PTR (*MatrixOffsetB_ptr);
    char (*(*Heap_ptr));
    void (*(*StartHeap_ptr));
    PTR (*RowIncrementA_ptr);
    PTR (*RowIncrementB_ptr);
    PTR (*RowIncrementC_ptr);
    REAL (*(*C_ptr));
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    unsigned int (*MatrixSize_ptr);
    unsigned int (*RowWidthC_ptr);
    unsigned int (*RowWidthA_ptr);
    unsigned int (*RowWidthB_ptr);
    int (*Depth_ptr);
 } pragma695;

typedef struct _pragma699 {
    unsigned int (*QuadrantSize_ptr);
    unsigned int (*QuadrantSizeInBytes_ptr);
    unsigned int (*Column_ptr);
    unsigned int (*Row_ptr);
    REAL (*(*A12_ptr));
    REAL (*(*B12_ptr));
    REAL (*(*C12_ptr));
    REAL (*(*A21_ptr));
    REAL (*(*B21_ptr));
    REAL (*(*C21_ptr));
    REAL (*(*A22_ptr));
    REAL (*(*B22_ptr));
    REAL (*(*C22_ptr));
    REAL (*(*S1_ptr));
    REAL (*(*S2_ptr));
    REAL (*(*S3_ptr));
    REAL (*(*S4_ptr));
    REAL (*(*S5_ptr));
    REAL (*(*S6_ptr));
    REAL (*(*S7_ptr));
    REAL (*(*S8_ptr));
    REAL (*(*M2_ptr));
    REAL (*(*M5_ptr));
    REAL (*(*T1sMULT_ptr));
    const int (*NumberOfVariables_ptr);
    PTR (*TempMatrixOffset_ptr);
    PTR (*MatrixOffsetA_ptr);
    PTR (*MatrixOffsetB_ptr);
    char (*(*Heap_ptr));
    void (*(*StartHeap_ptr));
    PTR (*RowIncrementA_ptr);
    PTR (*RowIncrementB_ptr);
    PTR (*RowIncrementC_ptr);
    REAL (*(*C_ptr));
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    unsigned int (*MatrixSize_ptr);
    unsigned int (*RowWidthC_ptr);
    unsigned int (*RowWidthA_ptr);
    unsigned int (*RowWidthB_ptr);
    int (*Depth_ptr);
 } pragma699;

typedef struct _pragma703 {
    unsigned int (*QuadrantSize_ptr);
    unsigned int (*QuadrantSizeInBytes_ptr);
    unsigned int (*Column_ptr);
    unsigned int (*Row_ptr);
    REAL (*(*A12_ptr));
    REAL (*(*B12_ptr));
    REAL (*(*C12_ptr));
    REAL (*(*A21_ptr));
    REAL (*(*B21_ptr));
    REAL (*(*C21_ptr));
    REAL (*(*A22_ptr));
    REAL (*(*B22_ptr));
    REAL (*(*C22_ptr));
    REAL (*(*S1_ptr));
    REAL (*(*S2_ptr));
    REAL (*(*S3_ptr));
    REAL (*(*S4_ptr));
    REAL (*(*S5_ptr));
    REAL (*(*S6_ptr));
    REAL (*(*S7_ptr));
    REAL (*(*S8_ptr));
    REAL (*(*M2_ptr));
    REAL (*(*M5_ptr));
    REAL (*(*T1sMULT_ptr));
    const int (*NumberOfVariables_ptr);
    PTR (*TempMatrixOffset_ptr);
    PTR (*MatrixOffsetA_ptr);
    PTR (*MatrixOffsetB_ptr);
    char (*(*Heap_ptr));
    void (*(*StartHeap_ptr));
    PTR (*RowIncrementA_ptr);
    PTR (*RowIncrementB_ptr);
    PTR (*RowIncrementC_ptr);
    REAL (*(*C_ptr));
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    unsigned int (*MatrixSize_ptr);
    unsigned int (*RowWidthC_ptr);
    unsigned int (*RowWidthA_ptr);
    unsigned int (*RowWidthB_ptr);
    int (*Depth_ptr);
 } pragma703;

static void pragma679_hclib_async(void *____arg);
static void pragma683_hclib_async(void *____arg);
static void pragma687_hclib_async(void *____arg);
static void pragma691_hclib_async(void *____arg);
static void pragma695_hclib_async(void *____arg);
static void pragma699_hclib_async(void *____arg);
static void pragma703_hclib_async(void *____arg);
void OptimizedStrassenMultiply_par(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
     unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB, int Depth)
{
  unsigned QuadrantSize = MatrixSize >> 1; /* MatixSize / 2 */
  unsigned QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize
                                 + 32;
  unsigned Column, Row;
  
  /************************************************************************
  ** For each matrix A, B, and C, we'll want pointers to each quandrant
  ** in the matrix. These quandrants will be addressed as follows:
  **  --        --
  **  | A11  A12 |
  **  |          |
  **  | A21  A22 |
  **  --        --
  ************************************************************************/
  REAL /* *A11, *B11, *C11, */ *A12, *B12, *C12,
       *A21, *B21, *C21, *A22, *B22, *C22;

  REAL *S1,*S2,*S3,*S4,*S5,*S6,*S7,*S8,*M2,*M5,*T1sMULT;
  const int NumberOfVariables = 11;

  PTR TempMatrixOffset = 0;
  PTR MatrixOffsetA = 0;
  PTR MatrixOffsetB = 0;

  char *Heap;
  void *StartHeap;

  /* Distance between the end of a matrix row and the start of the next row */
  PTR RowIncrementA = ( RowWidthA - QuadrantSize ) << 3;
  PTR RowIncrementB = ( RowWidthB - QuadrantSize ) << 3;
  PTR RowIncrementC = ( RowWidthC - QuadrantSize ) << 3;

  if (MatrixSize <= bots_app_cutoff_value) {
    MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
    return;
  }

  /* Initialize quandrant matrices */
  A12 = A + QuadrantSize;
  B12 = B + QuadrantSize;
  C12 = C + QuadrantSize;
  A21 = A + (RowWidthA * QuadrantSize);
  B21 = B + (RowWidthB * QuadrantSize);
  C21 = C + (RowWidthC * QuadrantSize);
  A22 = A21 + QuadrantSize;
  B22 = B21 + QuadrantSize;
  C22 = C21 + QuadrantSize;

  /* Allocate Heap Space Here */
  StartHeap = Heap = (char *)malloc(QuadrantSizeInBytes * NumberOfVariables);
  /* ensure that heap is on cache boundary */
  if ( ((PTR) Heap) & 31)
     Heap = (char*) ( ((PTR) Heap) + 32 - ( ((PTR) Heap) & 31) );
  
  /* Distribute the heap space over the variables */
  S1 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S3 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S4 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S6 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S7 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  S8 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  M2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  M5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  T1sMULT = (REAL*) Heap; Heap += QuadrantSizeInBytes;
  
  /***************************************************************************
  ** Step through all columns row by row (vertically)
  ** (jumps in memory by RowWidth => bad locality)
  ** (but we want the best locality on the innermost loop)
  ***************************************************************************/
  for (Row = 0; Row < QuadrantSize; Row++) {
    
    /*************************************************************************
    ** Step through each row horizontally (addressing elements in each column)
    ** (jumps linearly througn memory => good locality)
    *************************************************************************/
    for (Column = 0; Column < QuadrantSize; Column++) {
      
      /***********************************************************
      ** Within this loop, the following holds for MatrixOffset:
      ** MatrixOffset = (Row * RowWidth) + Column
      ** (note: that the unit of the offset is number of reals)
      ***********************************************************/
      /* Element of Global Matrix, such as A, B, C */
      #define E(Matrix)   (* (REAL*) ( ((PTR) Matrix) + TempMatrixOffset ) )
      #define EA(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetA ) )
      #define EB(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetB ) )

      /* FIXME - may pay to expand these out - got higher speed-ups below */
      /* S4 = A12 - ( S2 = ( S1 = A21 + A22 ) - A11 ) */
      E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A) );

      /* S8 = (S6 = B22 - ( S5 = B12 - B11 ) ) - B21 */
      E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B) ) ) - EB(B21);

      /* S3 = A11 - A21 */
      E(S3) = EA(A) - EA(A21);
      
      /* S7 = B22 - B12 */
      E(S7) = EB(B22) - EB(B12);

      TempMatrixOffset += sizeof(REAL);
      MatrixOffsetA += sizeof(REAL);
      MatrixOffsetB += sizeof(REAL);
    } /* end row loop*/

    MatrixOffsetA += RowIncrementA;
    MatrixOffsetB += RowIncrementB;
  } /* end column loop */

  /* M2 = A11 x B11 */
 { 
pragma679 *new_ctx = (pragma679 *)malloc(sizeof(pragma679));
new_ctx->QuadrantSize_ptr = &(QuadrantSize);
new_ctx->QuadrantSizeInBytes_ptr = &(QuadrantSizeInBytes);
new_ctx->Column_ptr = &(Column);
new_ctx->Row_ptr = &(Row);
new_ctx->A12_ptr = &(A12);
new_ctx->B12_ptr = &(B12);
new_ctx->C12_ptr = &(C12);
new_ctx->A21_ptr = &(A21);
new_ctx->B21_ptr = &(B21);
new_ctx->C21_ptr = &(C21);
new_ctx->A22_ptr = &(A22);
new_ctx->B22_ptr = &(B22);
new_ctx->C22_ptr = &(C22);
new_ctx->S1_ptr = &(S1);
new_ctx->S2_ptr = &(S2);
new_ctx->S3_ptr = &(S3);
new_ctx->S4_ptr = &(S4);
new_ctx->S5_ptr = &(S5);
new_ctx->S6_ptr = &(S6);
new_ctx->S7_ptr = &(S7);
new_ctx->S8_ptr = &(S8);
new_ctx->M2_ptr = &(M2);
new_ctx->M5_ptr = &(M5);
new_ctx->T1sMULT_ptr = &(T1sMULT);
new_ctx->NumberOfVariables_ptr = &(NumberOfVariables);
new_ctx->TempMatrixOffset_ptr = &(TempMatrixOffset);
new_ctx->MatrixOffsetA_ptr = &(MatrixOffsetA);
new_ctx->MatrixOffsetB_ptr = &(MatrixOffsetB);
new_ctx->Heap_ptr = &(Heap);
new_ctx->StartHeap_ptr = &(StartHeap);
new_ctx->RowIncrementA_ptr = &(RowIncrementA);
new_ctx->RowIncrementB_ptr = &(RowIncrementB);
new_ctx->RowIncrementC_ptr = &(RowIncrementC);
new_ctx->C_ptr = &(C);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->MatrixSize_ptr = &(MatrixSize);
new_ctx->RowWidthC_ptr = &(RowWidthC);
new_ctx->RowWidthA_ptr = &(RowWidthA);
new_ctx->RowWidthB_ptr = &(RowWidthB);
new_ctx->Depth_ptr = &(Depth);
hclib_async(pragma679_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* M5 = S1 * S5 */
 { 
pragma683 *new_ctx = (pragma683 *)malloc(sizeof(pragma683));
new_ctx->QuadrantSize_ptr = &(QuadrantSize);
new_ctx->QuadrantSizeInBytes_ptr = &(QuadrantSizeInBytes);
new_ctx->Column_ptr = &(Column);
new_ctx->Row_ptr = &(Row);
new_ctx->A12_ptr = &(A12);
new_ctx->B12_ptr = &(B12);
new_ctx->C12_ptr = &(C12);
new_ctx->A21_ptr = &(A21);
new_ctx->B21_ptr = &(B21);
new_ctx->C21_ptr = &(C21);
new_ctx->A22_ptr = &(A22);
new_ctx->B22_ptr = &(B22);
new_ctx->C22_ptr = &(C22);
new_ctx->S1_ptr = &(S1);
new_ctx->S2_ptr = &(S2);
new_ctx->S3_ptr = &(S3);
new_ctx->S4_ptr = &(S4);
new_ctx->S5_ptr = &(S5);
new_ctx->S6_ptr = &(S6);
new_ctx->S7_ptr = &(S7);
new_ctx->S8_ptr = &(S8);
new_ctx->M2_ptr = &(M2);
new_ctx->M5_ptr = &(M5);
new_ctx->T1sMULT_ptr = &(T1sMULT);
new_ctx->NumberOfVariables_ptr = &(NumberOfVariables);
new_ctx->TempMatrixOffset_ptr = &(TempMatrixOffset);
new_ctx->MatrixOffsetA_ptr = &(MatrixOffsetA);
new_ctx->MatrixOffsetB_ptr = &(MatrixOffsetB);
new_ctx->Heap_ptr = &(Heap);
new_ctx->StartHeap_ptr = &(StartHeap);
new_ctx->RowIncrementA_ptr = &(RowIncrementA);
new_ctx->RowIncrementB_ptr = &(RowIncrementB);
new_ctx->RowIncrementC_ptr = &(RowIncrementC);
new_ctx->C_ptr = &(C);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->MatrixSize_ptr = &(MatrixSize);
new_ctx->RowWidthC_ptr = &(RowWidthC);
new_ctx->RowWidthA_ptr = &(RowWidthA);
new_ctx->RowWidthB_ptr = &(RowWidthB);
new_ctx->Depth_ptr = &(Depth);
hclib_async(pragma683_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of T1 = S2 x S6 + M2 */
 { 
pragma687 *new_ctx = (pragma687 *)malloc(sizeof(pragma687));
new_ctx->QuadrantSize_ptr = &(QuadrantSize);
new_ctx->QuadrantSizeInBytes_ptr = &(QuadrantSizeInBytes);
new_ctx->Column_ptr = &(Column);
new_ctx->Row_ptr = &(Row);
new_ctx->A12_ptr = &(A12);
new_ctx->B12_ptr = &(B12);
new_ctx->C12_ptr = &(C12);
new_ctx->A21_ptr = &(A21);
new_ctx->B21_ptr = &(B21);
new_ctx->C21_ptr = &(C21);
new_ctx->A22_ptr = &(A22);
new_ctx->B22_ptr = &(B22);
new_ctx->C22_ptr = &(C22);
new_ctx->S1_ptr = &(S1);
new_ctx->S2_ptr = &(S2);
new_ctx->S3_ptr = &(S3);
new_ctx->S4_ptr = &(S4);
new_ctx->S5_ptr = &(S5);
new_ctx->S6_ptr = &(S6);
new_ctx->S7_ptr = &(S7);
new_ctx->S8_ptr = &(S8);
new_ctx->M2_ptr = &(M2);
new_ctx->M5_ptr = &(M5);
new_ctx->T1sMULT_ptr = &(T1sMULT);
new_ctx->NumberOfVariables_ptr = &(NumberOfVariables);
new_ctx->TempMatrixOffset_ptr = &(TempMatrixOffset);
new_ctx->MatrixOffsetA_ptr = &(MatrixOffsetA);
new_ctx->MatrixOffsetB_ptr = &(MatrixOffsetB);
new_ctx->Heap_ptr = &(Heap);
new_ctx->StartHeap_ptr = &(StartHeap);
new_ctx->RowIncrementA_ptr = &(RowIncrementA);
new_ctx->RowIncrementB_ptr = &(RowIncrementB);
new_ctx->RowIncrementC_ptr = &(RowIncrementC);
new_ctx->C_ptr = &(C);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->MatrixSize_ptr = &(MatrixSize);
new_ctx->RowWidthC_ptr = &(RowWidthC);
new_ctx->RowWidthA_ptr = &(RowWidthA);
new_ctx->RowWidthB_ptr = &(RowWidthB);
new_ctx->Depth_ptr = &(Depth);
hclib_async(pragma687_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of T2 = T1 + S3 x S7 */
 { 
pragma691 *new_ctx = (pragma691 *)malloc(sizeof(pragma691));
new_ctx->QuadrantSize_ptr = &(QuadrantSize);
new_ctx->QuadrantSizeInBytes_ptr = &(QuadrantSizeInBytes);
new_ctx->Column_ptr = &(Column);
new_ctx->Row_ptr = &(Row);
new_ctx->A12_ptr = &(A12);
new_ctx->B12_ptr = &(B12);
new_ctx->C12_ptr = &(C12);
new_ctx->A21_ptr = &(A21);
new_ctx->B21_ptr = &(B21);
new_ctx->C21_ptr = &(C21);
new_ctx->A22_ptr = &(A22);
new_ctx->B22_ptr = &(B22);
new_ctx->C22_ptr = &(C22);
new_ctx->S1_ptr = &(S1);
new_ctx->S2_ptr = &(S2);
new_ctx->S3_ptr = &(S3);
new_ctx->S4_ptr = &(S4);
new_ctx->S5_ptr = &(S5);
new_ctx->S6_ptr = &(S6);
new_ctx->S7_ptr = &(S7);
new_ctx->S8_ptr = &(S8);
new_ctx->M2_ptr = &(M2);
new_ctx->M5_ptr = &(M5);
new_ctx->T1sMULT_ptr = &(T1sMULT);
new_ctx->NumberOfVariables_ptr = &(NumberOfVariables);
new_ctx->TempMatrixOffset_ptr = &(TempMatrixOffset);
new_ctx->MatrixOffsetA_ptr = &(MatrixOffsetA);
new_ctx->MatrixOffsetB_ptr = &(MatrixOffsetB);
new_ctx->Heap_ptr = &(Heap);
new_ctx->StartHeap_ptr = &(StartHeap);
new_ctx->RowIncrementA_ptr = &(RowIncrementA);
new_ctx->RowIncrementB_ptr = &(RowIncrementB);
new_ctx->RowIncrementC_ptr = &(RowIncrementC);
new_ctx->C_ptr = &(C);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->MatrixSize_ptr = &(MatrixSize);
new_ctx->RowWidthC_ptr = &(RowWidthC);
new_ctx->RowWidthA_ptr = &(RowWidthA);
new_ctx->RowWidthB_ptr = &(RowWidthB);
new_ctx->Depth_ptr = &(Depth);
hclib_async(pragma691_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of C11 = M2 + A12 * B21 */
 { 
pragma695 *new_ctx = (pragma695 *)malloc(sizeof(pragma695));
new_ctx->QuadrantSize_ptr = &(QuadrantSize);
new_ctx->QuadrantSizeInBytes_ptr = &(QuadrantSizeInBytes);
new_ctx->Column_ptr = &(Column);
new_ctx->Row_ptr = &(Row);
new_ctx->A12_ptr = &(A12);
new_ctx->B12_ptr = &(B12);
new_ctx->C12_ptr = &(C12);
new_ctx->A21_ptr = &(A21);
new_ctx->B21_ptr = &(B21);
new_ctx->C21_ptr = &(C21);
new_ctx->A22_ptr = &(A22);
new_ctx->B22_ptr = &(B22);
new_ctx->C22_ptr = &(C22);
new_ctx->S1_ptr = &(S1);
new_ctx->S2_ptr = &(S2);
new_ctx->S3_ptr = &(S3);
new_ctx->S4_ptr = &(S4);
new_ctx->S5_ptr = &(S5);
new_ctx->S6_ptr = &(S6);
new_ctx->S7_ptr = &(S7);
new_ctx->S8_ptr = &(S8);
new_ctx->M2_ptr = &(M2);
new_ctx->M5_ptr = &(M5);
new_ctx->T1sMULT_ptr = &(T1sMULT);
new_ctx->NumberOfVariables_ptr = &(NumberOfVariables);
new_ctx->TempMatrixOffset_ptr = &(TempMatrixOffset);
new_ctx->MatrixOffsetA_ptr = &(MatrixOffsetA);
new_ctx->MatrixOffsetB_ptr = &(MatrixOffsetB);
new_ctx->Heap_ptr = &(Heap);
new_ctx->StartHeap_ptr = &(StartHeap);
new_ctx->RowIncrementA_ptr = &(RowIncrementA);
new_ctx->RowIncrementB_ptr = &(RowIncrementB);
new_ctx->RowIncrementC_ptr = &(RowIncrementC);
new_ctx->C_ptr = &(C);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->MatrixSize_ptr = &(MatrixSize);
new_ctx->RowWidthC_ptr = &(RowWidthC);
new_ctx->RowWidthA_ptr = &(RowWidthA);
new_ctx->RowWidthB_ptr = &(RowWidthB);
new_ctx->Depth_ptr = &(Depth);
hclib_async(pragma695_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
  
  /* Step 1 of C12 = S4 x B22 + T1 + M5 */
 { 
pragma699 *new_ctx = (pragma699 *)malloc(sizeof(pragma699));
new_ctx->QuadrantSize_ptr = &(QuadrantSize);
new_ctx->QuadrantSizeInBytes_ptr = &(QuadrantSizeInBytes);
new_ctx->Column_ptr = &(Column);
new_ctx->Row_ptr = &(Row);
new_ctx->A12_ptr = &(A12);
new_ctx->B12_ptr = &(B12);
new_ctx->C12_ptr = &(C12);
new_ctx->A21_ptr = &(A21);
new_ctx->B21_ptr = &(B21);
new_ctx->C21_ptr = &(C21);
new_ctx->A22_ptr = &(A22);
new_ctx->B22_ptr = &(B22);
new_ctx->C22_ptr = &(C22);
new_ctx->S1_ptr = &(S1);
new_ctx->S2_ptr = &(S2);
new_ctx->S3_ptr = &(S3);
new_ctx->S4_ptr = &(S4);
new_ctx->S5_ptr = &(S5);
new_ctx->S6_ptr = &(S6);
new_ctx->S7_ptr = &(S7);
new_ctx->S8_ptr = &(S8);
new_ctx->M2_ptr = &(M2);
new_ctx->M5_ptr = &(M5);
new_ctx->T1sMULT_ptr = &(T1sMULT);
new_ctx->NumberOfVariables_ptr = &(NumberOfVariables);
new_ctx->TempMatrixOffset_ptr = &(TempMatrixOffset);
new_ctx->MatrixOffsetA_ptr = &(MatrixOffsetA);
new_ctx->MatrixOffsetB_ptr = &(MatrixOffsetB);
new_ctx->Heap_ptr = &(Heap);
new_ctx->StartHeap_ptr = &(StartHeap);
new_ctx->RowIncrementA_ptr = &(RowIncrementA);
new_ctx->RowIncrementB_ptr = &(RowIncrementB);
new_ctx->RowIncrementC_ptr = &(RowIncrementC);
new_ctx->C_ptr = &(C);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->MatrixSize_ptr = &(MatrixSize);
new_ctx->RowWidthC_ptr = &(RowWidthC);
new_ctx->RowWidthA_ptr = &(RowWidthA);
new_ctx->RowWidthB_ptr = &(RowWidthB);
new_ctx->Depth_ptr = &(Depth);
hclib_async(pragma699_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of C21 = T2 - A22 * S8 */
 { 
pragma703 *new_ctx = (pragma703 *)malloc(sizeof(pragma703));
new_ctx->QuadrantSize_ptr = &(QuadrantSize);
new_ctx->QuadrantSizeInBytes_ptr = &(QuadrantSizeInBytes);
new_ctx->Column_ptr = &(Column);
new_ctx->Row_ptr = &(Row);
new_ctx->A12_ptr = &(A12);
new_ctx->B12_ptr = &(B12);
new_ctx->C12_ptr = &(C12);
new_ctx->A21_ptr = &(A21);
new_ctx->B21_ptr = &(B21);
new_ctx->C21_ptr = &(C21);
new_ctx->A22_ptr = &(A22);
new_ctx->B22_ptr = &(B22);
new_ctx->C22_ptr = &(C22);
new_ctx->S1_ptr = &(S1);
new_ctx->S2_ptr = &(S2);
new_ctx->S3_ptr = &(S3);
new_ctx->S4_ptr = &(S4);
new_ctx->S5_ptr = &(S5);
new_ctx->S6_ptr = &(S6);
new_ctx->S7_ptr = &(S7);
new_ctx->S8_ptr = &(S8);
new_ctx->M2_ptr = &(M2);
new_ctx->M5_ptr = &(M5);
new_ctx->T1sMULT_ptr = &(T1sMULT);
new_ctx->NumberOfVariables_ptr = &(NumberOfVariables);
new_ctx->TempMatrixOffset_ptr = &(TempMatrixOffset);
new_ctx->MatrixOffsetA_ptr = &(MatrixOffsetA);
new_ctx->MatrixOffsetB_ptr = &(MatrixOffsetB);
new_ctx->Heap_ptr = &(Heap);
new_ctx->StartHeap_ptr = &(StartHeap);
new_ctx->RowIncrementA_ptr = &(RowIncrementA);
new_ctx->RowIncrementB_ptr = &(RowIncrementB);
new_ctx->RowIncrementC_ptr = &(RowIncrementC);
new_ctx->C_ptr = &(C);
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->MatrixSize_ptr = &(MatrixSize);
new_ctx->RowWidthC_ptr = &(RowWidthC);
new_ctx->RowWidthA_ptr = &(RowWidthA);
new_ctx->RowWidthB_ptr = &(RowWidthB);
new_ctx->Depth_ptr = &(Depth);
hclib_async(pragma703_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /**********************************************
  ** Synchronization Point
  **********************************************/
 hclib_end_finish(); hclib_start_finish(); ;
  /***************************************************************************
  ** Step through all columns row by row (vertically)
  ** (jumps in memory by RowWidth => bad locality)
  ** (but we want the best locality on the innermost loop)
  ***************************************************************************/
  for (Row = 0; Row < QuadrantSize; Row++) {
    /*************************************************************************
    ** Step through each row horizontally (addressing elements in each column)
    ** (jumps linearly througn memory => good locality)
    *************************************************************************/
    for (Column = 0; Column < QuadrantSize; Column += 4) {
      REAL LocalM5_0 = *(M5);
      REAL LocalM5_1 = *(M5+1);
      REAL LocalM5_2 = *(M5+2);
      REAL LocalM5_3 = *(M5+3);
      REAL LocalM2_0 = *(M2);
      REAL LocalM2_1 = *(M2+1);
      REAL LocalM2_2 = *(M2+2);
      REAL LocalM2_3 = *(M2+3);
      REAL T1_0 = *(T1sMULT) + LocalM2_0;
      REAL T1_1 = *(T1sMULT+1) + LocalM2_1;
      REAL T1_2 = *(T1sMULT+2) + LocalM2_2;
      REAL T1_3 = *(T1sMULT+3) + LocalM2_3;
      REAL T2_0 = *(C22) + T1_0;
      REAL T2_1 = *(C22+1) + T1_1;
      REAL T2_2 = *(C22+2) + T1_2;
      REAL T2_3 = *(C22+3) + T1_3;
      (*(C))   += LocalM2_0;
      (*(C+1)) += LocalM2_1;
      (*(C+2)) += LocalM2_2;
      (*(C+3)) += LocalM2_3;
      (*(C12))   += LocalM5_0 + T1_0;
      (*(C12+1)) += LocalM5_1 + T1_1;
      (*(C12+2)) += LocalM5_2 + T1_2;
      (*(C12+3)) += LocalM5_3 + T1_3;
      (*(C22))   = LocalM5_0 + T2_0;
      (*(C22+1)) = LocalM5_1 + T2_1;
      (*(C22+2)) = LocalM5_2 + T2_2;
      (*(C22+3)) = LocalM5_3 + T2_3;
      (*(C21  )) = (- *(C21  )) + T2_0;
      (*(C21+1)) = (- *(C21+1)) + T2_1;
      (*(C21+2)) = (- *(C21+2)) + T2_2;
      (*(C21+3)) = (- *(C21+3)) + T2_3;
      M5 += 4;
      M2 += 4;
      T1sMULT += 4;
      C += 4;
      C12 += 4;
      C21 += 4;
      C22 += 4;
    }
    C = (REAL*) ( ((PTR) C ) + RowIncrementC);
    C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
    C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
    C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
  }
  free(StartHeap);
} 
static void pragma679_hclib_async(void *____arg) {
    pragma679 *ctx = (pragma679 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->M2_ptr)), (*(ctx->A_ptr)), (*(ctx->B_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->RowWidthA_ptr)), (*(ctx->RowWidthB_ptr)), (*(ctx->Depth_ptr))+1) ;     ; hclib_end_finish();

}


static void pragma683_hclib_async(void *____arg) {
    pragma683 *ctx = (pragma683 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->M5_ptr)), (*(ctx->S1_ptr)), (*(ctx->S5_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->Depth_ptr))+1) ;     ; hclib_end_finish();

}


static void pragma687_hclib_async(void *____arg) {
    pragma687 *ctx = (pragma687 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->T1sMULT_ptr)), (*(ctx->S2_ptr)), (*(ctx->S6_ptr)),  (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->Depth_ptr))+1) ;     ; hclib_end_finish();

}


static void pragma691_hclib_async(void *____arg) {
    pragma691 *ctx = (pragma691 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->C22_ptr)), (*(ctx->S3_ptr)), (*(ctx->S7_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->RowWidthC_ptr)) /*FIXME*/, (*(ctx->QuadrantSize_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->Depth_ptr))+1) ;     ; hclib_end_finish();

}


static void pragma695_hclib_async(void *____arg) {
    pragma695 *ctx = (pragma695 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->C_ptr)), (*(ctx->A12_ptr)), (*(ctx->B21_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->RowWidthC_ptr)), (*(ctx->RowWidthA_ptr)), (*(ctx->RowWidthB_ptr)), (*(ctx->Depth_ptr))+1) ;     ; hclib_end_finish();

}


static void pragma699_hclib_async(void *____arg) {
    pragma699 *ctx = (pragma699 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->C12_ptr)), (*(ctx->S4_ptr)), (*(ctx->B22_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->RowWidthC_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->RowWidthB_ptr)), (*(ctx->Depth_ptr))+1) ;     ; hclib_end_finish();

}


static void pragma703_hclib_async(void *____arg) {
    pragma703 *ctx = (pragma703 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->C21_ptr)), (*(ctx->A22_ptr)), (*(ctx->S8_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->RowWidthC_ptr)), (*(ctx->RowWidthA_ptr)), (*(ctx->QuadrantSize_ptr)), (*(ctx->Depth_ptr))+1) ;     ; hclib_end_finish();

}


/*
 * Set an n by n matrix A to random values.  The distance between
 * rows is an
 */
void init_matrix(int n, REAL *A, int an)
{
     int i, j;

     for (i = 0; i < n; ++i)
	  for (j = 0; j < n; ++j) 
	       ELEM(A, an, i, j) = ((double) rand()) / (double) RAND_MAX; 
}

/*
 * Compare two matrices.  Print an error message if they differ by
 * more than EPSILON.
 */
int compare_matrix(int n, REAL *A, int an, REAL *B, int bn)
{
     int i, j;
     REAL c;

     for (i = 0; i < n; ++i)
	  for (j = 0; j < n; ++j) {
	       /* compute the relative error c */
	       c = ELEM(A, an, i, j) - ELEM(B, bn, i, j);
	       if (c < 0.0) 
		    c = -c;

	       c = c / ELEM(A, an, i, j);
	       if (c > EPSILON) {
		    bots_message("Strassen: Wrong answer!\n");
		    return BOTS_RESULT_UNSUCCESSFUL;
	       }
	  }

     return BOTS_RESULT_SUCCESSFUL;
}
	       
/*
 * Allocate a matrix of side n (therefore n^2 elements)
 */
REAL *alloc_matrix(int n) 
{
     return (REAL *)malloc(n * n * sizeof(REAL));
}

typedef struct _pragma824 {
    REAL (*(*A_ptr));
    REAL (*(*B_ptr));
    REAL (*(*C_ptr));
    int (*n_ptr);
 } pragma824;

static void pragma824_hclib_async(void *____arg);
typedef struct _main_entrypoint_ctx {
    REAL (*A);
    REAL (*B);
    REAL (*C);
    int n;
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    REAL (*A); A = ctx->A;
    REAL (*B); B = ctx->B;
    REAL (*C); C = ctx->C;
    int n; n = ctx->n;
{
hclib_start_finish(); {
 { 
pragma824 *new_ctx = (pragma824 *)malloc(sizeof(pragma824));
new_ctx->A_ptr = &(A);
new_ctx->B_ptr = &(B);
new_ctx->C_ptr = &(C);
new_ctx->n_ptr = &(n);
hclib_async(pragma824_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
            } ; hclib_end_finish(); 
	bots_message(" completed!\n");
    } ; }

void strassen_main_par(REAL *A, REAL *B, REAL *C, int n)
{
	bots_message("Computing parallel Strassen algorithm (n=%d) ", n);
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->A = A;
new_ctx->B = B;
new_ctx->C = C;
new_ctx->n = n;
hclib_launch(main_entrypoint, new_ctx);
free(new_ctx);

}  
static void pragma824_hclib_async(void *____arg) {
    pragma824 *ctx = (pragma824 *)____arg;
    hclib_start_finish();
OptimizedStrassenMultiply_par((*(ctx->C_ptr)), (*(ctx->A_ptr)), (*(ctx->B_ptr)), (*(ctx->n_ptr)), (*(ctx->n_ptr)), (*(ctx->n_ptr)), (*(ctx->n_ptr)), 1) ;     ; hclib_end_finish();

}



void strassen_main_seq(REAL *A, REAL *B, REAL *C, int n)
{
	bots_message("Computing sequential Strassen algorithm (n=%d) ", n);
	OptimizedStrassenMultiply_seq(C, A, B, n, n, n, n, 1);
	bots_message(" completed!\n");
}

