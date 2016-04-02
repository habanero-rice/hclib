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
  #define A00 A
  #define B00 B
  #define C00 C
  REAL  *A01, *A10, *A11, *B01, *B10, *B11, *C01, *C10, *C11;
  unsigned QuadrantSize = MatrixSize >> 1;

  /* partition the matrix */
  A01 = A00 + QuadrantSize;
  A10 = A00 + RowWidthA * QuadrantSize;
  A11 = A10 + QuadrantSize;

  B01 = B00 + QuadrantSize;
  B10 = B00 + RowWidthB * QuadrantSize;
  B11 = B10 + QuadrantSize;

  C01 = C00 + QuadrantSize;
  C10 = C00 + RowWidthC * QuadrantSize;
  C11 = C10 + QuadrantSize;

  if (QuadrantSize > SizeAtWhichNaiveAlgorithmIsMoreEfficient) {

    MultiplyByDivideAndConquer(C00, A00, B00, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C01, A00, B01, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C11, A10, B01, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C10, A10, B00, QuadrantSize,
				     RowWidthC, RowWidthA, RowWidthB,
				     AdditiveMode);

    MultiplyByDivideAndConquer(C00, A01, B10, QuadrantSize,
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
      FastAdditiveNaiveMatrixMultiply(C00, A00, B00, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastAdditiveNaiveMatrixMultiply(C01, A00, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastAdditiveNaiveMatrixMultiply(C11, A10, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastAdditiveNaiveMatrixMultiply(C10, A10, B00, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
    } else {
      
      FastNaiveMatrixMultiply(C00, A00, B00, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastNaiveMatrixMultiply(C01, A00, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastNaiveMatrixMultiply(C11, A10, B01, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
      
      FastNaiveMatrixMultiply(C10, A10, B00, QuadrantSize,
			      RowWidthC, RowWidthA, RowWidthB);
    }

    FastAdditiveNaiveMatrixMultiply(C00, A01, B10, QuadrantSize,
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
  #define T2sMULT C22
  #define NumberOfVariables 11

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
  #define A11 A
  #define B11 B
  #define C11 C
  A12 = A11 + QuadrantSize;
  B12 = B11 + QuadrantSize;
  C12 = C11 + QuadrantSize;
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
      E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );

      /* S8 = (S6 = B22 - ( S5 = B12 - B11 ) ) - B21 */
      E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);

      /* S3 = A11 - A21 */
      E(S3) = EA(A11) - EA(A21);
      
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
  OptimizedStrassenMultiply_seq(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);

  /* M5 = S1 * S5 */
  OptimizedStrassenMultiply_seq(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of T1 = S2 x S6 + M2 */
  OptimizedStrassenMultiply_seq(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of T2 = T1 + S3 x S7 */
  OptimizedStrassenMultiply_seq(C22, S3, S7, QuadrantSize, RowWidthC /*FIXME*/, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of C11 = M2 + A12 * B21 */
  OptimizedStrassenMultiply_seq(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
  
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
      (*(C11))   += LocalM2_0;
      (*(C11+1)) += LocalM2_1;
      (*(C11+2)) += LocalM2_2;
      (*(C11+3)) += LocalM2_3;
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
      C11 += 4;
      C12 += 4;
      C21 += 4;
      C22 += 4;
    }
    C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
    C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
    C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
    C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
  }
  free(StartHeap);
}
#if defined(IF_CUTOFF)
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
  #define T2sMULT C22
  #define NumberOfVariables 11

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
  #define A11 A
  #define B11 B
  #define C11 C
  A12 = A11 + QuadrantSize;
  B12 = B11 + QuadrantSize;
  C12 = C11 + QuadrantSize;
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
      E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );

      /* S8 = (S6 = B22 - ( S5 = B12 - B11 ) ) - B21 */
      E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);

      /* S3 = A11 - A21 */
      E(S3) = EA(A11) - EA(A21);
      
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
hclib_pragma_marker("omp", "task untied if (Depth < bots_cutoff_value)");
  OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);

  /* M5 = S1 * S5 */
hclib_pragma_marker("omp", "task untied if (Depth < bots_cutoff_value)");
  OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of T1 = S2 x S6 + M2 */
hclib_pragma_marker("omp", "task untied if (Depth < bots_cutoff_value)");
  OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of T2 = T1 + S3 x S7 */
hclib_pragma_marker("omp", "task untied if (Depth < bots_cutoff_value)");
  OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC /*FIXME*/, QuadrantSize, QuadrantSize, Depth+1);

  /* Step 1 of C11 = M2 + A12 * B21 */
hclib_pragma_marker("omp", "task untied if (Depth < bots_cutoff_value)");
  OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
  
  /* Step 1 of C12 = S4 x B22 + T1 + M5 */
hclib_pragma_marker("omp", "task untied if (Depth < bots_cutoff_value)");
  OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);

  /* Step 1 of C21 = T2 - A22 * S8 */
hclib_pragma_marker("omp", "task untied if (Depth < bots_cutoff_value)");
  OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);

  /**********************************************
  ** Synchronization Point
  **********************************************/
hclib_pragma_marker("omp", "taskwait");
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
      (*(C11))   += LocalM2_0;
      (*(C11+1)) += LocalM2_1;
      (*(C11+2)) += LocalM2_2;
      (*(C11+3)) += LocalM2_3;
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
      C11 += 4;
      C12 += 4;
      C21 += 4;
      C22 += 4;
    }
    C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
    C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
    C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
    C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
  }
  free(StartHeap);
}
#elif defined(MANUAL_CUTOFF)
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
  #define T2sMULT C22
  #define NumberOfVariables 11

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
  #define A11 A
  #define B11 B
  #define C11 C
  A12 = A11 + QuadrantSize;
  B12 = B11 + QuadrantSize;
  C12 = C11 + QuadrantSize;
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
      E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );

      /* S8 = (S6 = B22 - ( S5 = B12 - B11 ) ) - B21 */
      E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);

      /* S3 = A11 - A21 */
      E(S3) = EA(A11) - EA(A21);
      
      /* S7 = B22 - B12 */
      E(S7) = EB(B22) - EB(B12);

      TempMatrixOffset += sizeof(REAL);
      MatrixOffsetA += sizeof(REAL);
      MatrixOffsetB += sizeof(REAL);
    } /* end row loop*/

    MatrixOffsetA += RowIncrementA;
    MatrixOffsetB += RowIncrementB;
  } /* end column loop */

  if (Depth < bots_cutoff_value)
  {
    /* M2 = A11 x B11 */
hclib_pragma_marker("omp", "task untied");
    OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);

    /* M5 = S1 * S5 */
hclib_pragma_marker("omp", "task untied");
    OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

    /* Step 1 of T1 = S2 x S6 + M2 */
hclib_pragma_marker("omp", "task untied");
    OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);

    /* Step 1 of T2 = T1 + S3 x S7 */
hclib_pragma_marker("omp", "task untied");
    OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC /*FIXME*/, QuadrantSize, QuadrantSize, Depth+1);

    /* Step 1 of C11 = M2 + A12 * B21 */
hclib_pragma_marker("omp", "task untied");
    OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
  
    /* Step 1 of C12 = S4 x B22 + T1 + M5 */
hclib_pragma_marker("omp", "task untied");
    OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);

    /* Step 1 of C21 = T2 - A22 * S8 */
hclib_pragma_marker("omp", "task untied");
    OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);

    /**********************************************
    ** Synchronization Point
    **********************************************/
hclib_pragma_marker("omp", "taskwait");
  }
  else
  {
    /* M2 = A11 x B11 */
    OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1);
    /* M5 = S1 * S5 */
    OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
    /* Step 1 of T1 = S2 x S6 + M2 */
    OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1);
    /* Step 1 of T2 = T1 + S3 x S7 */
    OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC /*FIXME*/, QuadrantSize, QuadrantSize, Depth+1);
    /* Step 1 of C11 = M2 + A12 * B21 */
    OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1);
    /* Step 1 of C12 = S4 x B22 + T1 + M5 */
    OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1);
    /* Step 1 of C21 = T2 - A22 * S8 */
    OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1);
  }

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
      (*(C11))   += LocalM2_0;
      (*(C11+1)) += LocalM2_1;
      (*(C11+2)) += LocalM2_2;
      (*(C11+3)) += LocalM2_3;
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
      C11 += 4;
      C12 += 4;
      C21 += 4;
      C22 += 4;
    }
    C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
    C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
    C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
    C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
  }
  free(StartHeap);
}
#else
typedef struct _pragma1131 {
    REAL *C;
    REAL *A;
    REAL *B;
    unsigned int MatrixSize;
    unsigned int RowWidthC;
    unsigned int RowWidthA;
    unsigned int RowWidthB;
    int Depth;
    unsigned int QuadrantSize;
    unsigned int QuadrantSizeInBytes;
    unsigned int Column;
    unsigned int Row;
    REAL *A12;
    REAL *B12;
    REAL *C12;
    REAL *A21;
    REAL *B21;
    REAL *C21;
    REAL *A22;
    REAL *B22;
    REAL *C22;
    REAL *S1;
    REAL *S2;
    REAL *S3;
    REAL *S4;
    REAL *S5;
    REAL *S6;
    REAL *S7;
    REAL *S8;
    REAL *M2;
    REAL *M5;
    REAL *T1sMULT;
    PTR TempMatrixOffset;
    PTR MatrixOffsetA;
    PTR MatrixOffsetB;
    char *Heap;
    void *StartHeap;
    PTR RowIncrementA;
    PTR RowIncrementB;
    PTR RowIncrementC;
 } pragma1131;

typedef struct _pragma1135 {
    REAL *C;
    REAL *A;
    REAL *B;
    unsigned int MatrixSize;
    unsigned int RowWidthC;
    unsigned int RowWidthA;
    unsigned int RowWidthB;
    int Depth;
    unsigned int QuadrantSize;
    unsigned int QuadrantSizeInBytes;
    unsigned int Column;
    unsigned int Row;
    REAL *A12;
    REAL *B12;
    REAL *C12;
    REAL *A21;
    REAL *B21;
    REAL *C21;
    REAL *A22;
    REAL *B22;
    REAL *C22;
    REAL *S1;
    REAL *S2;
    REAL *S3;
    REAL *S4;
    REAL *S5;
    REAL *S6;
    REAL *S7;
    REAL *S8;
    REAL *M2;
    REAL *M5;
    REAL *T1sMULT;
    PTR TempMatrixOffset;
    PTR MatrixOffsetA;
    PTR MatrixOffsetB;
    char *Heap;
    void *StartHeap;
    PTR RowIncrementA;
    PTR RowIncrementB;
    PTR RowIncrementC;
 } pragma1135;

typedef struct _pragma1139 {
    REAL *C;
    REAL *A;
    REAL *B;
    unsigned int MatrixSize;
    unsigned int RowWidthC;
    unsigned int RowWidthA;
    unsigned int RowWidthB;
    int Depth;
    unsigned int QuadrantSize;
    unsigned int QuadrantSizeInBytes;
    unsigned int Column;
    unsigned int Row;
    REAL *A12;
    REAL *B12;
    REAL *C12;
    REAL *A21;
    REAL *B21;
    REAL *C21;
    REAL *A22;
    REAL *B22;
    REAL *C22;
    REAL *S1;
    REAL *S2;
    REAL *S3;
    REAL *S4;
    REAL *S5;
    REAL *S6;
    REAL *S7;
    REAL *S8;
    REAL *M2;
    REAL *M5;
    REAL *T1sMULT;
    PTR TempMatrixOffset;
    PTR MatrixOffsetA;
    PTR MatrixOffsetB;
    char *Heap;
    void *StartHeap;
    PTR RowIncrementA;
    PTR RowIncrementB;
    PTR RowIncrementC;
 } pragma1139;

typedef struct _pragma1143 {
    REAL *C;
    REAL *A;
    REAL *B;
    unsigned int MatrixSize;
    unsigned int RowWidthC;
    unsigned int RowWidthA;
    unsigned int RowWidthB;
    int Depth;
    unsigned int QuadrantSize;
    unsigned int QuadrantSizeInBytes;
    unsigned int Column;
    unsigned int Row;
    REAL *A12;
    REAL *B12;
    REAL *C12;
    REAL *A21;
    REAL *B21;
    REAL *C21;
    REAL *A22;
    REAL *B22;
    REAL *C22;
    REAL *S1;
    REAL *S2;
    REAL *S3;
    REAL *S4;
    REAL *S5;
    REAL *S6;
    REAL *S7;
    REAL *S8;
    REAL *M2;
    REAL *M5;
    REAL *T1sMULT;
    PTR TempMatrixOffset;
    PTR MatrixOffsetA;
    PTR MatrixOffsetB;
    char *Heap;
    void *StartHeap;
    PTR RowIncrementA;
    PTR RowIncrementB;
    PTR RowIncrementC;
 } pragma1143;

typedef struct _pragma1147 {
    REAL *C;
    REAL *A;
    REAL *B;
    unsigned int MatrixSize;
    unsigned int RowWidthC;
    unsigned int RowWidthA;
    unsigned int RowWidthB;
    int Depth;
    unsigned int QuadrantSize;
    unsigned int QuadrantSizeInBytes;
    unsigned int Column;
    unsigned int Row;
    REAL *A12;
    REAL *B12;
    REAL *C12;
    REAL *A21;
    REAL *B21;
    REAL *C21;
    REAL *A22;
    REAL *B22;
    REAL *C22;
    REAL *S1;
    REAL *S2;
    REAL *S3;
    REAL *S4;
    REAL *S5;
    REAL *S6;
    REAL *S7;
    REAL *S8;
    REAL *M2;
    REAL *M5;
    REAL *T1sMULT;
    PTR TempMatrixOffset;
    PTR MatrixOffsetA;
    PTR MatrixOffsetB;
    char *Heap;
    void *StartHeap;
    PTR RowIncrementA;
    PTR RowIncrementB;
    PTR RowIncrementC;
 } pragma1147;

typedef struct _pragma1151 {
    REAL *C;
    REAL *A;
    REAL *B;
    unsigned int MatrixSize;
    unsigned int RowWidthC;
    unsigned int RowWidthA;
    unsigned int RowWidthB;
    int Depth;
    unsigned int QuadrantSize;
    unsigned int QuadrantSizeInBytes;
    unsigned int Column;
    unsigned int Row;
    REAL *A12;
    REAL *B12;
    REAL *C12;
    REAL *A21;
    REAL *B21;
    REAL *C21;
    REAL *A22;
    REAL *B22;
    REAL *C22;
    REAL *S1;
    REAL *S2;
    REAL *S3;
    REAL *S4;
    REAL *S5;
    REAL *S6;
    REAL *S7;
    REAL *S8;
    REAL *M2;
    REAL *M5;
    REAL *T1sMULT;
    PTR TempMatrixOffset;
    PTR MatrixOffsetA;
    PTR MatrixOffsetB;
    char *Heap;
    void *StartHeap;
    PTR RowIncrementA;
    PTR RowIncrementB;
    PTR RowIncrementC;
 } pragma1151;

typedef struct _pragma1155 {
    REAL *C;
    REAL *A;
    REAL *B;
    unsigned int MatrixSize;
    unsigned int RowWidthC;
    unsigned int RowWidthA;
    unsigned int RowWidthB;
    int Depth;
    unsigned int QuadrantSize;
    unsigned int QuadrantSizeInBytes;
    unsigned int Column;
    unsigned int Row;
    REAL *A12;
    REAL *B12;
    REAL *C12;
    REAL *A21;
    REAL *B21;
    REAL *C21;
    REAL *A22;
    REAL *B22;
    REAL *C22;
    REAL *S1;
    REAL *S2;
    REAL *S3;
    REAL *S4;
    REAL *S5;
    REAL *S6;
    REAL *S7;
    REAL *S8;
    REAL *M2;
    REAL *M5;
    REAL *T1sMULT;
    PTR TempMatrixOffset;
    PTR MatrixOffsetA;
    PTR MatrixOffsetB;
    char *Heap;
    void *StartHeap;
    PTR RowIncrementA;
    PTR RowIncrementB;
    PTR RowIncrementC;
 } pragma1155;

static void pragma1131_hclib_async(void *____arg);
static void pragma1135_hclib_async(void *____arg);
static void pragma1139_hclib_async(void *____arg);
static void pragma1143_hclib_async(void *____arg);
static void pragma1147_hclib_async(void *____arg);
static void pragma1151_hclib_async(void *____arg);
static void pragma1155_hclib_async(void *____arg);
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
  #define T2sMULT C22
  #define NumberOfVariables 11

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
  #define A11 A
  #define B11 B
  #define C11 C
  A12 = A11 + QuadrantSize;
  B12 = B11 + QuadrantSize;
  C12 = C11 + QuadrantSize;
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
      E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );

      /* S8 = (S6 = B22 - ( S5 = B12 - B11 ) ) - B21 */
      E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);

      /* S3 = A11 - A21 */
      E(S3) = EA(A11) - EA(A21);
      
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
pragma1131 *ctx = (pragma1131 *)malloc(sizeof(pragma1131));
ctx->C = C;
ctx->A = A;
ctx->B = B;
ctx->MatrixSize = MatrixSize;
ctx->RowWidthC = RowWidthC;
ctx->RowWidthA = RowWidthA;
ctx->RowWidthB = RowWidthB;
ctx->Depth = Depth;
ctx->QuadrantSize = QuadrantSize;
ctx->QuadrantSizeInBytes = QuadrantSizeInBytes;
ctx->Column = Column;
ctx->Row = Row;
ctx->A12 = A12;
ctx->B12 = B12;
ctx->C12 = C12;
ctx->A21 = A21;
ctx->B21 = B21;
ctx->C21 = C21;
ctx->A22 = A22;
ctx->B22 = B22;
ctx->C22 = C22;
ctx->S1 = S1;
ctx->S2 = S2;
ctx->S3 = S3;
ctx->S4 = S4;
ctx->S5 = S5;
ctx->S6 = S6;
ctx->S7 = S7;
ctx->S8 = S8;
ctx->M2 = M2;
ctx->M5 = M5;
ctx->T1sMULT = T1sMULT;
ctx->TempMatrixOffset = TempMatrixOffset;
ctx->MatrixOffsetA = MatrixOffsetA;
ctx->MatrixOffsetB = MatrixOffsetB;
ctx->Heap = Heap;
ctx->StartHeap = StartHeap;
ctx->RowIncrementA = RowIncrementA;
ctx->RowIncrementB = RowIncrementB;
ctx->RowIncrementC = RowIncrementC;
hclib_async(pragma1131_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* M5 = S1 * S5 */
 { 
pragma1135 *ctx = (pragma1135 *)malloc(sizeof(pragma1135));
ctx->C = C;
ctx->A = A;
ctx->B = B;
ctx->MatrixSize = MatrixSize;
ctx->RowWidthC = RowWidthC;
ctx->RowWidthA = RowWidthA;
ctx->RowWidthB = RowWidthB;
ctx->Depth = Depth;
ctx->QuadrantSize = QuadrantSize;
ctx->QuadrantSizeInBytes = QuadrantSizeInBytes;
ctx->Column = Column;
ctx->Row = Row;
ctx->A12 = A12;
ctx->B12 = B12;
ctx->C12 = C12;
ctx->A21 = A21;
ctx->B21 = B21;
ctx->C21 = C21;
ctx->A22 = A22;
ctx->B22 = B22;
ctx->C22 = C22;
ctx->S1 = S1;
ctx->S2 = S2;
ctx->S3 = S3;
ctx->S4 = S4;
ctx->S5 = S5;
ctx->S6 = S6;
ctx->S7 = S7;
ctx->S8 = S8;
ctx->M2 = M2;
ctx->M5 = M5;
ctx->T1sMULT = T1sMULT;
ctx->TempMatrixOffset = TempMatrixOffset;
ctx->MatrixOffsetA = MatrixOffsetA;
ctx->MatrixOffsetB = MatrixOffsetB;
ctx->Heap = Heap;
ctx->StartHeap = StartHeap;
ctx->RowIncrementA = RowIncrementA;
ctx->RowIncrementB = RowIncrementB;
ctx->RowIncrementC = RowIncrementC;
hclib_async(pragma1135_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of T1 = S2 x S6 + M2 */
 { 
pragma1139 *ctx = (pragma1139 *)malloc(sizeof(pragma1139));
ctx->C = C;
ctx->A = A;
ctx->B = B;
ctx->MatrixSize = MatrixSize;
ctx->RowWidthC = RowWidthC;
ctx->RowWidthA = RowWidthA;
ctx->RowWidthB = RowWidthB;
ctx->Depth = Depth;
ctx->QuadrantSize = QuadrantSize;
ctx->QuadrantSizeInBytes = QuadrantSizeInBytes;
ctx->Column = Column;
ctx->Row = Row;
ctx->A12 = A12;
ctx->B12 = B12;
ctx->C12 = C12;
ctx->A21 = A21;
ctx->B21 = B21;
ctx->C21 = C21;
ctx->A22 = A22;
ctx->B22 = B22;
ctx->C22 = C22;
ctx->S1 = S1;
ctx->S2 = S2;
ctx->S3 = S3;
ctx->S4 = S4;
ctx->S5 = S5;
ctx->S6 = S6;
ctx->S7 = S7;
ctx->S8 = S8;
ctx->M2 = M2;
ctx->M5 = M5;
ctx->T1sMULT = T1sMULT;
ctx->TempMatrixOffset = TempMatrixOffset;
ctx->MatrixOffsetA = MatrixOffsetA;
ctx->MatrixOffsetB = MatrixOffsetB;
ctx->Heap = Heap;
ctx->StartHeap = StartHeap;
ctx->RowIncrementA = RowIncrementA;
ctx->RowIncrementB = RowIncrementB;
ctx->RowIncrementC = RowIncrementC;
hclib_async(pragma1139_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of T2 = T1 + S3 x S7 */
 { 
pragma1143 *ctx = (pragma1143 *)malloc(sizeof(pragma1143));
ctx->C = C;
ctx->A = A;
ctx->B = B;
ctx->MatrixSize = MatrixSize;
ctx->RowWidthC = RowWidthC;
ctx->RowWidthA = RowWidthA;
ctx->RowWidthB = RowWidthB;
ctx->Depth = Depth;
ctx->QuadrantSize = QuadrantSize;
ctx->QuadrantSizeInBytes = QuadrantSizeInBytes;
ctx->Column = Column;
ctx->Row = Row;
ctx->A12 = A12;
ctx->B12 = B12;
ctx->C12 = C12;
ctx->A21 = A21;
ctx->B21 = B21;
ctx->C21 = C21;
ctx->A22 = A22;
ctx->B22 = B22;
ctx->C22 = C22;
ctx->S1 = S1;
ctx->S2 = S2;
ctx->S3 = S3;
ctx->S4 = S4;
ctx->S5 = S5;
ctx->S6 = S6;
ctx->S7 = S7;
ctx->S8 = S8;
ctx->M2 = M2;
ctx->M5 = M5;
ctx->T1sMULT = T1sMULT;
ctx->TempMatrixOffset = TempMatrixOffset;
ctx->MatrixOffsetA = MatrixOffsetA;
ctx->MatrixOffsetB = MatrixOffsetB;
ctx->Heap = Heap;
ctx->StartHeap = StartHeap;
ctx->RowIncrementA = RowIncrementA;
ctx->RowIncrementB = RowIncrementB;
ctx->RowIncrementC = RowIncrementC;
hclib_async(pragma1143_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of C11 = M2 + A12 * B21 */
 { 
pragma1147 *ctx = (pragma1147 *)malloc(sizeof(pragma1147));
ctx->C = C;
ctx->A = A;
ctx->B = B;
ctx->MatrixSize = MatrixSize;
ctx->RowWidthC = RowWidthC;
ctx->RowWidthA = RowWidthA;
ctx->RowWidthB = RowWidthB;
ctx->Depth = Depth;
ctx->QuadrantSize = QuadrantSize;
ctx->QuadrantSizeInBytes = QuadrantSizeInBytes;
ctx->Column = Column;
ctx->Row = Row;
ctx->A12 = A12;
ctx->B12 = B12;
ctx->C12 = C12;
ctx->A21 = A21;
ctx->B21 = B21;
ctx->C21 = C21;
ctx->A22 = A22;
ctx->B22 = B22;
ctx->C22 = C22;
ctx->S1 = S1;
ctx->S2 = S2;
ctx->S3 = S3;
ctx->S4 = S4;
ctx->S5 = S5;
ctx->S6 = S6;
ctx->S7 = S7;
ctx->S8 = S8;
ctx->M2 = M2;
ctx->M5 = M5;
ctx->T1sMULT = T1sMULT;
ctx->TempMatrixOffset = TempMatrixOffset;
ctx->MatrixOffsetA = MatrixOffsetA;
ctx->MatrixOffsetB = MatrixOffsetB;
ctx->Heap = Heap;
ctx->StartHeap = StartHeap;
ctx->RowIncrementA = RowIncrementA;
ctx->RowIncrementB = RowIncrementB;
ctx->RowIncrementC = RowIncrementC;
hclib_async(pragma1147_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;
  
  /* Step 1 of C12 = S4 x B22 + T1 + M5 */
 { 
pragma1151 *ctx = (pragma1151 *)malloc(sizeof(pragma1151));
ctx->C = C;
ctx->A = A;
ctx->B = B;
ctx->MatrixSize = MatrixSize;
ctx->RowWidthC = RowWidthC;
ctx->RowWidthA = RowWidthA;
ctx->RowWidthB = RowWidthB;
ctx->Depth = Depth;
ctx->QuadrantSize = QuadrantSize;
ctx->QuadrantSizeInBytes = QuadrantSizeInBytes;
ctx->Column = Column;
ctx->Row = Row;
ctx->A12 = A12;
ctx->B12 = B12;
ctx->C12 = C12;
ctx->A21 = A21;
ctx->B21 = B21;
ctx->C21 = C21;
ctx->A22 = A22;
ctx->B22 = B22;
ctx->C22 = C22;
ctx->S1 = S1;
ctx->S2 = S2;
ctx->S3 = S3;
ctx->S4 = S4;
ctx->S5 = S5;
ctx->S6 = S6;
ctx->S7 = S7;
ctx->S8 = S8;
ctx->M2 = M2;
ctx->M5 = M5;
ctx->T1sMULT = T1sMULT;
ctx->TempMatrixOffset = TempMatrixOffset;
ctx->MatrixOffsetA = MatrixOffsetA;
ctx->MatrixOffsetB = MatrixOffsetB;
ctx->Heap = Heap;
ctx->StartHeap = StartHeap;
ctx->RowIncrementA = RowIncrementA;
ctx->RowIncrementB = RowIncrementB;
ctx->RowIncrementC = RowIncrementC;
hclib_async(pragma1151_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;

  /* Step 1 of C21 = T2 - A22 * S8 */
 { 
pragma1155 *ctx = (pragma1155 *)malloc(sizeof(pragma1155));
ctx->C = C;
ctx->A = A;
ctx->B = B;
ctx->MatrixSize = MatrixSize;
ctx->RowWidthC = RowWidthC;
ctx->RowWidthA = RowWidthA;
ctx->RowWidthB = RowWidthB;
ctx->Depth = Depth;
ctx->QuadrantSize = QuadrantSize;
ctx->QuadrantSizeInBytes = QuadrantSizeInBytes;
ctx->Column = Column;
ctx->Row = Row;
ctx->A12 = A12;
ctx->B12 = B12;
ctx->C12 = C12;
ctx->A21 = A21;
ctx->B21 = B21;
ctx->C21 = C21;
ctx->A22 = A22;
ctx->B22 = B22;
ctx->C22 = C22;
ctx->S1 = S1;
ctx->S2 = S2;
ctx->S3 = S3;
ctx->S4 = S4;
ctx->S5 = S5;
ctx->S6 = S6;
ctx->S7 = S7;
ctx->S8 = S8;
ctx->M2 = M2;
ctx->M5 = M5;
ctx->T1sMULT = T1sMULT;
ctx->TempMatrixOffset = TempMatrixOffset;
ctx->MatrixOffsetA = MatrixOffsetA;
ctx->MatrixOffsetB = MatrixOffsetB;
ctx->Heap = Heap;
ctx->StartHeap = StartHeap;
ctx->RowIncrementA = RowIncrementA;
ctx->RowIncrementB = RowIncrementB;
ctx->RowIncrementC = RowIncrementC;
hclib_async(pragma1155_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
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
      (*(C11))   += LocalM2_0;
      (*(C11+1)) += LocalM2_1;
      (*(C11+2)) += LocalM2_2;
      (*(C11+3)) += LocalM2_3;
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
      C11 += 4;
      C12 += 4;
      C21 += 4;
      C22 += 4;
    }
    C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
    C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
    C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
    C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
  }
  free(StartHeap);
} static void pragma1131_hclib_async(void *____arg) {
    pragma1131 *ctx = (pragma1131 *)____arg;
    REAL *C; C = ctx->C;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    unsigned int MatrixSize; MatrixSize = ctx->MatrixSize;
    unsigned int RowWidthC; RowWidthC = ctx->RowWidthC;
    unsigned int RowWidthA; RowWidthA = ctx->RowWidthA;
    unsigned int RowWidthB; RowWidthB = ctx->RowWidthB;
    int Depth; Depth = ctx->Depth;
    unsigned int QuadrantSize; QuadrantSize = ctx->QuadrantSize;
    unsigned int QuadrantSizeInBytes; QuadrantSizeInBytes = ctx->QuadrantSizeInBytes;
    unsigned int Column; Column = ctx->Column;
    unsigned int Row; Row = ctx->Row;
    REAL *A12; A12 = ctx->A12;
    REAL *B12; B12 = ctx->B12;
    REAL *C12; C12 = ctx->C12;
    REAL *A21; A21 = ctx->A21;
    REAL *B21; B21 = ctx->B21;
    REAL *C21; C21 = ctx->C21;
    REAL *A22; A22 = ctx->A22;
    REAL *B22; B22 = ctx->B22;
    REAL *C22; C22 = ctx->C22;
    REAL *S1; S1 = ctx->S1;
    REAL *S2; S2 = ctx->S2;
    REAL *S3; S3 = ctx->S3;
    REAL *S4; S4 = ctx->S4;
    REAL *S5; S5 = ctx->S5;
    REAL *S6; S6 = ctx->S6;
    REAL *S7; S7 = ctx->S7;
    REAL *S8; S8 = ctx->S8;
    REAL *M2; M2 = ctx->M2;
    REAL *M5; M5 = ctx->M5;
    REAL *T1sMULT; T1sMULT = ctx->T1sMULT;
    PTR TempMatrixOffset; TempMatrixOffset = ctx->TempMatrixOffset;
    PTR MatrixOffsetA; MatrixOffsetA = ctx->MatrixOffsetA;
    PTR MatrixOffsetB; MatrixOffsetB = ctx->MatrixOffsetB;
    char *Heap; Heap = ctx->Heap;
    void *StartHeap; StartHeap = ctx->StartHeap;
    PTR RowIncrementA; RowIncrementA = ctx->RowIncrementA;
    PTR RowIncrementB; RowIncrementB = ctx->RowIncrementB;
    PTR RowIncrementC; RowIncrementC = ctx->RowIncrementC;
    hclib_start_finish();
OptimizedStrassenMultiply_par(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1) ;     ; hclib_end_finish();
}

static void pragma1135_hclib_async(void *____arg) {
    pragma1135 *ctx = (pragma1135 *)____arg;
    REAL *C; C = ctx->C;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    unsigned int MatrixSize; MatrixSize = ctx->MatrixSize;
    unsigned int RowWidthC; RowWidthC = ctx->RowWidthC;
    unsigned int RowWidthA; RowWidthA = ctx->RowWidthA;
    unsigned int RowWidthB; RowWidthB = ctx->RowWidthB;
    int Depth; Depth = ctx->Depth;
    unsigned int QuadrantSize; QuadrantSize = ctx->QuadrantSize;
    unsigned int QuadrantSizeInBytes; QuadrantSizeInBytes = ctx->QuadrantSizeInBytes;
    unsigned int Column; Column = ctx->Column;
    unsigned int Row; Row = ctx->Row;
    REAL *A12; A12 = ctx->A12;
    REAL *B12; B12 = ctx->B12;
    REAL *C12; C12 = ctx->C12;
    REAL *A21; A21 = ctx->A21;
    REAL *B21; B21 = ctx->B21;
    REAL *C21; C21 = ctx->C21;
    REAL *A22; A22 = ctx->A22;
    REAL *B22; B22 = ctx->B22;
    REAL *C22; C22 = ctx->C22;
    REAL *S1; S1 = ctx->S1;
    REAL *S2; S2 = ctx->S2;
    REAL *S3; S3 = ctx->S3;
    REAL *S4; S4 = ctx->S4;
    REAL *S5; S5 = ctx->S5;
    REAL *S6; S6 = ctx->S6;
    REAL *S7; S7 = ctx->S7;
    REAL *S8; S8 = ctx->S8;
    REAL *M2; M2 = ctx->M2;
    REAL *M5; M5 = ctx->M5;
    REAL *T1sMULT; T1sMULT = ctx->T1sMULT;
    PTR TempMatrixOffset; TempMatrixOffset = ctx->TempMatrixOffset;
    PTR MatrixOffsetA; MatrixOffsetA = ctx->MatrixOffsetA;
    PTR MatrixOffsetB; MatrixOffsetB = ctx->MatrixOffsetB;
    char *Heap; Heap = ctx->Heap;
    void *StartHeap; StartHeap = ctx->StartHeap;
    PTR RowIncrementA; RowIncrementA = ctx->RowIncrementA;
    PTR RowIncrementB; RowIncrementB = ctx->RowIncrementB;
    PTR RowIncrementC; RowIncrementC = ctx->RowIncrementC;
    hclib_start_finish();
OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1) ;     ; hclib_end_finish();
}

static void pragma1139_hclib_async(void *____arg) {
    pragma1139 *ctx = (pragma1139 *)____arg;
    REAL *C; C = ctx->C;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    unsigned int MatrixSize; MatrixSize = ctx->MatrixSize;
    unsigned int RowWidthC; RowWidthC = ctx->RowWidthC;
    unsigned int RowWidthA; RowWidthA = ctx->RowWidthA;
    unsigned int RowWidthB; RowWidthB = ctx->RowWidthB;
    int Depth; Depth = ctx->Depth;
    unsigned int QuadrantSize; QuadrantSize = ctx->QuadrantSize;
    unsigned int QuadrantSizeInBytes; QuadrantSizeInBytes = ctx->QuadrantSizeInBytes;
    unsigned int Column; Column = ctx->Column;
    unsigned int Row; Row = ctx->Row;
    REAL *A12; A12 = ctx->A12;
    REAL *B12; B12 = ctx->B12;
    REAL *C12; C12 = ctx->C12;
    REAL *A21; A21 = ctx->A21;
    REAL *B21; B21 = ctx->B21;
    REAL *C21; C21 = ctx->C21;
    REAL *A22; A22 = ctx->A22;
    REAL *B22; B22 = ctx->B22;
    REAL *C22; C22 = ctx->C22;
    REAL *S1; S1 = ctx->S1;
    REAL *S2; S2 = ctx->S2;
    REAL *S3; S3 = ctx->S3;
    REAL *S4; S4 = ctx->S4;
    REAL *S5; S5 = ctx->S5;
    REAL *S6; S6 = ctx->S6;
    REAL *S7; S7 = ctx->S7;
    REAL *S8; S8 = ctx->S8;
    REAL *M2; M2 = ctx->M2;
    REAL *M5; M5 = ctx->M5;
    REAL *T1sMULT; T1sMULT = ctx->T1sMULT;
    PTR TempMatrixOffset; TempMatrixOffset = ctx->TempMatrixOffset;
    PTR MatrixOffsetA; MatrixOffsetA = ctx->MatrixOffsetA;
    PTR MatrixOffsetB; MatrixOffsetB = ctx->MatrixOffsetB;
    char *Heap; Heap = ctx->Heap;
    void *StartHeap; StartHeap = ctx->StartHeap;
    PTR RowIncrementA; RowIncrementA = ctx->RowIncrementA;
    PTR RowIncrementB; RowIncrementB = ctx->RowIncrementB;
    PTR RowIncrementC; RowIncrementC = ctx->RowIncrementC;
    hclib_start_finish();
OptimizedStrassenMultiply_par(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1) ;     ; hclib_end_finish();
}

static void pragma1143_hclib_async(void *____arg) {
    pragma1143 *ctx = (pragma1143 *)____arg;
    REAL *C; C = ctx->C;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    unsigned int MatrixSize; MatrixSize = ctx->MatrixSize;
    unsigned int RowWidthC; RowWidthC = ctx->RowWidthC;
    unsigned int RowWidthA; RowWidthA = ctx->RowWidthA;
    unsigned int RowWidthB; RowWidthB = ctx->RowWidthB;
    int Depth; Depth = ctx->Depth;
    unsigned int QuadrantSize; QuadrantSize = ctx->QuadrantSize;
    unsigned int QuadrantSizeInBytes; QuadrantSizeInBytes = ctx->QuadrantSizeInBytes;
    unsigned int Column; Column = ctx->Column;
    unsigned int Row; Row = ctx->Row;
    REAL *A12; A12 = ctx->A12;
    REAL *B12; B12 = ctx->B12;
    REAL *C12; C12 = ctx->C12;
    REAL *A21; A21 = ctx->A21;
    REAL *B21; B21 = ctx->B21;
    REAL *C21; C21 = ctx->C21;
    REAL *A22; A22 = ctx->A22;
    REAL *B22; B22 = ctx->B22;
    REAL *C22; C22 = ctx->C22;
    REAL *S1; S1 = ctx->S1;
    REAL *S2; S2 = ctx->S2;
    REAL *S3; S3 = ctx->S3;
    REAL *S4; S4 = ctx->S4;
    REAL *S5; S5 = ctx->S5;
    REAL *S6; S6 = ctx->S6;
    REAL *S7; S7 = ctx->S7;
    REAL *S8; S8 = ctx->S8;
    REAL *M2; M2 = ctx->M2;
    REAL *M5; M5 = ctx->M5;
    REAL *T1sMULT; T1sMULT = ctx->T1sMULT;
    PTR TempMatrixOffset; TempMatrixOffset = ctx->TempMatrixOffset;
    PTR MatrixOffsetA; MatrixOffsetA = ctx->MatrixOffsetA;
    PTR MatrixOffsetB; MatrixOffsetB = ctx->MatrixOffsetB;
    char *Heap; Heap = ctx->Heap;
    void *StartHeap; StartHeap = ctx->StartHeap;
    PTR RowIncrementA; RowIncrementA = ctx->RowIncrementA;
    PTR RowIncrementB; RowIncrementB = ctx->RowIncrementB;
    PTR RowIncrementC; RowIncrementC = ctx->RowIncrementC;
    hclib_start_finish();
OptimizedStrassenMultiply_par(C22, S3, S7, QuadrantSize, RowWidthC /*FIXME*/, QuadrantSize, QuadrantSize, Depth+1) ;     ; hclib_end_finish();
}

static void pragma1147_hclib_async(void *____arg) {
    pragma1147 *ctx = (pragma1147 *)____arg;
    REAL *C; C = ctx->C;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    unsigned int MatrixSize; MatrixSize = ctx->MatrixSize;
    unsigned int RowWidthC; RowWidthC = ctx->RowWidthC;
    unsigned int RowWidthA; RowWidthA = ctx->RowWidthA;
    unsigned int RowWidthB; RowWidthB = ctx->RowWidthB;
    int Depth; Depth = ctx->Depth;
    unsigned int QuadrantSize; QuadrantSize = ctx->QuadrantSize;
    unsigned int QuadrantSizeInBytes; QuadrantSizeInBytes = ctx->QuadrantSizeInBytes;
    unsigned int Column; Column = ctx->Column;
    unsigned int Row; Row = ctx->Row;
    REAL *A12; A12 = ctx->A12;
    REAL *B12; B12 = ctx->B12;
    REAL *C12; C12 = ctx->C12;
    REAL *A21; A21 = ctx->A21;
    REAL *B21; B21 = ctx->B21;
    REAL *C21; C21 = ctx->C21;
    REAL *A22; A22 = ctx->A22;
    REAL *B22; B22 = ctx->B22;
    REAL *C22; C22 = ctx->C22;
    REAL *S1; S1 = ctx->S1;
    REAL *S2; S2 = ctx->S2;
    REAL *S3; S3 = ctx->S3;
    REAL *S4; S4 = ctx->S4;
    REAL *S5; S5 = ctx->S5;
    REAL *S6; S6 = ctx->S6;
    REAL *S7; S7 = ctx->S7;
    REAL *S8; S8 = ctx->S8;
    REAL *M2; M2 = ctx->M2;
    REAL *M5; M5 = ctx->M5;
    REAL *T1sMULT; T1sMULT = ctx->T1sMULT;
    PTR TempMatrixOffset; TempMatrixOffset = ctx->TempMatrixOffset;
    PTR MatrixOffsetA; MatrixOffsetA = ctx->MatrixOffsetA;
    PTR MatrixOffsetB; MatrixOffsetB = ctx->MatrixOffsetB;
    char *Heap; Heap = ctx->Heap;
    void *StartHeap; StartHeap = ctx->StartHeap;
    PTR RowIncrementA; RowIncrementA = ctx->RowIncrementA;
    PTR RowIncrementB; RowIncrementB = ctx->RowIncrementB;
    PTR RowIncrementC; RowIncrementC = ctx->RowIncrementC;
    hclib_start_finish();
OptimizedStrassenMultiply_par(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1) ;     ; hclib_end_finish();
}

static void pragma1151_hclib_async(void *____arg) {
    pragma1151 *ctx = (pragma1151 *)____arg;
    REAL *C; C = ctx->C;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    unsigned int MatrixSize; MatrixSize = ctx->MatrixSize;
    unsigned int RowWidthC; RowWidthC = ctx->RowWidthC;
    unsigned int RowWidthA; RowWidthA = ctx->RowWidthA;
    unsigned int RowWidthB; RowWidthB = ctx->RowWidthB;
    int Depth; Depth = ctx->Depth;
    unsigned int QuadrantSize; QuadrantSize = ctx->QuadrantSize;
    unsigned int QuadrantSizeInBytes; QuadrantSizeInBytes = ctx->QuadrantSizeInBytes;
    unsigned int Column; Column = ctx->Column;
    unsigned int Row; Row = ctx->Row;
    REAL *A12; A12 = ctx->A12;
    REAL *B12; B12 = ctx->B12;
    REAL *C12; C12 = ctx->C12;
    REAL *A21; A21 = ctx->A21;
    REAL *B21; B21 = ctx->B21;
    REAL *C21; C21 = ctx->C21;
    REAL *A22; A22 = ctx->A22;
    REAL *B22; B22 = ctx->B22;
    REAL *C22; C22 = ctx->C22;
    REAL *S1; S1 = ctx->S1;
    REAL *S2; S2 = ctx->S2;
    REAL *S3; S3 = ctx->S3;
    REAL *S4; S4 = ctx->S4;
    REAL *S5; S5 = ctx->S5;
    REAL *S6; S6 = ctx->S6;
    REAL *S7; S7 = ctx->S7;
    REAL *S8; S8 = ctx->S8;
    REAL *M2; M2 = ctx->M2;
    REAL *M5; M5 = ctx->M5;
    REAL *T1sMULT; T1sMULT = ctx->T1sMULT;
    PTR TempMatrixOffset; TempMatrixOffset = ctx->TempMatrixOffset;
    PTR MatrixOffsetA; MatrixOffsetA = ctx->MatrixOffsetA;
    PTR MatrixOffsetB; MatrixOffsetB = ctx->MatrixOffsetB;
    char *Heap; Heap = ctx->Heap;
    void *StartHeap; StartHeap = ctx->StartHeap;
    PTR RowIncrementA; RowIncrementA = ctx->RowIncrementA;
    PTR RowIncrementB; RowIncrementB = ctx->RowIncrementB;
    PTR RowIncrementC; RowIncrementC = ctx->RowIncrementC;
    hclib_start_finish();
OptimizedStrassenMultiply_par(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1) ;     ; hclib_end_finish();
}

static void pragma1155_hclib_async(void *____arg) {
    pragma1155 *ctx = (pragma1155 *)____arg;
    REAL *C; C = ctx->C;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    unsigned int MatrixSize; MatrixSize = ctx->MatrixSize;
    unsigned int RowWidthC; RowWidthC = ctx->RowWidthC;
    unsigned int RowWidthA; RowWidthA = ctx->RowWidthA;
    unsigned int RowWidthB; RowWidthB = ctx->RowWidthB;
    int Depth; Depth = ctx->Depth;
    unsigned int QuadrantSize; QuadrantSize = ctx->QuadrantSize;
    unsigned int QuadrantSizeInBytes; QuadrantSizeInBytes = ctx->QuadrantSizeInBytes;
    unsigned int Column; Column = ctx->Column;
    unsigned int Row; Row = ctx->Row;
    REAL *A12; A12 = ctx->A12;
    REAL *B12; B12 = ctx->B12;
    REAL *C12; C12 = ctx->C12;
    REAL *A21; A21 = ctx->A21;
    REAL *B21; B21 = ctx->B21;
    REAL *C21; C21 = ctx->C21;
    REAL *A22; A22 = ctx->A22;
    REAL *B22; B22 = ctx->B22;
    REAL *C22; C22 = ctx->C22;
    REAL *S1; S1 = ctx->S1;
    REAL *S2; S2 = ctx->S2;
    REAL *S3; S3 = ctx->S3;
    REAL *S4; S4 = ctx->S4;
    REAL *S5; S5 = ctx->S5;
    REAL *S6; S6 = ctx->S6;
    REAL *S7; S7 = ctx->S7;
    REAL *S8; S8 = ctx->S8;
    REAL *M2; M2 = ctx->M2;
    REAL *M5; M5 = ctx->M5;
    REAL *T1sMULT; T1sMULT = ctx->T1sMULT;
    PTR TempMatrixOffset; TempMatrixOffset = ctx->TempMatrixOffset;
    PTR MatrixOffsetA; MatrixOffsetA = ctx->MatrixOffsetA;
    PTR MatrixOffsetB; MatrixOffsetB = ctx->MatrixOffsetB;
    char *Heap; Heap = ctx->Heap;
    void *StartHeap; StartHeap = ctx->StartHeap;
    PTR RowIncrementA; RowIncrementA = ctx->RowIncrementA;
    PTR RowIncrementB; RowIncrementB = ctx->RowIncrementB;
    PTR RowIncrementC; RowIncrementC = ctx->RowIncrementC;
    hclib_start_finish();
OptimizedStrassenMultiply_par(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1) ;     ; hclib_end_finish();
}


#endif
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

typedef struct _pragma1277 {
    REAL *A;
    REAL *B;
    REAL *C;
    int n;
 } pragma1277;

static void pragma1277_hclib_async(void *____arg);
typedef struct _main_entrypoint_ctx {
    REAL *A;
    REAL *B;
    REAL *C;
    int n;
 } main_entrypoint_ctx;

static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    REAL *C; C = ctx->C;
    int n; n = ctx->n;
{
hclib_start_finish(); {
 { 
pragma1277 *ctx = (pragma1277 *)malloc(sizeof(pragma1277));
ctx->A = A;
ctx->B = B;
ctx->C = C;
ctx->n = n;
hclib_async(pragma1277_hclib_async, ctx, NO_FUTURE, ANY_PLACE);
 } ;
            } ; hclib_end_finish(); 
	bots_message(" completed!\n");
    } ; }

void strassen_main_par(REAL *A, REAL *B, REAL *C, int n)
{
	bots_message("Computing parallel Strassen algorithm (n=%d) ", n);
main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
ctx->A = A;
ctx->B = B;
ctx->C = C;
ctx->n = n;
hclib_launch(main_entrypoint, ctx);
free(ctx);

}  static void pragma1277_hclib_async(void *____arg) {
    pragma1277 *ctx = (pragma1277 *)____arg;
    REAL *A; A = ctx->A;
    REAL *B; B = ctx->B;
    REAL *C; C = ctx->C;
    int n; n = ctx->n;
    hclib_start_finish();
OptimizedStrassenMultiply_par(C, A, B, n, n, n, n, 1) ;     ; hclib_end_finish();
}



void strassen_main_seq(REAL *A, REAL *B, REAL *C, int n)
{
	bots_message("Computing sequential Strassen algorithm (n=%d) ", n);
	OptimizedStrassenMultiply_seq(C, A, B, n, n, n, n, 1);
	bots_message(" completed!\n");
}

