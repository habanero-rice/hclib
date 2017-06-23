/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except f_in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to f_in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "hc.h"

typedef struct { double** matrixBlock; } TileBlock;

void sequential_cholesky(int k, int tileSize, TileBlock* in_lkji_kkk,
                         TileBlock* out_lkji_kkkp1);
void trisolve(int k, int j, int tileSize, TileBlock* in_lkji_jkk,
              TileBlock* in_lkji_kkkp1, TileBlock* out_lkji_jkkp1);
void update_diagonal(int k, int j, int i, int tileSize, TileBlock* in_lkji_jjk,
                     TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jjkp1);
void update_nondiagonal(int k, int j, int i, int tileSize,
                        TileBlock* in_lkji_jik, TileBlock* in_lkji_ikkp1,
                        TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jikp1);

void sequential_cholesky(int k, int tileSize, TileBlock* in_lkji_kkk,
                         TileBlock* out_lkji_kkkp1) {
    int index = 0, iB = 0, jB = 0, kB = 0, jBB = 0;
    double** aBlock = in_lkji_kkk->matrixBlock;
    double** lBlock = malloc(tileSize * sizeof(double*));
    for (index = 0; index < tileSize; ++index)
        lBlock[index] = malloc((index + 1) * sizeof(double));

    for (kB = 0; kB < tileSize; ++kB) {
        if (aBlock[kB][kB] <= 0) {
            fprintf(stderr, "Not a symmetric positive definite (SPD) matrix\n");
            exit(1);
        } else {
            lBlock[kB][kB] = sqrt(aBlock[kB][kB]);
        }

        for (jB = kB + 1; jB < tileSize; ++jB)
            lBlock[jB][kB] = aBlock[jB][kB] / lBlock[kB][kB];

        for (jBB = kB + 1; jBB < tileSize; ++jBB)
            for (iB = jBB; iB < tileSize; ++iB)
                aBlock[iB][jBB] -= lBlock[iB][kB] * lBlock[jBB][kB];
    }
    out_lkji_kkkp1->matrixBlock = lBlock;
}

void trisolve(int k, int j, int tileSize, TileBlock* in_lkji_jkk,
              TileBlock* in_lkji_kkkp1, TileBlock* out_lkji_jkkp1) {
    int iB, jB, kB;
    double** aBlock = in_lkji_jkk->matrixBlock;
    double** liBlock = in_lkji_kkkp1->matrixBlock;
    double** loBlock = out_lkji_jkkp1->matrixBlock;

    for (kB = 0; kB < tileSize; ++kB) {
        for (iB = 0; iB < tileSize; ++iB)
            loBlock[iB][kB] = aBlock[iB][kB] / liBlock[kB][kB];

        for (jB = kB + 1; jB < tileSize; ++jB)
            for (iB = 0; iB < tileSize; ++iB)
                aBlock[iB][jB] -= liBlock[jB][kB] * loBlock[iB][kB];
    }
}

void update_diagonal(int k, int j, int i, int tileSize, TileBlock* in_lkji_jjk,
                     TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jjkp1) {
    int iB, jB, kB;
    double temp = 0;
    double** aBlock = in_lkji_jjk->matrixBlock;
    double** l2Block = in_lkji_jkkp1->matrixBlock;

    for (jB = 0; jB < tileSize; ++jB) {
        for (kB = 0; kB < tileSize; ++kB) {
            temp = 0 - l2Block[jB][kB];
            for (iB = jB; iB < tileSize; ++iB)
                aBlock[iB][jB] += temp * l2Block[iB][kB];
        }
    }
    out_lkji_jjkp1->matrixBlock = aBlock;
}

void update_nondiagonal(int k, int j, int i, int tileSize,
                        TileBlock* in_lkji_jik, TileBlock* in_lkji_ikkp1,
                        TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jikp1) {
    double temp;
    int jB, kB, iB;
    double** aBlock = in_lkji_jik->matrixBlock;
    double** l1Block = in_lkji_jkkp1->matrixBlock;
    double** l2Block = in_lkji_ikkp1->matrixBlock;

    for (jB = 0; jB < tileSize; ++jB) {
        for (kB = 0; kB < tileSize; ++kB) {
            temp = 0 - l2Block[jB][kB];
            for (iB = 0; iB < tileSize; ++iB)
                aBlock[iB][jB] += temp * l1Block[iB][kB];
        }
    }
    out_lkji_jikp1->matrixBlock = aBlock;
}

int main(int argc, char** argv) {
    int i, j, k, ii;
    double **A, **temp;
    int A_i, A_j, T_i, T_j;
    FILE *f_in, *f_out;
    int i_b, j_b;

    int matrixSize = -1;
    int tileSize = -1;
    int numTiles = -1;

    TileBlock**** lkji;
    if (argc != 4) {
        printf("Usage: ./Cholesky matrixSize tileSize fileName ");
        printf("(found %d args)\n", argc);
        exit(1);
    }

    matrixSize = atoi(argv[1]);
    tileSize = atoi(argv[2]);
    if (matrixSize % tileSize != 0) {
        printf("Incorrect tile size %d for the matrix of size %d \n", tileSize,
               matrixSize);
        exit(1);
    }

    numTiles = matrixSize / tileSize;

    f_in = fopen(argv[3], "r");
    if (!f_in) {
        printf("Cannot find file: %s\n", argv[3]);
        exit(1);
    }

    lkji = malloc(numTiles * sizeof(TileBlock***));
    for (i = 0; i < numTiles; ++i) {
        lkji[i] = malloc((i + 1) * sizeof(TileBlock**));
        for (j = 0; j <= i; ++j) {
            lkji[i][j] = malloc((numTiles + 1) * sizeof(TileBlock*));
            for (k = 0; k <= numTiles; ++k)
                lkji[i][j][k] = malloc(sizeof(TileBlock));
            // Allocate memory for the tiles.
            lkji[i][j][0]->matrixBlock = malloc(tileSize * sizeof(double*));
            for (ii = 0; ii < tileSize; ++ii)
                lkji[i][j][0]->matrixBlock[ii] =
                        malloc(tileSize * sizeof(double));
        }
    }

    A = malloc(matrixSize * sizeof(double*));
    for (i = 0; i < matrixSize; ++i) A[i] = malloc(matrixSize * sizeof(double));

    for (i = 0; i < matrixSize; ++i) {
        for (j = 0; j < matrixSize - 1; ++j) fscanf(f_in, "%lf ", &A[i][j]);
        fscanf(f_in, "%lf\n", &A[i][j]);
    }

    for (i = 0; i < numTiles; ++i) {
        for (j = 0; j <= i; ++j) {
            // Split the matrix into tiles and write it into the item space
            // at time 0. The tiles are indexed by tile indices (which are
            // tag values).
            temp = lkji[i][j][0]->matrixBlock;
            for (A_i = i * tileSize, T_i = 0; T_i < tileSize; ++A_i, ++T_i) {
                for (A_j = j * tileSize, T_j = 0; T_j < tileSize;
                     ++A_j, ++T_j) {
                    temp[T_i][T_j] = A[A_i][A_j];
                }
            }
        }
    }

    struct timeval a;
    struct timeval b;
    gettimeofday(&a, 0);

    for (int k = 0; k < numTiles; ++k) {
        TileBlock* prevPivotTile = lkji[k][k][k];
        TileBlock* currPivotTile = lkji[k][k][k + 1];

        sequential_cholesky(k, tileSize, prevPivotTile, currPivotTile);
        // Taking this malloc f_out from triSolve method
        for (int j = k + 1; j < numTiles; ++j) {
            TileBlock* currPivotColumnTile = lkji[j][k][k + 1];
            currPivotColumnTile->matrixBlock =
                    malloc(tileSize * sizeof(double*));
            for (i = 0; i < tileSize; ++i)
                currPivotColumnTile->matrixBlock[i] =
                        malloc(tileSize * sizeof(double));
        }

        finish {
            for (int j = k + 1; j < numTiles; ++j) {
                TileBlock* prevPivotColumnTile = lkji[j][k][k];
                TileBlock* currPivotColumnTile = lkji[j][k][k + 1];

                async IN(k, j, tileSize, prevPivotColumnTile, currPivotTile,
                         currPivotColumnTile) {
                    trisolve(k, j, tileSize, prevPivotColumnTile, currPivotTile,
                             currPivotColumnTile);
                }
            }
        }
        finish {
            for (int j = k + 1; j < numTiles; ++j) {
                TileBlock* prevPivotColumnTile = lkji[j][k][k];
                TileBlock* currPivotColumnTile = lkji[j][k][k + 1];
                for (int i = k + 1; i < j; ++i) {
                    TileBlock* prevTileForUpdate = lkji[j][i][k];
                    TileBlock* currTileForUpdate = lkji[j][i][k + 1];
                    TileBlock* currPivotColumnOtherTile = lkji[i][k][k + 1];

                    async IN(k, j, i, tileSize, prevTileForUpdate,
                             currPivotColumnOtherTile, currPivotColumnTile,
                             currTileForUpdate) {
                        update_nondiagonal(k, j, i, tileSize, prevTileForUpdate,
                                           currPivotColumnOtherTile,
                                           currPivotColumnTile,
                                           currTileForUpdate);
                    }
                }
                TileBlock* prevDiagonalTileForUpdate = lkji[j][j][k];
                TileBlock* currDiagonalTileForUpdate = lkji[j][j][k + 1];

                async IN(k, j, tileSize, prevDiagonalTileForUpdate,
                         currPivotColumnTile, currDiagonalTileForUpdate) {
                    update_diagonal(
                            k, j, j, tileSize, prevDiagonalTileForUpdate,
                            currPivotColumnTile, currDiagonalTileForUpdate);
                }
            }
        }
    }

    gettimeofday(&b, 0);
    printf("The computation took %f seconds\r\n",
           ((b.tv_sec - a.tv_sec) * 1000000 + (b.tv_usec - a.tv_usec)) * 1.0 /
                   1000000);

    f_out = fopen("cholesky.out", "w");
    for (i = 0; i < numTiles; ++i) {
        for (i_b = 0; i_b < tileSize; ++i_b) {
            int k = 1;
            for (j = 0; j <= i; ++j) {
                temp = lkji[i][j][k]->matrixBlock;
                if (i != j) {
                    for (j_b = 0; j_b < tileSize; ++j_b) {
                        fprintf(f_out, "%lf ", temp[i_b][j_b]);
                    }
                } else {
                    for (j_b = 0; j_b <= i_b; ++j_b) {
                        fprintf(f_out, "%lf ", temp[i_b][j_b]);
                    }
                }
                ++k;
            }
        }
    }

    for (i = 0; i < matrixSize; ++i) free(A[i]);
    free(A);

    fclose(f_out);
    fclose(f_in);
}
