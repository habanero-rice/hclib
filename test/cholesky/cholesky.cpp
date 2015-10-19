#include "header.h"

int main(int argc, char** argv) {
	init(&argc, argv);
	int i, j, k, ii;
	double **A, ** temp;
	int A_i, A_j, T_i, T_j;
	FILE *in, *out;
	int i_b, j_b;

	int matrixSize = -1;
	int tileSize = -1;
	int numTiles = -1;

	TileBlock **** lkji; 
	if ( argc !=  4 ) {
		printf("Usage: ./Cholesky matrixSize tileSize fileName (found %d args)\n", argc);
		return 1;
	}

	matrixSize = atoi(argv[1]);
	tileSize = atoi(argv[2]);
	if ( matrixSize % tileSize != 0) {
		printf("Incorrect tile size %d for the matrix of size %d \n", tileSize, matrixSize);
		return 1;
	}

	numTiles = matrixSize/tileSize;


	lkji = (TileBlock ****) malloc(sizeof(TileBlock***)*numTiles);
	for( i = 0 ; i < numTiles ; ++i ) {
		lkji[i] = (TileBlock***) malloc(sizeof(TileBlock**)*(i+1));;
		for( j = 0 ; j <= i ; ++j ) {
			lkji[i][j] = (TileBlock**) malloc(sizeof(TileBlock*)*(numTiles+1));
			for( k = 0 ; k <= numTiles ; ++k )
				lkji[i][j][k] = (TileBlock*) malloc(sizeof(TileBlock));
			// Allocate memory for the tiles.
			lkji[i][j][0]->matrixBlock = (double**) malloc(sizeof(double*)*tileSize);
			for( ii = 0; ii < tileSize; ++ii )
                                lkji[i][j][0]->matrixBlock[ii] = (double*) malloc(sizeof(double)*tileSize); 
		}
	}


	in = fopen(argv[3], "r");
	if( !in ) {
		printf("Cannot find file: %s\n", argv[3]);
		return 1;
	}
	A = (double**) malloc(sizeof(double*)*matrixSize);
	for( i = 0; i < matrixSize; ++i)
		A[i] = (double*) malloc(sizeof(double)*matrixSize);

	for( i = 0; i < matrixSize; ++i ) {
		for( j = 0; j < matrixSize-1; ++j )
			fscanf(in, "%lf ", &A[i][j]);
		fscanf(in, "%lf\n", &A[i][j]);
	}

	for( i = 0 ; i < numTiles ; ++i ) {
		for( j = 0 ; j <= i ; ++j ) {
			// Split the matrix into tiles and write it into the item space at time 0.
			// The tiles are indexed by tile indices (which are tag values).
			temp = lkji[i][j][0]->matrixBlock;
			for( A_i = i*tileSize, T_i = 0 ; T_i < tileSize; ++A_i, ++T_i ) {
				for( A_j = j*tileSize, T_j = 0 ; T_j < tileSize; ++A_j, ++T_j ) {
					temp[ T_i ][ T_j ] = A[ A_i ][ A_j ];
				}
			}
		}
	}

	struct timeval a;
	struct timeval b;
	gettimeofday(&a, 0);

	for (int k = 0; k < numTiles; ++k ) {
		TileBlock *prevPivotTile = lkji[k][k][k];
		TileBlock *currPivotTile = lkji[k][k][k+1];

		sequential_cholesky (k, tileSize, prevPivotTile, currPivotTile );
		// Taking this malloc out from triSolve method
		for(int j = k + 1 ; j < numTiles ; ++j ) {
			TileBlock *currPivotColumnTile = lkji[j][k][k+1];
			currPivotColumnTile->matrixBlock = (double**) malloc (sizeof(double*)*tileSize);
        		for(i = 0; i < tileSize; ++i)
                		currPivotColumnTile->matrixBlock[i] = (double*) malloc (sizeof(double)*tileSize);
		}

		finish ([=]() {
			for(int j = k + 1 ; j < numTiles ; ++j ) {
				TileBlock *prevPivotColumnTile = lkji[j][k][k];
				TileBlock *currPivotColumnTile = lkji[j][k][k+1];

				async([=]() {
					trisolve (k, j, tileSize , prevPivotColumnTile, currPivotTile, currPivotColumnTile);
				});
			}
		});
		finish([=]() {
			for(int j = k + 1 ; j < numTiles ; ++j ) {
				TileBlock *prevPivotColumnTile = lkji[j][k][k];
				TileBlock *currPivotColumnTile = lkji[j][k][k+1];
				for(int i = k + 1 ; i < j ; ++i ) {
					TileBlock *prevTileForUpdate = lkji[j][i][k];
					TileBlock *currTileForUpdate = lkji[j][i][k+1];
					TileBlock *currPivotColumnOtherTile = lkji[i][k][k+1];

					async([=]() {
						update_nondiagonal ( k, j, i, tileSize, prevTileForUpdate, currPivotColumnOtherTile, currPivotColumnTile, currTileForUpdate);
					});
				}
				TileBlock *prevDiagonalTileForUpdate = lkji[j][j][k];
				TileBlock *currDiagonalTileForUpdate = lkji[j][j][k+1];

				async([=]() {
					update_diagonal ( k, j, j, tileSize , prevDiagonalTileForUpdate, currPivotColumnTile, currDiagonalTileForUpdate);
				});
			}
		});
	}

	gettimeofday(&b, 0);
	printf("The computation took %f seconds\r\n",((b.tv_sec - a.tv_sec)*1000000+(b.tv_usec - a.tv_usec))*1.0/1000000);

	out = fopen("cholesky.out", "w");
	for ( i = 0; i < numTiles; ++i ) {
		for( i_b = 0; i_b < tileSize; ++i_b) {
			int k = 1;
			for ( j = 0; j <= i; ++j ) {
				temp = lkji[i][j][k]->matrixBlock;
				if(i != j) {
					for(j_b = 0; j_b < tileSize; ++j_b) {
						fprintf( out, "%lf ", temp[i_b][j_b]);
					}
				} else {
					for(j_b = 0; j_b <= i_b; ++j_b) {
						fprintf( out, "%lf ", temp[i_b][j_b]);
					}
				}
				++k;
			}
		}
	}

	for( i = 0; i < matrixSize; ++i )
		free(A[i]);
	free(A);
	finalize();
	return 0;
}






