#include "header.h"

void sequential_cholesky ( int k, int tileSize, TileBlock* in_lkji_kkk, TileBlock* out_lkji_kkkp1 ) {
	int index = 0, iB = 0, jB = 0, kB = 0, jBB = 0;
	double** aBlock = in_lkji_kkk->matrixBlock;
	double** lBlock = (double**) malloc(sizeof(double*)*tileSize);
	for( index = 0; index < tileSize; ++index )
		lBlock[index] = (double*) malloc(sizeof(double)*(index+1));

	for( kB = 0 ; kB < tileSize ; ++kB ) {
		if( aBlock[ kB ][ kB ] <= 0 ) {
			fprintf(stderr,"Not a symmetric positive definite (SPD) matrix\n"); exit(1);
		} else {
			lBlock[ kB ][ kB ] = sqrt( aBlock[ kB ][ kB ] );
		}

		for(jB = kB + 1; jB < tileSize ; ++jB )
			lBlock[ jB ][ kB ] = aBlock[ jB ][ kB ] / lBlock[ kB ][ kB ];

		for(jBB= kB + 1; jBB < tileSize ; ++jBB )
			for(iB = jBB ; iB < tileSize ; ++iB )
				aBlock[ iB ][ jBB ] -= lBlock[ iB ][ kB ] * lBlock[ jBB ][ kB ];
	}
	out_lkji_kkkp1->matrixBlock = lBlock;
}
