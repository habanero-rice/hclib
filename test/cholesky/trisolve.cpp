#include "header.h"

void trisolve ( int k, int j, int tileSize, TileBlock* in_lkji_jkk, TileBlock* in_lkji_kkkp1, TileBlock* out_lkji_jkkp1 ) {
	int iB, jB, kB;
	double** aBlock = in_lkji_jkk->matrixBlock;
	double** liBlock = in_lkji_kkkp1->matrixBlock;
	double ** loBlock = out_lkji_jkkp1->matrixBlock;

	for( kB = 0; kB < tileSize ; ++kB ) {
		for( iB = 0; iB < tileSize ; ++iB )
			loBlock[ iB ][ kB ] = aBlock[ iB ][ kB ] / liBlock[ kB ][ kB ];

		for( jB = kB + 1 ; jB < tileSize; ++jB )
			for( iB = 0; iB < tileSize; ++iB )
				aBlock[ iB ][ jB ] -= liBlock[ jB ][ kB ] * loBlock[ iB ][ kB ];
	}
}
