#include "header.h"


void update_diagonal ( int k, int j, int i, int tileSize, TileBlock* in_lkji_jjk, TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jjkp1 ) {
	int iB, jB, kB;
	double temp = 0;
	double** aBlock = in_lkji_jjk->matrixBlock;
	double** l2Block = in_lkji_jkkp1->matrixBlock;

	for( jB = 0; jB < tileSize ; ++jB ) {
		for( kB = 0; kB < tileSize ; ++kB ) {
			temp = 0 - l2Block[ jB ][ kB ];
			for( iB = jB; iB < tileSize; ++iB )
				aBlock[ iB ][ jB ] += temp * l2Block[ iB ][ kB ];
		}
	}
	out_lkji_jjkp1->matrixBlock = aBlock;
}
