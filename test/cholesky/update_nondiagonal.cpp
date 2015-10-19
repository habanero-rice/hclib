#include "header.h"


void update_nondiagonal ( int k, int j, int i, int tileSize, TileBlock* in_lkji_jik, TileBlock* in_lkji_ikkp1, TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jikp1 ) {
	double temp;
	int jB, kB, iB;
	double**  aBlock = in_lkji_jik->matrixBlock;
	double** l1Block = in_lkji_jkkp1->matrixBlock;
	double** l2Block = in_lkji_ikkp1->matrixBlock;

	for( jB = 0; jB < tileSize ; ++jB ) {
		for( kB = 0; kB < tileSize ; ++kB ) {
			temp = 0 - l2Block[ jB ][ kB ];
			for( iB = 0; iB < tileSize ; ++iB )
				aBlock[ iB ][ jB ] += temp * l1Block[ iB ][ kB ];
		}
	}
	out_lkji_jikp1->matrixBlock = aBlock;
}
