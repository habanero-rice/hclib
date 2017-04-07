/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "header.h"

void sequential_cholesky ( int k, int tileSize, TileBlock* in_lkji_kkk, TileBlock* out_lkji_kkkp1 ) {
	int index = 0, iB = 0, jB = 0, kB = 0, jBB = 0;
	double** aBlock = in_lkji_kkk->matrixBlock;
	double** lBlock = new double*[tileSize];
	for( index = 0; index < tileSize; ++index )
		lBlock[index] = new double[(index + 1)];

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
