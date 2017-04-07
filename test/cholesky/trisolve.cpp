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
