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
