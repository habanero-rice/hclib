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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include "hclib.hpp"

using namespace hclib;

typedef struct TileBlock {
	double **matrixBlock;
} TileBlock;

void sequential_cholesky (int k, int tileSize, TileBlock* in_lkji_kkk, TileBlock* out_lkji_kkkp1);
void trisolve (int k, int j, int tileSize, TileBlock* in_lkji_jkk, TileBlock* in_lkji_kkkp1, TileBlock* out_lkji_jkkp1);
void update_diagonal (int k, int j, int i, int tileSize, TileBlock* in_lkji_jjk, TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jjkp1);
void update_nondiagonal (int k, int j, int i, int tileSize, TileBlock* in_lkji_jik, TileBlock* in_lkji_ikkp1, TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jikp1);

