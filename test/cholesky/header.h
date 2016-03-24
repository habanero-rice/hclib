#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include "hclib_cpp.h"

using namespace hclib;

typedef struct TileBlock {
	double **matrixBlock;
} TileBlock;

void sequential_cholesky (int k, int tileSize, TileBlock* in_lkji_kkk, TileBlock* out_lkji_kkkp1);
void trisolve (int k, int j, int tileSize, TileBlock* in_lkji_jkk, TileBlock* in_lkji_kkkp1, TileBlock* out_lkji_jkkp1);
void update_diagonal (int k, int j, int i, int tileSize, TileBlock* in_lkji_jjk, TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jjkp1);
void update_nondiagonal (int k, int j, int i, int tileSize, TileBlock* in_lkji_jik, TileBlock* in_lkji_ikkp1, TileBlock* in_lkji_jkkp1, TileBlock* out_lkji_jikp1);

