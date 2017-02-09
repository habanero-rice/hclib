#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
pthread_mutex_t critical_0_lock = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

/* Original code from the Application Kernel Matrix by Cray */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "app-desc.h"
#include "bots.h"

#define ROWS 64
#define COLS 64
#define DMAX 64
#define max(a, b) ((a > b) ? a : b)
#define min(a, b) ((a < b) ? a : b)

int solution = -1;

typedef int  coor[2];
typedef char ibrd[ROWS * COLS];
typedef char (*pibrd)[COLS];

FILE * inputFile;

struct cell {
  int   n;
  coor *alt;
  int   top;
  int   bot;
  int   lhs;
  int   rhs;
  int   left;
  int   above;
  int   next;
};

struct cell * gcells;

int  MIN_AREA;
ibrd BEST_BOARD;
coor MIN_FOOTPRINT;

int N;

/* compute all possible locations for nw corner for cell */
static int starts(int id, int shape, coor *NWS, struct cell *cells) {
  int i, n, top, bot, lhs, rhs;
  int rows, cols, left, above;

/* size of cell */
  rows  = cells[id].alt[shape][0];
  cols  = cells[id].alt[shape][1];

/* the cells to the left and above */
  left  = cells[id].left;
  above = cells[id].above;

/* if there is a vertical and horizontal dependence */
  if ((left >= 0) && (above >= 0)) {

     top = cells[above].bot + 1;
     lhs = cells[left].rhs + 1;
     bot = top + rows;
     rhs = lhs + cols;

/* if footprint of cell touches the cells to the left and above */
     if ((top <= cells[left].bot) && (bot >= cells[left].top) &&
         (lhs <= cells[above].rhs) && (rhs >= cells[above].lhs))
          { n = 1; NWS[0][0] = top; NWS[0][1] = lhs;  }
     else { n = 0; }

/* if there is only a horizontal dependence */
   } else if (left >= 0) {

/* highest initial row is top of cell to the left - rows */ 
     top = max(cells[left].top - rows + 1, 0);
/* lowest initial row is bottom of cell to the left */
     bot = min(cells[left].bot, ROWS);
     n   = bot - top + 1;

     for (i = 0; i < n; i++) {
         NWS[i][0] = i + top;
         NWS[i][1] = cells[left].rhs + 1;
     }

  } else {

/* leftmost initial col is lhs of cell above - cols */
     lhs = max(cells[above].lhs - cols + 1, 0);
/* rightmost initial col is rhs of cell above */
     rhs = min(cells[above].rhs, COLS);
     n   = rhs - lhs + 1;

     for (i = 0; i < n; i++) {
         NWS[i][0] = cells[above].bot + 1;
         NWS[i][1] = i + lhs;
  }  }

  return (n);
}



/* lay the cell down on the board in the rectangular space defined
   by the cells top, bottom, left, and right edges. If the cell can
   not be layed down, return 0; else 1.
*/
static int lay_down(int id, ibrd board, struct cell *cells) {
  int  i, j, top, bot, lhs, rhs;

  top = cells[id].top;
  bot = cells[id].bot;
  lhs = cells[id].lhs;
  rhs = cells[id].rhs;

  for (i = top; i <= bot; i++) {
  for (j = lhs; j <= rhs; j++) {
      if (board[i * COLS + j] == 0) board[i * COLS + j] = (char)id;
      else                  return(0);
  } }

  return (1);
}


#define read_integer(file,var) \
  if ( fscanf(file, "%d", &var) == EOF ) {\
	bots_message(" Bogus input file\n");\
	exit(-1);\
  }

static void read_inputs() {
  int i, j, n;

  read_integer(inputFile,n);
  N = n;
  
  gcells = (struct cell *) malloc((n + 1) * sizeof(struct cell));

  gcells[0].n     =  0;
  gcells[0].alt   =  0;
  gcells[0].top   =  0;
  gcells[0].bot   =  0;
  gcells[0].lhs   = -1;
  gcells[0].rhs   = -1;
  gcells[0].left  =  0;
  gcells[0].above =  0;
  gcells[0].next  =  0;

  for (i = 1; i < n + 1; i++) {

      read_integer(inputFile, gcells[i].n);
      gcells[i].alt = (coor *) malloc(gcells[i].n * sizeof(coor));

      for (j = 0; j < gcells[i].n; j++) {
          read_integer(inputFile, gcells[i].alt[j][0]);
          read_integer(inputFile, gcells[i].alt[j][1]);
      }

      read_integer(inputFile, gcells[i].left);
      read_integer(inputFile, gcells[i].above);
      read_integer(inputFile, gcells[i].next);
      }

  if (!feof(inputFile)) {
      read_integer(inputFile, solution);
  }
}


static void write_outputs() {
  int i, j;

    bots_message("Minimum area = %d\n\n", MIN_AREA);

    for (i = 0; i < MIN_FOOTPRINT[0]; i++) {
      for (j = 0; j < MIN_FOOTPRINT[1]; j++) {
          if (BEST_BOARD[i * COLS + j] == 0) {bots_message(" ");}
          else                       bots_message("%c", 'A' + BEST_BOARD[i * COLS + j] - 1);
      } 
      bots_message("\n");
    }  
}

typedef struct _pragma220_omp_task {
    int i;
    int j;
    int nn;
    int area;
    int (*nnc_ptr);
    int (*nnl_ptr);
    char board[4096];
    int footprint[2];
    int NWS[64][2];
    int id;
    int (*(*FOOTPRINT_ptr));
    char (*(*BOARD_ptr));
    struct cell (*(*CELLS_ptr));
    int (*dummy_level_ptr);
 } pragma220_omp_task;

static void pragma220_omp_task_hclib_async(void *____arg);
static int add_cell(int id, coor FOOTPRINT, ibrd BOARD, struct cell *CELLS, int dummy_level) {
  int  i, j, nn, area, nnc,nnl;

  ibrd board;
  coor footprint, NWS[DMAX];

  nnc = nnl = 0;

/* for each possible shape */
  for (i = 0; i < CELLS[id].n; i++) {
/* compute all possible locations for nw corner */
      nn = starts(id, i, NWS, CELLS);
      nnl += nn;
/* for all possible locations */
      for (j = 0; j < nn; j++) {
 { 
pragma220_omp_task *new_ctx = (pragma220_omp_task *)malloc(sizeof(pragma220_omp_task));
new_ctx->i = i;
new_ctx->j = j;
new_ctx->nn = nn;
new_ctx->area = area;
new_ctx->nnc_ptr = &(nnc);
new_ctx->nnl_ptr = &(nnl);
memcpy(new_ctx->board, board, 4096 * (sizeof(char))); 
memcpy(new_ctx->footprint, footprint, 2 * (sizeof(int))); 
memcpy(new_ctx->NWS, NWS, 64 * (2 * (sizeof(int)))); 
new_ctx->id = id;
new_ctx->FOOTPRINT_ptr = &(FOOTPRINT);
new_ctx->BOARD_ptr = &(BOARD);
new_ctx->CELLS_ptr = &(CELLS);
new_ctx->dummy_level_ptr = &(dummy_level);
hclib_async(pragma220_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
      }
}
 hclib_end_finish(); hclib_start_finish(); ;
return nnc+nnl;
} 
static void pragma220_omp_task_hclib_async(void *____arg) {
    pragma220_omp_task *ctx = (pragma220_omp_task *)____arg;
    int i; i = ctx->i;
    int j; j = ctx->j;
    int nn; nn = ctx->nn;
    int area; area = ctx->area;
    char board[4096]; memcpy(board, ctx->board, 4096 * (sizeof(char))); 
    int footprint[2]; memcpy(footprint, ctx->footprint, 2 * (sizeof(int))); 
    int NWS[64][2]; memcpy(NWS, ctx->NWS, 64 * (2 * (sizeof(int)))); 
    int id; id = ctx->id;
    hclib_start_finish();
{
	  struct cell cells[N+1];
	  memcpy(cells,(*(ctx->CELLS_ptr)),sizeof(struct cell)*(N+1));
/* extent of shape */
          cells[id].top = NWS[j][0];
          cells[id].bot = cells[id].top + cells[id].alt[i][0] - 1;
          cells[id].lhs = NWS[j][1];
          cells[id].rhs = cells[id].lhs + cells[id].alt[i][1] - 1;

          memcpy(board, (*(ctx->BOARD_ptr)), sizeof(ibrd));

/* if the cell cannot be layed down, prune search */
          if (! lay_down(id, board, cells)) {
             bots_debug("Chip %d, shape %d does not fit\n", id, i);
             goto _end;
          }

/* calculate new footprint of board and area of footprint */
          footprint[0] = ((*(ctx->FOOTPRINT_ptr))[0] > cells[id].bot+1) ? (*(ctx->FOOTPRINT_ptr))[0] : cells[id].bot + 1;
          footprint[1] = ((*(ctx->FOOTPRINT_ptr))[1] > cells[id].rhs+1) ? (*(ctx->FOOTPRINT_ptr))[1] : cells[id].rhs + 1;
          area         = footprint[0] * footprint[1];

/* if last cell */
          if (cells[id].next == 0) {

/* if area is minimum, update global values */
		  if (area < MIN_AREA) {
 { const int ____lock_0_err = pthread_mutex_lock(&critical_0_lock); assert(____lock_0_err == 0); if (area < MIN_AREA) {
				  MIN_AREA         = area;
				  MIN_FOOTPRINT[0] = footprint[0];
				  MIN_FOOTPRINT[1] = footprint[1];
				  memcpy(BEST_BOARD, board, sizeof(ibrd));
				  bots_debug("N  %d\n", MIN_AREA);
			  }; const int ____unlock_0_err = pthread_mutex_unlock(&critical_0_lock); assert(____unlock_0_err == 0); } 
		  }

/* if area is less than best area */
          } else if (area < MIN_AREA) {
__sync_fetch_and_add(&((*(ctx->nnc_ptr))), add_cell(cells[id].next, footprint, board,cells, 0)); ;
/* if area is greater than or equal to best area, prune search */
          } else {

             bots_debug("T  %d, %d\n", area, MIN_AREA);
 
	  }
_end:;  
} ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}



ibrd board;

void floorplan_init (char *filename)
{
    int i,j;

    inputFile = fopen(filename, "r");
    
    if(NULL == inputFile) {
        bots_message("Couldn't open %s file for reading\n", filename);
        exit(1);
    }
    
    /* read input file and initialize global minimum area */
    read_inputs();
    MIN_AREA = ROWS * COLS;
    
    /* initialize board is empty */
    for (i = 0; i < ROWS; i++)
    for (j = 0; j < COLS; j++) board[i * COLS + j] = 0;
    
}

typedef struct _main_entrypoint_ctx {
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
{
        coor footprint;
        /* footprint of initial board is zero */
        footprint[0] = 0;
        footprint[1] = 0;
        bots_message("Computing floorplan ");
hclib_start_finish(); bots_number_of_tasks = add_cell(1, footprint, board, gcells, 0) ; hclib_end_finish(); 
        bots_message(" completed!\n");
    } ;     free(____arg);
}

void compute_floorplan (void)
{
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);

} 

void floorplan_end (void)
{
    /* write results */
    write_outputs();
    exit(0);
}

int floorplan_verify (void)
{
    if (solution != -1 )
      return MIN_AREA == solution ? BOTS_RESULT_SUCCESSFUL : BOTS_RESULT_UNSUCCESSFUL;
    else
      return BOTS_RESULT_NA;
}
