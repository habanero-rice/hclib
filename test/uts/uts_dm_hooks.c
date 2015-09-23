/*
 *         ---- The Unbalanced Tree Search (UTS) Benchmark ----
 *  
 *  Copyright (c) 2010 See AUTHORS file for copyright holders
 *
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.  See AUTHORS file for more information.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "uts.h"
#include "uts_dm_common.h"

/***********************************************************
 *  UTS Implementation Hooks                               *
 ***********************************************************/

// Return a string describing this implementation
char * impl_getName() {
  return ss_get_par_description();
}


// Construct string with this implementation's parameter settings 
int impl_paramsToStr(char *strBuf, int ind) {
  // search method
  ind += sprintf(strBuf+ind, "Execution strategy:  ");
  ind += sprintf(strBuf+ind, "Parallel search using %d threads\n", ss_get_num_threads());
  ind += sprintf(strBuf+ind, "   Load balance by work stealing\n");
      
  return ind;
}

// Parse command-line flags
int impl_parseParam(char *param, char *value) {
  int err = 0;  // Return 0 on a match, nonzero on an error

  switch (param[1]) {
    case 'c':
      break;
    case 'i':
      break;
    default:
      err = 1;
      break;
  }

  return err;
}

// Add this to the generic help message
void impl_helpMessage() {
}


void impl_abort(int err) {
  ss_abort(err);
}


