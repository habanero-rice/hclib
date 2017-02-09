/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; -*- */

#include <hpcc.h>

static void
PTRANS_ladd(int *J, int *K, int *I) {
  int itmp0 = K[0] + J[0], itmp1;
  itmp1 = itmp0 >> 16;         I[0] = itmp0 - ( itmp1 << 16 );
  itmp0 = itmp1 + K[1] + J[1]; I[1] = itmp0 - (( itmp0 >> 15 ) << 15);
}

static void
PTRANS_lmul(int *K, int *J, int *I) {
  static int ipow30 = ( 1 << 30 );
  int kt, lt;
  kt   = K[0] * J[0]; if( kt < 0 ) kt = ( kt + ipow30 ) + ipow30;
  I[0] = kt - ( ( kt >> 16 ) << 16 );
  lt   = K[0] * J[1] + K[1] * J[0];
  if( lt < 0 ) lt = ( lt + ipow30 ) + ipow30;
  kt = ( kt >> 16 ) + lt;
  if( kt < 0 ) kt = ( kt + ipow30 ) + ipow30;
  I[1] = kt - ( ( kt >> 15 ) << 15 );
}

static struct {
    int irand[2], ias[2], ics[2];
} rancom_;

#define rancom_1 rancom_

/*  -- ScaLAPACK routines (version 1.7) -- */
/*     University of Tennessee, Knoxville, Oak Ridge National Laboratory, */
/*     and University of California, Berkeley. */
/*     May 1, 1997 */

int
xjumpm_(int *jumpm, int *mult, int *iadd,
	int *irann, int *iranm, int *iam, int *icm) {
  int i__1;

  int i, j[2];

  if (*jumpm > 0) {
	  for (i = 0; i < 2; ++i) {
	    iam[i] = mult[i];
	    icm[i] = iadd[i];
	  }
	  i__1 = *jumpm - 1;
	  for (i = 0; i < i__1; ++i) {
	    PTRANS_lmul( iam, mult, j);
	    iam[0] = j[0];
	    iam[1] = j[1];
	    PTRANS_lmul( icm, mult, j );
	    PTRANS_ladd( iadd, j, icm );
	  }
	  PTRANS_lmul( irann, iam, j );
	  PTRANS_ladd( j, icm, iranm );
  } else {
	  iranm[0] = irann[0];
	  iranm[1] = irann[1];
  }

  return 0;
} /* xjumpm_ */

int
setran_(int *iran, int *ia, int *ic) {
  int i;

  for (i = 0; i < 2; ++i) {
    rancom_1.irand[i] = iran[i];
    rancom_1.ias[i] = ia[i];
    rancom_1.ics[i] = ic[i];
  }

  return 0;
} /* setran_ */


int
jumpit_(int *mult, int *iadd, int *irann, int *iranm) {
  int j[2];

  PTRANS_lmul( irann, mult, j);
  PTRANS_ladd( j, iadd, iranm );

  rancom_1.irand[0] = iranm[0];
  rancom_1.irand[1] = iranm[1];

  return 0;
} /* jumpit_ */

double
pdrand() {
  /* System generated locals */
  double ret_val;

  /* Local variables */
  int j[2];

  ret_val = ((double) rancom_1.irand[0] + (double) rancom_1.irand[1]
	     * 65536.0) / 2147483648.0;

  PTRANS_lmul(rancom_1.irand, rancom_1.ias, j);
  PTRANS_ladd(j, rancom_1.ics, rancom_1.irand);

  return ret_val;
} /* pdrand */
