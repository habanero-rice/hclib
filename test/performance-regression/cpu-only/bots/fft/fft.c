#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#endif
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

/* 
 * Original code from the Cilk project 
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "bots.h"
#include "app-desc.h"

/* Definitions and operations for complex numbers */

/*
 * compute the W coefficients (that is, powers of the root of 1)
 * and store them into an array.
 */
typedef struct _pragma65_omp_task {
    int (*ab_ptr);
    double (*twoPiOverN_ptr);
    int (*k_ptr);
    double (*s_ptr);
    double (*c_ptr);
    int (*n_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*W_ptr));
 } pragma65_omp_task;

typedef struct _pragma67_omp_task {
    int (*ab_ptr);
    double (*twoPiOverN_ptr);
    int (*k_ptr);
    double (*s_ptr);
    double (*c_ptr);
    int (*n_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*W_ptr));
 } pragma67_omp_task;

static void pragma65_omp_task_hclib_async(void *____arg);
static void pragma67_omp_task_hclib_async(void *____arg);
void compute_w_coefficients(int n, int a, int b, COMPLEX * W)
{
     register double twoPiOverN;
     register int k;
     register REAL s, c;

     if (b - a < 128) {
	  twoPiOverN = 2.0 * 3.1415926535897932384626434 / n;
	  for (k = a; k <= b; ++k) {
	       c = cos(twoPiOverN * k);
	       c_re(W[k]) = c_re(W[n - k]) = c;
	       s = sin(twoPiOverN * k);
	       c_im(W[k]) = -s;
	       c_im(W[n - k]) = s;
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma65_omp_task *new_ctx = (pragma65_omp_task *)malloc(sizeof(pragma65_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->twoPiOverN_ptr = &(twoPiOverN);
new_ctx->k_ptr = &(k);
new_ctx->s_ptr = &(s);
new_ctx->c_ptr = &(c);
new_ctx->n_ptr = &(n);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->W_ptr = &(W);
hclib_async(pragma65_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma67_omp_task *new_ctx = (pragma67_omp_task *)malloc(sizeof(pragma67_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->twoPiOverN_ptr = &(twoPiOverN);
new_ctx->k_ptr = &(k);
new_ctx->s_ptr = &(s);
new_ctx->c_ptr = &(c);
new_ctx->n_ptr = &(n);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->W_ptr = &(W);
hclib_async(pragma67_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma65_omp_task_hclib_async(void *____arg) {
    pragma65_omp_task *ctx = (pragma65_omp_task *)____arg;
    hclib_start_finish();
compute_w_coefficients((*(ctx->n_ptr)), (*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->W_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma67_omp_task_hclib_async(void *____arg) {
    pragma67_omp_task *ctx = (pragma67_omp_task *)____arg;
    hclib_start_finish();
compute_w_coefficients((*(ctx->n_ptr)), (*(ctx->ab_ptr)) + 1, (*(ctx->b_ptr)), (*(ctx->W_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void compute_w_coefficients_seq(int n, int a, int b, COMPLEX * W)
{
     register double twoPiOverN;
     register int k;
     register REAL s, c;

     if (b - a < 128) {
	  twoPiOverN = 2.0 * 3.1415926535897932384626434 / n;
	  for (k = a; k <= b; ++k) {
	       c = cos(twoPiOverN * k);
	       c_re(W[k]) = c_re(W[n - k]) = c;
	       s = sin(twoPiOverN * k);
	       c_im(W[k]) = -s;
	       c_im(W[n - k]) = s;
	  }
     } else {
	  int ab = (a + b) / 2;
	  compute_w_coefficients_seq(n, a, ab, W);
	  compute_w_coefficients_seq(n, ab + 1, b, W);
     }
}
/*
 * Determine (in a stupid way) if n is divisible by eight, then by four, else
 * find the smallest prime factor of n.
 */
int factor(int n)
{
     int r;

     if (n < 2) return 1;
     if (n == 64 || n == 128 || n == 256 || n == 1024 || n == 2048 || n == 4096) return 8;
     if ((n & 15) == 0) return 16;
     if ((n & 7) == 0) return 8;
     if ((n & 3) == 0) return 4;
     if ((n & 1) == 0) return 2;

     /* try odd numbers up to n (computing the sqrt may be slower) */
     for (r = 3; r < n; r += 2) if (n % r == 0) return r;

     /* n is prime */
     return n;
}

typedef struct _pragma142_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    int (*j_ptr);
    int (*r4_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*r_ptr);
    int (*m_ptr);
 } pragma142_omp_task;

typedef struct _pragma144_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    int (*j_ptr);
    int (*r4_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*r_ptr);
    int (*m_ptr);
 } pragma144_omp_task;

static void pragma142_omp_task_hclib_async(void *____arg);
static void pragma144_omp_task_hclib_async(void *____arg);
void unshuffle(int a, int b, COMPLEX * in, COMPLEX * out, int r, int m)
{
     int i, j;
     int r4 = r & (~0x3);
     COMPLEX *ip;
     COMPLEX *jp;

     if (b - a < 16) {
	  ip = in + a * r;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       for (j = 0; j < r4; j += 4) {
		    jp[0] = ip[0];
		    jp[m] = ip[1];
		    jp[2 * m] = ip[2];
		    jp[3 * m] = ip[3];
		    jp += 4 * m;
		    ip += 4;
	       }
	       for (; j < r; ++j) {
		    *jp = *ip;
		    ip++;
		    jp += m;
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma142_omp_task *new_ctx = (pragma142_omp_task *)malloc(sizeof(pragma142_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->j_ptr = &(j);
new_ctx->r4_ptr = &(r4);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
hclib_async(pragma142_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma144_omp_task *new_ctx = (pragma144_omp_task *)malloc(sizeof(pragma144_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->j_ptr = &(j);
new_ctx->r4_ptr = &(r4);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
hclib_async(pragma144_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma142_omp_task_hclib_async(void *____arg) {
    pragma142_omp_task *ctx = (pragma142_omp_task *)____arg;
    hclib_start_finish();
unshuffle((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->r_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma144_omp_task_hclib_async(void *____arg) {
    pragma144_omp_task *ctx = (pragma144_omp_task *)____arg;
    hclib_start_finish();
unshuffle((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->r_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void unshuffle_seq(int a, int b, COMPLEX * in, COMPLEX * out, int r, int m)
{
     int i, j;
     int r4 = r & (~0x3);
     COMPLEX *ip;
     COMPLEX *jp;

     if (b - a < 16) {
	  ip = in + a * r;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       for (j = 0; j < r4; j += 4) {
		    jp[0] = ip[0];
		    jp[m] = ip[1];
		    jp[2 * m] = ip[2];
		    jp[3 * m] = ip[3];
		    jp += 4 * m;
		    ip += 4;
	       }
	       for (; j < r; ++j) {
		    *jp = *ip;
		    ip++;
		    jp += m;
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
	  unshuffle_seq(a, ab, in, out, r, m);
	  unshuffle_seq(ab, b, in, out, r, m);
     }
}
void fft_twiddle_gen1(COMPLEX * in, COMPLEX * out,
				  COMPLEX * W, int r, int m,
				  int nW, int nWdnti, int nWdntm)
{
     int j, k;
     COMPLEX *jp, *kp;

     for (k = 0, kp = out; k < r; ++k, kp += m) {
	  REAL r0, i0, rt, it, rw, iw;
	  int l1 = nWdnti + nWdntm * k;
	  int l0;

	  r0 = i0 = 0.0;
	  for (j = 0, jp = in, l0 = 0; j < r; ++j, jp += m) {
	       rw = c_re(W[l0]);
	       iw = c_im(W[l0]);
	       rt = c_re(*jp);
	       it = c_im(*jp);
	       r0 += rt * rw - it * iw;
	       i0 += rt * iw + it * rw;
	       l0 += l1;
	       if (l0 > nW)
		    l0 -= nW;
	  }
	  c_re(*kp) = r0;
	  c_im(*kp) = i0;
     }
}

typedef struct _pragma212_omp_task {
    int (*i_ptr);
    int (*i1_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*r_ptr);
    int (*m_ptr);
 } pragma212_omp_task;

typedef struct _pragma217_omp_task {
    int (*i2_ptr);
    int (*i_ptr);
    int (*i1_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*r_ptr);
    int (*m_ptr);
 } pragma217_omp_task;

typedef struct _pragma220_omp_task {
    int (*i2_ptr);
    int (*i_ptr);
    int (*i1_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*r_ptr);
    int (*m_ptr);
 } pragma220_omp_task;

static void pragma212_omp_task_hclib_async(void *____arg);
static void pragma217_omp_task_hclib_async(void *____arg);
static void pragma220_omp_task_hclib_async(void *____arg);
void fft_twiddle_gen(int i, int i1, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int r, int m)
{
     if (i == i1 - 1) {
 { 
pragma212_omp_task *new_ctx = (pragma212_omp_task *)malloc(sizeof(pragma212_omp_task));
new_ctx->i_ptr = &(i);
new_ctx->i1_ptr = &(i1);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
hclib_async(pragma212_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     } else {
	  int i2 = (i + i1) / 2;
 { 
pragma217_omp_task *new_ctx = (pragma217_omp_task *)malloc(sizeof(pragma217_omp_task));
new_ctx->i2_ptr = &(i2);
new_ctx->i_ptr = &(i);
new_ctx->i1_ptr = &(i1);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
hclib_async(pragma217_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma220_omp_task *new_ctx = (pragma220_omp_task *)malloc(sizeof(pragma220_omp_task));
new_ctx->i2_ptr = &(i2);
new_ctx->i_ptr = &(i);
new_ctx->i1_ptr = &(i1);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
hclib_async(pragma220_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     }
 hclib_end_finish(); hclib_start_finish(); ;
} 
static void pragma212_omp_task_hclib_async(void *____arg) {
    pragma212_omp_task *ctx = (pragma212_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_gen1((*(ctx->in_ptr)) + (*(ctx->i_ptr)), (*(ctx->out_ptr)) + (*(ctx->i_ptr)), (*(ctx->W_ptr)),
				 (*(ctx->r_ptr)), (*(ctx->m_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)) * (*(ctx->i_ptr)), (*(ctx->nWdn_ptr)) * (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma217_omp_task_hclib_async(void *____arg) {
    pragma217_omp_task *ctx = (pragma217_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_gen((*(ctx->i_ptr)), (*(ctx->i2_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)),
				(*(ctx->nWdn_ptr)), (*(ctx->r_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma220_omp_task_hclib_async(void *____arg) {
    pragma220_omp_task *ctx = (pragma220_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_gen((*(ctx->i2_ptr)), (*(ctx->i1_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)),
				(*(ctx->nWdn_ptr)), (*(ctx->r_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_twiddle_gen_seq(int i, int i1, COMPLEX * in, COMPLEX * out, COMPLEX * W,
                         int nW, int nWdn, int r, int m)
{
     if (i == i1 - 1) {
	  fft_twiddle_gen1(in + i, out + i, W,
				 r, m, nW, nWdn * i, nWdn * m);
     } else {
	  int i2 = (i + i1) / 2;
	  fft_twiddle_gen_seq(i, i2, in, out, W, nW,
				nWdn, r, m);
	  fft_twiddle_gen_seq(i2, i1, in, out, W, nW,
				nWdn, r, m);
     }
}
/* machine-generated code begins here */
void fft_base_2(COMPLEX * in, COMPLEX * out)
{
     REAL r1_0, i1_0;
     REAL r1_1, i1_1;
     r1_0 = c_re(in[0]);
     i1_0 = c_im(in[0]);
     r1_1 = c_re(in[1]);
     i1_1 = c_im(in[1]);
     c_re(out[0]) = (r1_0 + r1_1);
     c_im(out[0]) = (i1_0 + i1_1);
     c_re(out[1]) = (r1_0 - r1_1);
     c_im(out[1]) = (i1_0 - i1_1);
}
typedef struct _pragma282_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma282_omp_task;

typedef struct _pragma284_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma284_omp_task;

static void pragma282_omp_task_hclib_async(void *____arg);
static void pragma284_omp_task_hclib_async(void *____arg);
void fft_twiddle_2(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    r1_0 = c_re(jp[0 * m]);
		    i1_0 = c_im(jp[0 * m]);
		    wr = c_re(W[1 * l1]);
		    wi = c_im(W[1 * l1]);
		    tmpr = c_re(jp[1 * m]);
		    tmpi = c_im(jp[1 * m]);
		    r1_1 = ((wr * tmpr) - (wi * tmpi));
		    i1_1 = ((wi * tmpr) + (wr * tmpi));
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[1 * m]) = (r1_0 - r1_1);
		    c_im(kp[1 * m]) = (i1_0 - i1_1);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma282_omp_task *new_ctx = (pragma282_omp_task *)malloc(sizeof(pragma282_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma282_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma284_omp_task *new_ctx = (pragma284_omp_task *)malloc(sizeof(pragma284_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma284_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma282_omp_task_hclib_async(void *____arg) {
    pragma282_omp_task *ctx = (pragma282_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_2((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma284_omp_task_hclib_async(void *____arg) {
    pragma284_omp_task *ctx = (pragma284_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_2((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_twiddle_2_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    r1_0 = c_re(jp[0 * m]);
		    i1_0 = c_im(jp[0 * m]);
		    wr = c_re(W[1 * l1]);
		    wi = c_im(W[1 * l1]);
		    tmpr = c_re(jp[1 * m]);
		    tmpi = c_im(jp[1 * m]);
		    r1_1 = ((wr * tmpr) - (wi * tmpi));
		    i1_1 = ((wi * tmpr) + (wr * tmpi));
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[1 * m]) = (r1_0 - r1_1);
		    c_im(kp[1 * m]) = (i1_0 - i1_1);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_twiddle_2_seq(a, ab, in, out, W, nW, nWdn, m);
	  fft_twiddle_2_seq(ab, b, in, out, W, nW, nWdn, m);
     }
}
typedef struct _pragma336_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma336_omp_task;

typedef struct _pragma338_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma338_omp_task;

static void pragma336_omp_task_hclib_async(void *____arg);
static void pragma338_omp_task_hclib_async(void *____arg);
void fft_unshuffle_2(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 2;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma336_omp_task *new_ctx = (pragma336_omp_task *)malloc(sizeof(pragma336_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma336_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma338_omp_task *new_ctx = (pragma338_omp_task *)malloc(sizeof(pragma338_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma338_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma336_omp_task_hclib_async(void *____arg) {
    pragma336_omp_task *ctx = (pragma336_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_2((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma338_omp_task_hclib_async(void *____arg) {
    pragma338_omp_task *ctx = (pragma338_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_2((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_unshuffle_2_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 2;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_unshuffle_2_seq(a, ab, in, out, m);
	  fft_unshuffle_2_seq(ab, b, in, out, m);
     }
}
void fft_base_4(COMPLEX * in, COMPLEX * out)
{
     REAL r1_0, i1_0;
     REAL r1_1, i1_1;
     REAL r1_2, i1_2;
     REAL r1_3, i1_3;
     {
	  REAL r2_0, i2_0;
	  REAL r2_2, i2_2;
	  r2_0 = c_re(in[0]);
	  i2_0 = c_im(in[0]);
	  r2_2 = c_re(in[2]);
	  i2_2 = c_im(in[2]);
	  r1_0 = (r2_0 + r2_2);
	  i1_0 = (i2_0 + i2_2);
	  r1_2 = (r2_0 - r2_2);
	  i1_2 = (i2_0 - i2_2);
     }
     {
	  REAL r2_1, i2_1;
	  REAL r2_3, i2_3;
	  r2_1 = c_re(in[1]);
	  i2_1 = c_im(in[1]);
	  r2_3 = c_re(in[3]);
	  i2_3 = c_im(in[3]);
	  r1_1 = (r2_1 + r2_3);
	  i1_1 = (i2_1 + i2_3);
	  r1_3 = (r2_1 - r2_3);
	  i1_3 = (i2_1 - i2_3);
     }
     c_re(out[0]) = (r1_0 + r1_1);
     c_im(out[0]) = (i1_0 + i1_1);
     c_re(out[2]) = (r1_0 - r1_1);
     c_im(out[2]) = (i1_0 - i1_1);
     c_re(out[1]) = (r1_2 + i1_3);
     c_im(out[1]) = (i1_2 - r1_3);
     c_re(out[3]) = (r1_2 - i1_3);
     c_im(out[3]) = (i1_2 + r1_3);
}
typedef struct _pragma463_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma463_omp_task;

typedef struct _pragma465_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma465_omp_task;

static void pragma463_omp_task_hclib_async(void *____arg);
static void pragma465_omp_task_hclib_async(void *____arg);
void fft_twiddle_4(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 r2_0 = c_re(jp[0 * m]);
			 i2_0 = c_im(jp[0 * m]);
			 wr = c_re(W[2 * l1]);
			 wi = c_im(W[2 * l1]);
			 tmpr = c_re(jp[2 * m]);
			 tmpi = c_im(jp[2 * m]);
			 r2_2 = ((wr * tmpr) - (wi * tmpi));
			 i2_2 = ((wi * tmpr) + (wr * tmpi));
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_2 = (r2_0 - r2_2);
			 i1_2 = (i2_0 - i2_2);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 wr = c_re(W[1 * l1]);
			 wi = c_im(W[1 * l1]);
			 tmpr = c_re(jp[1 * m]);
			 tmpi = c_im(jp[1 * m]);
			 r2_1 = ((wr * tmpr) - (wi * tmpi));
			 i2_1 = ((wi * tmpr) + (wr * tmpi));
			 wr = c_re(W[3 * l1]);
			 wi = c_im(W[3 * l1]);
			 tmpr = c_re(jp[3 * m]);
			 tmpi = c_im(jp[3 * m]);
			 r2_3 = ((wr * tmpr) - (wi * tmpi));
			 i2_3 = ((wi * tmpr) + (wr * tmpi));
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_3 = (r2_1 - r2_3);
			 i1_3 = (i2_1 - i2_3);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[2 * m]) = (r1_0 - r1_1);
		    c_im(kp[2 * m]) = (i1_0 - i1_1);
		    c_re(kp[1 * m]) = (r1_2 + i1_3);
		    c_im(kp[1 * m]) = (i1_2 - r1_3);
		    c_re(kp[3 * m]) = (r1_2 - i1_3);
		    c_im(kp[3 * m]) = (i1_2 + r1_3);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma463_omp_task *new_ctx = (pragma463_omp_task *)malloc(sizeof(pragma463_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma463_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma465_omp_task *new_ctx = (pragma465_omp_task *)malloc(sizeof(pragma465_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma465_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma463_omp_task_hclib_async(void *____arg) {
    pragma463_omp_task *ctx = (pragma463_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_4((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma465_omp_task_hclib_async(void *____arg) {
    pragma465_omp_task *ctx = (pragma465_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_4((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_twiddle_4_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 r2_0 = c_re(jp[0 * m]);
			 i2_0 = c_im(jp[0 * m]);
			 wr = c_re(W[2 * l1]);
			 wi = c_im(W[2 * l1]);
			 tmpr = c_re(jp[2 * m]);
			 tmpi = c_im(jp[2 * m]);
			 r2_2 = ((wr * tmpr) - (wi * tmpi));
			 i2_2 = ((wi * tmpr) + (wr * tmpi));
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_2 = (r2_0 - r2_2);
			 i1_2 = (i2_0 - i2_2);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 wr = c_re(W[1 * l1]);
			 wi = c_im(W[1 * l1]);
			 tmpr = c_re(jp[1 * m]);
			 tmpi = c_im(jp[1 * m]);
			 r2_1 = ((wr * tmpr) - (wi * tmpi));
			 i2_1 = ((wi * tmpr) + (wr * tmpi));
			 wr = c_re(W[3 * l1]);
			 wi = c_im(W[3 * l1]);
			 tmpr = c_re(jp[3 * m]);
			 tmpi = c_im(jp[3 * m]);
			 r2_3 = ((wr * tmpr) - (wi * tmpi));
			 i2_3 = ((wi * tmpr) + (wr * tmpi));
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_3 = (r2_1 - r2_3);
			 i1_3 = (i2_1 - i2_3);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[2 * m]) = (r1_0 - r1_1);
		    c_im(kp[2 * m]) = (i1_0 - i1_1);
		    c_re(kp[1 * m]) = (r1_2 + i1_3);
		    c_im(kp[1 * m]) = (i1_2 - r1_3);
		    c_re(kp[3 * m]) = (r1_2 - i1_3);
		    c_im(kp[3 * m]) = (i1_2 + r1_3);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_twiddle_4_seq(a, ab, in, out, W, nW, nWdn, m);
	  fft_twiddle_4_seq(ab, b, in, out, W, nW, nWdn, m);
     }
}
typedef struct _pragma555_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma555_omp_task;

typedef struct _pragma557_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma557_omp_task;

static void pragma555_omp_task_hclib_async(void *____arg);
static void pragma557_omp_task_hclib_async(void *____arg);
void fft_unshuffle_4(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 4;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma555_omp_task *new_ctx = (pragma555_omp_task *)malloc(sizeof(pragma555_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma555_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma557_omp_task *new_ctx = (pragma557_omp_task *)malloc(sizeof(pragma557_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma557_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma555_omp_task_hclib_async(void *____arg) {
    pragma555_omp_task *ctx = (pragma555_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_4((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma557_omp_task_hclib_async(void *____arg) {
    pragma557_omp_task *ctx = (pragma557_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_4((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_unshuffle_4_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 4;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_unshuffle_4_seq(a, ab, in, out, m);
	  fft_unshuffle_4_seq(ab, b, in, out, m);
     }
}
void fft_base_8(COMPLEX * in, COMPLEX * out)
{
     REAL tmpr, tmpi;
     {
	  REAL r1_0, i1_0;
	  REAL r1_1, i1_1;
	  REAL r1_2, i1_2;
	  REAL r1_3, i1_3;
	  REAL r1_4, i1_4;
	  REAL r1_5, i1_5;
	  REAL r1_6, i1_6;
	  REAL r1_7, i1_7;
	  {
	       REAL r2_0, i2_0;
	       REAL r2_2, i2_2;
	       REAL r2_4, i2_4;
	       REAL r2_6, i2_6;
	       {
		    REAL r3_0, i3_0;
		    REAL r3_4, i3_4;
		    r3_0 = c_re(in[0]);
		    i3_0 = c_im(in[0]);
		    r3_4 = c_re(in[4]);
		    i3_4 = c_im(in[4]);
		    r2_0 = (r3_0 + r3_4);
		    i2_0 = (i3_0 + i3_4);
		    r2_4 = (r3_0 - r3_4);
		    i2_4 = (i3_0 - i3_4);
	       }
	       {
		    REAL r3_2, i3_2;
		    REAL r3_6, i3_6;
		    r3_2 = c_re(in[2]);
		    i3_2 = c_im(in[2]);
		    r3_6 = c_re(in[6]);
		    i3_6 = c_im(in[6]);
		    r2_2 = (r3_2 + r3_6);
		    i2_2 = (i3_2 + i3_6);
		    r2_6 = (r3_2 - r3_6);
		    i2_6 = (i3_2 - i3_6);
	       }
	       r1_0 = (r2_0 + r2_2);
	       i1_0 = (i2_0 + i2_2);
	       r1_4 = (r2_0 - r2_2);
	       i1_4 = (i2_0 - i2_2);
	       r1_2 = (r2_4 + i2_6);
	       i1_2 = (i2_4 - r2_6);
	       r1_6 = (r2_4 - i2_6);
	       i1_6 = (i2_4 + r2_6);
	  }
	  {
	       REAL r2_1, i2_1;
	       REAL r2_3, i2_3;
	       REAL r2_5, i2_5;
	       REAL r2_7, i2_7;
	       {
		    REAL r3_1, i3_1;
		    REAL r3_5, i3_5;
		    r3_1 = c_re(in[1]);
		    i3_1 = c_im(in[1]);
		    r3_5 = c_re(in[5]);
		    i3_5 = c_im(in[5]);
		    r2_1 = (r3_1 + r3_5);
		    i2_1 = (i3_1 + i3_5);
		    r2_5 = (r3_1 - r3_5);
		    i2_5 = (i3_1 - i3_5);
	       }
	       {
		    REAL r3_3, i3_3;
		    REAL r3_7, i3_7;
		    r3_3 = c_re(in[3]);
		    i3_3 = c_im(in[3]);
		    r3_7 = c_re(in[7]);
		    i3_7 = c_im(in[7]);
		    r2_3 = (r3_3 + r3_7);
		    i2_3 = (i3_3 + i3_7);
		    r2_7 = (r3_3 - r3_7);
		    i2_7 = (i3_3 - i3_7);
	       }
	       r1_1 = (r2_1 + r2_3);
	       i1_1 = (i2_1 + i2_3);
	       r1_5 = (r2_1 - r2_3);
	       i1_5 = (i2_1 - i2_3);
	       r1_3 = (r2_5 + i2_7);
	       i1_3 = (i2_5 - r2_7);
	       r1_7 = (r2_5 - i2_7);
	       i1_7 = (i2_5 + r2_7);
	  }
	  c_re(out[0]) = (r1_0 + r1_1);
	  c_im(out[0]) = (i1_0 + i1_1);
	  c_re(out[4]) = (r1_0 - r1_1);
	  c_im(out[4]) = (i1_0 - i1_1);
	  tmpr = (0.707106781187 * (r1_3 + i1_3));
	  tmpi = (0.707106781187 * (i1_3 - r1_3));
	  c_re(out[1]) = (r1_2 + tmpr);
	  c_im(out[1]) = (i1_2 + tmpi);
	  c_re(out[5]) = (r1_2 - tmpr);
	  c_im(out[5]) = (i1_2 - tmpi);
	  c_re(out[2]) = (r1_4 + i1_5);
	  c_im(out[2]) = (i1_4 - r1_5);
	  c_re(out[6]) = (r1_4 - i1_5);
	  c_im(out[6]) = (i1_4 + r1_5);
	  tmpr = (0.707106781187 * (i1_7 - r1_7));
	  tmpi = (0.707106781187 * (r1_7 + i1_7));
	  c_re(out[3]) = (r1_6 + tmpr);
	  c_im(out[3]) = (i1_6 - tmpi);
	  c_re(out[7]) = (r1_6 - tmpr);
	  c_im(out[7]) = (i1_6 + tmpi);
     }
}
typedef struct _pragma841_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma841_omp_task;

typedef struct _pragma843_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma843_omp_task;

static void pragma841_omp_task_hclib_async(void *____arg);
static void pragma843_omp_task_hclib_async(void *____arg);
void fft_twiddle_8(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    REAL r1_4, i1_4;
		    REAL r1_5, i1_5;
		    REAL r1_6, i1_6;
		    REAL r1_7, i1_7;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 REAL r2_4, i2_4;
			 REAL r2_6, i2_6;
			 {
			      REAL r3_0, i3_0;
			      REAL r3_4, i3_4;
			      r3_0 = c_re(jp[0 * m]);
			      i3_0 = c_im(jp[0 * m]);
			      wr = c_re(W[4 * l1]);
			      wi = c_im(W[4 * l1]);
			      tmpr = c_re(jp[4 * m]);
			      tmpi = c_im(jp[4 * m]);
			      r3_4 = ((wr * tmpr) - (wi * tmpi));
			      i3_4 = ((wi * tmpr) + (wr * tmpi));
			      r2_0 = (r3_0 + r3_4);
			      i2_0 = (i3_0 + i3_4);
			      r2_4 = (r3_0 - r3_4);
			      i2_4 = (i3_0 - i3_4);
			 }
			 {
			      REAL r3_2, i3_2;
			      REAL r3_6, i3_6;
			      wr = c_re(W[2 * l1]);
			      wi = c_im(W[2 * l1]);
			      tmpr = c_re(jp[2 * m]);
			      tmpi = c_im(jp[2 * m]);
			      r3_2 = ((wr * tmpr) - (wi * tmpi));
			      i3_2 = ((wi * tmpr) + (wr * tmpi));
			      wr = c_re(W[6 * l1]);
			      wi = c_im(W[6 * l1]);
			      tmpr = c_re(jp[6 * m]);
			      tmpi = c_im(jp[6 * m]);
			      r3_6 = ((wr * tmpr) - (wi * tmpi));
			      i3_6 = ((wi * tmpr) + (wr * tmpi));
			      r2_2 = (r3_2 + r3_6);
			      i2_2 = (i3_2 + i3_6);
			      r2_6 = (r3_2 - r3_6);
			      i2_6 = (i3_2 - i3_6);
			 }
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_4 = (r2_0 - r2_2);
			 i1_4 = (i2_0 - i2_2);
			 r1_2 = (r2_4 + i2_6);
			 i1_2 = (i2_4 - r2_6);
			 r1_6 = (r2_4 - i2_6);
			 i1_6 = (i2_4 + r2_6);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 REAL r2_5, i2_5;
			 REAL r2_7, i2_7;
			 {
			      REAL r3_1, i3_1;
			      REAL r3_5, i3_5;
			      wr = c_re(W[1 * l1]);
			      wi = c_im(W[1 * l1]);
			      tmpr = c_re(jp[1 * m]);
			      tmpi = c_im(jp[1 * m]);
			      r3_1 = ((wr * tmpr) - (wi * tmpi));
			      i3_1 = ((wi * tmpr) + (wr * tmpi));
			      wr = c_re(W[5 * l1]);
			      wi = c_im(W[5 * l1]);
			      tmpr = c_re(jp[5 * m]);
			      tmpi = c_im(jp[5 * m]);
			      r3_5 = ((wr * tmpr) - (wi * tmpi));
			      i3_5 = ((wi * tmpr) + (wr * tmpi));
			      r2_1 = (r3_1 + r3_5);
			      i2_1 = (i3_1 + i3_5);
			      r2_5 = (r3_1 - r3_5);
			      i2_5 = (i3_1 - i3_5);
			 }
			 {
			      REAL r3_3, i3_3;
			      REAL r3_7, i3_7;
			      wr = c_re(W[3 * l1]);
			      wi = c_im(W[3 * l1]);
			      tmpr = c_re(jp[3 * m]);
			      tmpi = c_im(jp[3 * m]);
			      r3_3 = ((wr * tmpr) - (wi * tmpi));
			      i3_3 = ((wi * tmpr) + (wr * tmpi));
			      wr = c_re(W[7 * l1]);
			      wi = c_im(W[7 * l1]);
			      tmpr = c_re(jp[7 * m]);
			      tmpi = c_im(jp[7 * m]);
			      r3_7 = ((wr * tmpr) - (wi * tmpi));
			      i3_7 = ((wi * tmpr) + (wr * tmpi));
			      r2_3 = (r3_3 + r3_7);
			      i2_3 = (i3_3 + i3_7);
			      r2_7 = (r3_3 - r3_7);
			      i2_7 = (i3_3 - i3_7);
			 }
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_5 = (r2_1 - r2_3);
			 i1_5 = (i2_1 - i2_3);
			 r1_3 = (r2_5 + i2_7);
			 i1_3 = (i2_5 - r2_7);
			 r1_7 = (r2_5 - i2_7);
			 i1_7 = (i2_5 + r2_7);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[4 * m]) = (r1_0 - r1_1);
		    c_im(kp[4 * m]) = (i1_0 - i1_1);
		    tmpr = (0.707106781187 * (r1_3 + i1_3));
		    tmpi = (0.707106781187 * (i1_3 - r1_3));
		    c_re(kp[1 * m]) = (r1_2 + tmpr);
		    c_im(kp[1 * m]) = (i1_2 + tmpi);
		    c_re(kp[5 * m]) = (r1_2 - tmpr);
		    c_im(kp[5 * m]) = (i1_2 - tmpi);
		    c_re(kp[2 * m]) = (r1_4 + i1_5);
		    c_im(kp[2 * m]) = (i1_4 - r1_5);
		    c_re(kp[6 * m]) = (r1_4 - i1_5);
		    c_im(kp[6 * m]) = (i1_4 + r1_5);
		    tmpr = (0.707106781187 * (i1_7 - r1_7));
		    tmpi = (0.707106781187 * (r1_7 + i1_7));
		    c_re(kp[3 * m]) = (r1_6 + tmpr);
		    c_im(kp[3 * m]) = (i1_6 - tmpi);
		    c_re(kp[7 * m]) = (r1_6 - tmpr);
		    c_im(kp[7 * m]) = (i1_6 + tmpi);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma841_omp_task *new_ctx = (pragma841_omp_task *)malloc(sizeof(pragma841_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma841_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma843_omp_task *new_ctx = (pragma843_omp_task *)malloc(sizeof(pragma843_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma843_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma841_omp_task_hclib_async(void *____arg) {
    pragma841_omp_task *ctx = (pragma841_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_8((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma843_omp_task_hclib_async(void *____arg) {
    pragma843_omp_task *ctx = (pragma843_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_8((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_twiddle_8_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    REAL r1_4, i1_4;
		    REAL r1_5, i1_5;
		    REAL r1_6, i1_6;
		    REAL r1_7, i1_7;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 REAL r2_4, i2_4;
			 REAL r2_6, i2_6;
			 {
			      REAL r3_0, i3_0;
			      REAL r3_4, i3_4;
			      r3_0 = c_re(jp[0 * m]);
			      i3_0 = c_im(jp[0 * m]);
			      wr = c_re(W[4 * l1]);
			      wi = c_im(W[4 * l1]);
			      tmpr = c_re(jp[4 * m]);
			      tmpi = c_im(jp[4 * m]);
			      r3_4 = ((wr * tmpr) - (wi * tmpi));
			      i3_4 = ((wi * tmpr) + (wr * tmpi));
			      r2_0 = (r3_0 + r3_4);
			      i2_0 = (i3_0 + i3_4);
			      r2_4 = (r3_0 - r3_4);
			      i2_4 = (i3_0 - i3_4);
			 }
			 {
			      REAL r3_2, i3_2;
			      REAL r3_6, i3_6;
			      wr = c_re(W[2 * l1]);
			      wi = c_im(W[2 * l1]);
			      tmpr = c_re(jp[2 * m]);
			      tmpi = c_im(jp[2 * m]);
			      r3_2 = ((wr * tmpr) - (wi * tmpi));
			      i3_2 = ((wi * tmpr) + (wr * tmpi));
			      wr = c_re(W[6 * l1]);
			      wi = c_im(W[6 * l1]);
			      tmpr = c_re(jp[6 * m]);
			      tmpi = c_im(jp[6 * m]);
			      r3_6 = ((wr * tmpr) - (wi * tmpi));
			      i3_6 = ((wi * tmpr) + (wr * tmpi));
			      r2_2 = (r3_2 + r3_6);
			      i2_2 = (i3_2 + i3_6);
			      r2_6 = (r3_2 - r3_6);
			      i2_6 = (i3_2 - i3_6);
			 }
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_4 = (r2_0 - r2_2);
			 i1_4 = (i2_0 - i2_2);
			 r1_2 = (r2_4 + i2_6);
			 i1_2 = (i2_4 - r2_6);
			 r1_6 = (r2_4 - i2_6);
			 i1_6 = (i2_4 + r2_6);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 REAL r2_5, i2_5;
			 REAL r2_7, i2_7;
			 {
			      REAL r3_1, i3_1;
			      REAL r3_5, i3_5;
			      wr = c_re(W[1 * l1]);
			      wi = c_im(W[1 * l1]);
			      tmpr = c_re(jp[1 * m]);
			      tmpi = c_im(jp[1 * m]);
			      r3_1 = ((wr * tmpr) - (wi * tmpi));
			      i3_1 = ((wi * tmpr) + (wr * tmpi));
			      wr = c_re(W[5 * l1]);
			      wi = c_im(W[5 * l1]);
			      tmpr = c_re(jp[5 * m]);
			      tmpi = c_im(jp[5 * m]);
			      r3_5 = ((wr * tmpr) - (wi * tmpi));
			      i3_5 = ((wi * tmpr) + (wr * tmpi));
			      r2_1 = (r3_1 + r3_5);
			      i2_1 = (i3_1 + i3_5);
			      r2_5 = (r3_1 - r3_5);
			      i2_5 = (i3_1 - i3_5);
			 }
			 {
			      REAL r3_3, i3_3;
			      REAL r3_7, i3_7;
			      wr = c_re(W[3 * l1]);
			      wi = c_im(W[3 * l1]);
			      tmpr = c_re(jp[3 * m]);
			      tmpi = c_im(jp[3 * m]);
			      r3_3 = ((wr * tmpr) - (wi * tmpi));
			      i3_3 = ((wi * tmpr) + (wr * tmpi));
			      wr = c_re(W[7 * l1]);
			      wi = c_im(W[7 * l1]);
			      tmpr = c_re(jp[7 * m]);
			      tmpi = c_im(jp[7 * m]);
			      r3_7 = ((wr * tmpr) - (wi * tmpi));
			      i3_7 = ((wi * tmpr) + (wr * tmpi));
			      r2_3 = (r3_3 + r3_7);
			      i2_3 = (i3_3 + i3_7);
			      r2_7 = (r3_3 - r3_7);
			      i2_7 = (i3_3 - i3_7);
			 }
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_5 = (r2_1 - r2_3);
			 i1_5 = (i2_1 - i2_3);
			 r1_3 = (r2_5 + i2_7);
			 i1_3 = (i2_5 - r2_7);
			 r1_7 = (r2_5 - i2_7);
			 i1_7 = (i2_5 + r2_7);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[4 * m]) = (r1_0 - r1_1);
		    c_im(kp[4 * m]) = (i1_0 - i1_1);
		    tmpr = (0.707106781187 * (r1_3 + i1_3));
		    tmpi = (0.707106781187 * (i1_3 - r1_3));
		    c_re(kp[1 * m]) = (r1_2 + tmpr);
		    c_im(kp[1 * m]) = (i1_2 + tmpi);
		    c_re(kp[5 * m]) = (r1_2 - tmpr);
		    c_im(kp[5 * m]) = (i1_2 - tmpi);
		    c_re(kp[2 * m]) = (r1_4 + i1_5);
		    c_im(kp[2 * m]) = (i1_4 - r1_5);
		    c_re(kp[6 * m]) = (r1_4 - i1_5);
		    c_im(kp[6 * m]) = (i1_4 + r1_5);
		    tmpr = (0.707106781187 * (i1_7 - r1_7));
		    tmpi = (0.707106781187 * (r1_7 + i1_7));
		    c_re(kp[3 * m]) = (r1_6 + tmpr);
		    c_im(kp[3 * m]) = (i1_6 - tmpi);
		    c_re(kp[7 * m]) = (r1_6 - tmpr);
		    c_im(kp[7 * m]) = (i1_6 + tmpi);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_twiddle_8_seq(a, ab, in, out, W, nW, nWdn, m);
	  fft_twiddle_8_seq(ab, b, in, out, W, nW, nWdn, m);
     }
}
typedef struct _pragma1025_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma1025_omp_task;

typedef struct _pragma1027_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma1027_omp_task;

static void pragma1025_omp_task_hclib_async(void *____arg);
static void pragma1027_omp_task_hclib_async(void *____arg);
void fft_unshuffle_8(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 8;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma1025_omp_task *new_ctx = (pragma1025_omp_task *)malloc(sizeof(pragma1025_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma1025_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma1027_omp_task *new_ctx = (pragma1027_omp_task *)malloc(sizeof(pragma1027_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma1027_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma1025_omp_task_hclib_async(void *____arg) {
    pragma1025_omp_task *ctx = (pragma1025_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_8((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma1027_omp_task_hclib_async(void *____arg) {
    pragma1027_omp_task *ctx = (pragma1027_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_8((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_unshuffle_8_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 8;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_unshuffle_8_seq(a, ab, in, out, m);
	  fft_unshuffle_8_seq(ab, b, in, out, m);
     }
}
void fft_base_16(COMPLEX * in, COMPLEX * out)
{
     REAL tmpr, tmpi;
     {
	  REAL r1_0, i1_0;
	  REAL r1_1, i1_1;
	  REAL r1_2, i1_2;
	  REAL r1_3, i1_3;
	  REAL r1_4, i1_4;
	  REAL r1_5, i1_5;
	  REAL r1_6, i1_6;
	  REAL r1_7, i1_7;
	  REAL r1_8, i1_8;
	  REAL r1_9, i1_9;
	  REAL r1_10, i1_10;
	  REAL r1_11, i1_11;
	  REAL r1_12, i1_12;
	  REAL r1_13, i1_13;
	  REAL r1_14, i1_14;
	  REAL r1_15, i1_15;
	  {
	       REAL r2_0, i2_0;
	       REAL r2_2, i2_2;
	       REAL r2_4, i2_4;
	       REAL r2_6, i2_6;
	       REAL r2_8, i2_8;
	       REAL r2_10, i2_10;
	       REAL r2_12, i2_12;
	       REAL r2_14, i2_14;
	       {
		    REAL r3_0, i3_0;
		    REAL r3_4, i3_4;
		    REAL r3_8, i3_8;
		    REAL r3_12, i3_12;
		    {
			 REAL r4_0, i4_0;
			 REAL r4_8, i4_8;
			 r4_0 = c_re(in[0]);
			 i4_0 = c_im(in[0]);
			 r4_8 = c_re(in[8]);
			 i4_8 = c_im(in[8]);
			 r3_0 = (r4_0 + r4_8);
			 i3_0 = (i4_0 + i4_8);
			 r3_8 = (r4_0 - r4_8);
			 i3_8 = (i4_0 - i4_8);
		    }
		    {
			 REAL r4_4, i4_4;
			 REAL r4_12, i4_12;
			 r4_4 = c_re(in[4]);
			 i4_4 = c_im(in[4]);
			 r4_12 = c_re(in[12]);
			 i4_12 = c_im(in[12]);
			 r3_4 = (r4_4 + r4_12);
			 i3_4 = (i4_4 + i4_12);
			 r3_12 = (r4_4 - r4_12);
			 i3_12 = (i4_4 - i4_12);
		    }
		    r2_0 = (r3_0 + r3_4);
		    i2_0 = (i3_0 + i3_4);
		    r2_8 = (r3_0 - r3_4);
		    i2_8 = (i3_0 - i3_4);
		    r2_4 = (r3_8 + i3_12);
		    i2_4 = (i3_8 - r3_12);
		    r2_12 = (r3_8 - i3_12);
		    i2_12 = (i3_8 + r3_12);
	       }
	       {
		    REAL r3_2, i3_2;
		    REAL r3_6, i3_6;
		    REAL r3_10, i3_10;
		    REAL r3_14, i3_14;
		    {
			 REAL r4_2, i4_2;
			 REAL r4_10, i4_10;
			 r4_2 = c_re(in[2]);
			 i4_2 = c_im(in[2]);
			 r4_10 = c_re(in[10]);
			 i4_10 = c_im(in[10]);
			 r3_2 = (r4_2 + r4_10);
			 i3_2 = (i4_2 + i4_10);
			 r3_10 = (r4_2 - r4_10);
			 i3_10 = (i4_2 - i4_10);
		    }
		    {
			 REAL r4_6, i4_6;
			 REAL r4_14, i4_14;
			 r4_6 = c_re(in[6]);
			 i4_6 = c_im(in[6]);
			 r4_14 = c_re(in[14]);
			 i4_14 = c_im(in[14]);
			 r3_6 = (r4_6 + r4_14);
			 i3_6 = (i4_6 + i4_14);
			 r3_14 = (r4_6 - r4_14);
			 i3_14 = (i4_6 - i4_14);
		    }
		    r2_2 = (r3_2 + r3_6);
		    i2_2 = (i3_2 + i3_6);
		    r2_10 = (r3_2 - r3_6);
		    i2_10 = (i3_2 - i3_6);
		    r2_6 = (r3_10 + i3_14);
		    i2_6 = (i3_10 - r3_14);
		    r2_14 = (r3_10 - i3_14);
		    i2_14 = (i3_10 + r3_14);
	       }
	       r1_0 = (r2_0 + r2_2);
	       i1_0 = (i2_0 + i2_2);
	       r1_8 = (r2_0 - r2_2);
	       i1_8 = (i2_0 - i2_2);
	       tmpr = (0.707106781187 * (r2_6 + i2_6));
	       tmpi = (0.707106781187 * (i2_6 - r2_6));
	       r1_2 = (r2_4 + tmpr);
	       i1_2 = (i2_4 + tmpi);
	       r1_10 = (r2_4 - tmpr);
	       i1_10 = (i2_4 - tmpi);
	       r1_4 = (r2_8 + i2_10);
	       i1_4 = (i2_8 - r2_10);
	       r1_12 = (r2_8 - i2_10);
	       i1_12 = (i2_8 + r2_10);
	       tmpr = (0.707106781187 * (i2_14 - r2_14));
	       tmpi = (0.707106781187 * (r2_14 + i2_14));
	       r1_6 = (r2_12 + tmpr);
	       i1_6 = (i2_12 - tmpi);
	       r1_14 = (r2_12 - tmpr);
	       i1_14 = (i2_12 + tmpi);
	  }
	  {
	       REAL r2_1, i2_1;
	       REAL r2_3, i2_3;
	       REAL r2_5, i2_5;
	       REAL r2_7, i2_7;
	       REAL r2_9, i2_9;
	       REAL r2_11, i2_11;
	       REAL r2_13, i2_13;
	       REAL r2_15, i2_15;
	       {
		    REAL r3_1, i3_1;
		    REAL r3_5, i3_5;
		    REAL r3_9, i3_9;
		    REAL r3_13, i3_13;
		    {
			 REAL r4_1, i4_1;
			 REAL r4_9, i4_9;
			 r4_1 = c_re(in[1]);
			 i4_1 = c_im(in[1]);
			 r4_9 = c_re(in[9]);
			 i4_9 = c_im(in[9]);
			 r3_1 = (r4_1 + r4_9);
			 i3_1 = (i4_1 + i4_9);
			 r3_9 = (r4_1 - r4_9);
			 i3_9 = (i4_1 - i4_9);
		    }
		    {
			 REAL r4_5, i4_5;
			 REAL r4_13, i4_13;
			 r4_5 = c_re(in[5]);
			 i4_5 = c_im(in[5]);
			 r4_13 = c_re(in[13]);
			 i4_13 = c_im(in[13]);
			 r3_5 = (r4_5 + r4_13);
			 i3_5 = (i4_5 + i4_13);
			 r3_13 = (r4_5 - r4_13);
			 i3_13 = (i4_5 - i4_13);
		    }
		    r2_1 = (r3_1 + r3_5);
		    i2_1 = (i3_1 + i3_5);
		    r2_9 = (r3_1 - r3_5);
		    i2_9 = (i3_1 - i3_5);
		    r2_5 = (r3_9 + i3_13);
		    i2_5 = (i3_9 - r3_13);
		    r2_13 = (r3_9 - i3_13);
		    i2_13 = (i3_9 + r3_13);
	       }
	       {
		    REAL r3_3, i3_3;
		    REAL r3_7, i3_7;
		    REAL r3_11, i3_11;
		    REAL r3_15, i3_15;
		    {
			 REAL r4_3, i4_3;
			 REAL r4_11, i4_11;
			 r4_3 = c_re(in[3]);
			 i4_3 = c_im(in[3]);
			 r4_11 = c_re(in[11]);
			 i4_11 = c_im(in[11]);
			 r3_3 = (r4_3 + r4_11);
			 i3_3 = (i4_3 + i4_11);
			 r3_11 = (r4_3 - r4_11);
			 i3_11 = (i4_3 - i4_11);
		    }
		    {
			 REAL r4_7, i4_7;
			 REAL r4_15, i4_15;
			 r4_7 = c_re(in[7]);
			 i4_7 = c_im(in[7]);
			 r4_15 = c_re(in[15]);
			 i4_15 = c_im(in[15]);
			 r3_7 = (r4_7 + r4_15);
			 i3_7 = (i4_7 + i4_15);
			 r3_15 = (r4_7 - r4_15);
			 i3_15 = (i4_7 - i4_15);
		    }
		    r2_3 = (r3_3 + r3_7);
		    i2_3 = (i3_3 + i3_7);
		    r2_11 = (r3_3 - r3_7);
		    i2_11 = (i3_3 - i3_7);
		    r2_7 = (r3_11 + i3_15);
		    i2_7 = (i3_11 - r3_15);
		    r2_15 = (r3_11 - i3_15);
		    i2_15 = (i3_11 + r3_15);
	       }
	       r1_1 = (r2_1 + r2_3);
	       i1_1 = (i2_1 + i2_3);
	       r1_9 = (r2_1 - r2_3);
	       i1_9 = (i2_1 - i2_3);
	       tmpr = (0.707106781187 * (r2_7 + i2_7));
	       tmpi = (0.707106781187 * (i2_7 - r2_7));
	       r1_3 = (r2_5 + tmpr);
	       i1_3 = (i2_5 + tmpi);
	       r1_11 = (r2_5 - tmpr);
	       i1_11 = (i2_5 - tmpi);
	       r1_5 = (r2_9 + i2_11);
	       i1_5 = (i2_9 - r2_11);
	       r1_13 = (r2_9 - i2_11);
	       i1_13 = (i2_9 + r2_11);
	       tmpr = (0.707106781187 * (i2_15 - r2_15));
	       tmpi = (0.707106781187 * (r2_15 + i2_15));
	       r1_7 = (r2_13 + tmpr);
	       i1_7 = (i2_13 - tmpi);
	       r1_15 = (r2_13 - tmpr);
	       i1_15 = (i2_13 + tmpi);
	  }
	  c_re(out[0]) = (r1_0 + r1_1);
	  c_im(out[0]) = (i1_0 + i1_1);
	  c_re(out[8]) = (r1_0 - r1_1);
	  c_im(out[8]) = (i1_0 - i1_1);
	  tmpr = ((0.923879532511 * r1_3) + (0.382683432365 * i1_3));
	  tmpi = ((0.923879532511 * i1_3) - (0.382683432365 * r1_3));
	  c_re(out[1]) = (r1_2 + tmpr);
	  c_im(out[1]) = (i1_2 + tmpi);
	  c_re(out[9]) = (r1_2 - tmpr);
	  c_im(out[9]) = (i1_2 - tmpi);
	  tmpr = (0.707106781187 * (r1_5 + i1_5));
	  tmpi = (0.707106781187 * (i1_5 - r1_5));
	  c_re(out[2]) = (r1_4 + tmpr);
	  c_im(out[2]) = (i1_4 + tmpi);
	  c_re(out[10]) = (r1_4 - tmpr);
	  c_im(out[10]) = (i1_4 - tmpi);
	  tmpr = ((0.382683432365 * r1_7) + (0.923879532511 * i1_7));
	  tmpi = ((0.382683432365 * i1_7) - (0.923879532511 * r1_7));
	  c_re(out[3]) = (r1_6 + tmpr);
	  c_im(out[3]) = (i1_6 + tmpi);
	  c_re(out[11]) = (r1_6 - tmpr);
	  c_im(out[11]) = (i1_6 - tmpi);
	  c_re(out[4]) = (r1_8 + i1_9);
	  c_im(out[4]) = (i1_8 - r1_9);
	  c_re(out[12]) = (r1_8 - i1_9);
	  c_im(out[12]) = (i1_8 + r1_9);
	  tmpr = ((0.923879532511 * i1_11) - (0.382683432365 * r1_11));
	  tmpi = ((0.923879532511 * r1_11) + (0.382683432365 * i1_11));
	  c_re(out[5]) = (r1_10 + tmpr);
	  c_im(out[5]) = (i1_10 - tmpi);
	  c_re(out[13]) = (r1_10 - tmpr);
	  c_im(out[13]) = (i1_10 + tmpi);
	  tmpr = (0.707106781187 * (i1_13 - r1_13));
	  tmpi = (0.707106781187 * (r1_13 + i1_13));
	  c_re(out[6]) = (r1_12 + tmpr);
	  c_im(out[6]) = (i1_12 - tmpi);
	  c_re(out[14]) = (r1_12 - tmpr);
	  c_im(out[14]) = (i1_12 + tmpi);
	  tmpr = ((0.382683432365 * i1_15) - (0.923879532511 * r1_15));
	  tmpi = ((0.382683432365 * r1_15) + (0.923879532511 * i1_15));
	  c_re(out[7]) = (r1_14 + tmpr);
	  c_im(out[7]) = (i1_14 - tmpi);
	  c_re(out[15]) = (r1_14 - tmpr);
	  c_im(out[15]) = (i1_14 + tmpi);
     }
}
typedef struct _pragma1687_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma1687_omp_task;

typedef struct _pragma1689_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma1689_omp_task;

static void pragma1687_omp_task_hclib_async(void *____arg);
static void pragma1689_omp_task_hclib_async(void *____arg);
void fft_twiddle_16(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    REAL r1_4, i1_4;
		    REAL r1_5, i1_5;
		    REAL r1_6, i1_6;
		    REAL r1_7, i1_7;
		    REAL r1_8, i1_8;
		    REAL r1_9, i1_9;
		    REAL r1_10, i1_10;
		    REAL r1_11, i1_11;
		    REAL r1_12, i1_12;
		    REAL r1_13, i1_13;
		    REAL r1_14, i1_14;
		    REAL r1_15, i1_15;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 REAL r2_4, i2_4;
			 REAL r2_6, i2_6;
			 REAL r2_8, i2_8;
			 REAL r2_10, i2_10;
			 REAL r2_12, i2_12;
			 REAL r2_14, i2_14;
			 {
			      REAL r3_0, i3_0;
			      REAL r3_4, i3_4;
			      REAL r3_8, i3_8;
			      REAL r3_12, i3_12;
			      {
				   REAL r4_0, i4_0;
				   REAL r4_8, i4_8;
				   r4_0 = c_re(jp[0 * m]);
				   i4_0 = c_im(jp[0 * m]);
				   wr = c_re(W[8 * l1]);
				   wi = c_im(W[8 * l1]);
				   tmpr = c_re(jp[8 * m]);
				   tmpi = c_im(jp[8 * m]);
				   r4_8 = ((wr * tmpr) - (wi * tmpi));
				   i4_8 = ((wi * tmpr) + (wr * tmpi));
				   r3_0 = (r4_0 + r4_8);
				   i3_0 = (i4_0 + i4_8);
				   r3_8 = (r4_0 - r4_8);
				   i3_8 = (i4_0 - i4_8);
			      }
			      {
				   REAL r4_4, i4_4;
				   REAL r4_12, i4_12;
				   wr = c_re(W[4 * l1]);
				   wi = c_im(W[4 * l1]);
				   tmpr = c_re(jp[4 * m]);
				   tmpi = c_im(jp[4 * m]);
				   r4_4 = ((wr * tmpr) - (wi * tmpi));
				   i4_4 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[12 * l1]);
				   wi = c_im(W[12 * l1]);
				   tmpr = c_re(jp[12 * m]);
				   tmpi = c_im(jp[12 * m]);
				   r4_12 = ((wr * tmpr) - (wi * tmpi));
				   i4_12 = ((wi * tmpr) + (wr * tmpi));
				   r3_4 = (r4_4 + r4_12);
				   i3_4 = (i4_4 + i4_12);
				   r3_12 = (r4_4 - r4_12);
				   i3_12 = (i4_4 - i4_12);
			      }
			      r2_0 = (r3_0 + r3_4);
			      i2_0 = (i3_0 + i3_4);
			      r2_8 = (r3_0 - r3_4);
			      i2_8 = (i3_0 - i3_4);
			      r2_4 = (r3_8 + i3_12);
			      i2_4 = (i3_8 - r3_12);
			      r2_12 = (r3_8 - i3_12);
			      i2_12 = (i3_8 + r3_12);
			 }
			 {
			      REAL r3_2, i3_2;
			      REAL r3_6, i3_6;
			      REAL r3_10, i3_10;
			      REAL r3_14, i3_14;
			      {
				   REAL r4_2, i4_2;
				   REAL r4_10, i4_10;
				   wr = c_re(W[2 * l1]);
				   wi = c_im(W[2 * l1]);
				   tmpr = c_re(jp[2 * m]);
				   tmpi = c_im(jp[2 * m]);
				   r4_2 = ((wr * tmpr) - (wi * tmpi));
				   i4_2 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[10 * l1]);
				   wi = c_im(W[10 * l1]);
				   tmpr = c_re(jp[10 * m]);
				   tmpi = c_im(jp[10 * m]);
				   r4_10 = ((wr * tmpr) - (wi * tmpi));
				   i4_10 = ((wi * tmpr) + (wr * tmpi));
				   r3_2 = (r4_2 + r4_10);
				   i3_2 = (i4_2 + i4_10);
				   r3_10 = (r4_2 - r4_10);
				   i3_10 = (i4_2 - i4_10);
			      }
			      {
				   REAL r4_6, i4_6;
				   REAL r4_14, i4_14;
				   wr = c_re(W[6 * l1]);
				   wi = c_im(W[6 * l1]);
				   tmpr = c_re(jp[6 * m]);
				   tmpi = c_im(jp[6 * m]);
				   r4_6 = ((wr * tmpr) - (wi * tmpi));
				   i4_6 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[14 * l1]);
				   wi = c_im(W[14 * l1]);
				   tmpr = c_re(jp[14 * m]);
				   tmpi = c_im(jp[14 * m]);
				   r4_14 = ((wr * tmpr) - (wi * tmpi));
				   i4_14 = ((wi * tmpr) + (wr * tmpi));
				   r3_6 = (r4_6 + r4_14);
				   i3_6 = (i4_6 + i4_14);
				   r3_14 = (r4_6 - r4_14);
				   i3_14 = (i4_6 - i4_14);
			      }
			      r2_2 = (r3_2 + r3_6);
			      i2_2 = (i3_2 + i3_6);
			      r2_10 = (r3_2 - r3_6);
			      i2_10 = (i3_2 - i3_6);
			      r2_6 = (r3_10 + i3_14);
			      i2_6 = (i3_10 - r3_14);
			      r2_14 = (r3_10 - i3_14);
			      i2_14 = (i3_10 + r3_14);
			 }
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_8 = (r2_0 - r2_2);
			 i1_8 = (i2_0 - i2_2);
			 tmpr = (0.707106781187 * (r2_6 + i2_6));
			 tmpi = (0.707106781187 * (i2_6 - r2_6));
			 r1_2 = (r2_4 + tmpr);
			 i1_2 = (i2_4 + tmpi);
			 r1_10 = (r2_4 - tmpr);
			 i1_10 = (i2_4 - tmpi);
			 r1_4 = (r2_8 + i2_10);
			 i1_4 = (i2_8 - r2_10);
			 r1_12 = (r2_8 - i2_10);
			 i1_12 = (i2_8 + r2_10);
			 tmpr = (0.707106781187 * (i2_14 - r2_14));
			 tmpi = (0.707106781187 * (r2_14 + i2_14));
			 r1_6 = (r2_12 + tmpr);
			 i1_6 = (i2_12 - tmpi);
			 r1_14 = (r2_12 - tmpr);
			 i1_14 = (i2_12 + tmpi);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 REAL r2_5, i2_5;
			 REAL r2_7, i2_7;
			 REAL r2_9, i2_9;
			 REAL r2_11, i2_11;
			 REAL r2_13, i2_13;
			 REAL r2_15, i2_15;
			 {
			      REAL r3_1, i3_1;
			      REAL r3_5, i3_5;
			      REAL r3_9, i3_9;
			      REAL r3_13, i3_13;
			      {
				   REAL r4_1, i4_1;
				   REAL r4_9, i4_9;
				   wr = c_re(W[1 * l1]);
				   wi = c_im(W[1 * l1]);
				   tmpr = c_re(jp[1 * m]);
				   tmpi = c_im(jp[1 * m]);
				   r4_1 = ((wr * tmpr) - (wi * tmpi));
				   i4_1 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[9 * l1]);
				   wi = c_im(W[9 * l1]);
				   tmpr = c_re(jp[9 * m]);
				   tmpi = c_im(jp[9 * m]);
				   r4_9 = ((wr * tmpr) - (wi * tmpi));
				   i4_9 = ((wi * tmpr) + (wr * tmpi));
				   r3_1 = (r4_1 + r4_9);
				   i3_1 = (i4_1 + i4_9);
				   r3_9 = (r4_1 - r4_9);
				   i3_9 = (i4_1 - i4_9);
			      }
			      {
				   REAL r4_5, i4_5;
				   REAL r4_13, i4_13;
				   wr = c_re(W[5 * l1]);
				   wi = c_im(W[5 * l1]);
				   tmpr = c_re(jp[5 * m]);
				   tmpi = c_im(jp[5 * m]);
				   r4_5 = ((wr * tmpr) - (wi * tmpi));
				   i4_5 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[13 * l1]);
				   wi = c_im(W[13 * l1]);
				   tmpr = c_re(jp[13 * m]);
				   tmpi = c_im(jp[13 * m]);
				   r4_13 = ((wr * tmpr) - (wi * tmpi));
				   i4_13 = ((wi * tmpr) + (wr * tmpi));
				   r3_5 = (r4_5 + r4_13);
				   i3_5 = (i4_5 + i4_13);
				   r3_13 = (r4_5 - r4_13);
				   i3_13 = (i4_5 - i4_13);
			      }
			      r2_1 = (r3_1 + r3_5);
			      i2_1 = (i3_1 + i3_5);
			      r2_9 = (r3_1 - r3_5);
			      i2_9 = (i3_1 - i3_5);
			      r2_5 = (r3_9 + i3_13);
			      i2_5 = (i3_9 - r3_13);
			      r2_13 = (r3_9 - i3_13);
			      i2_13 = (i3_9 + r3_13);
			 }
			 {
			      REAL r3_3, i3_3;
			      REAL r3_7, i3_7;
			      REAL r3_11, i3_11;
			      REAL r3_15, i3_15;
			      {
				   REAL r4_3, i4_3;
				   REAL r4_11, i4_11;
				   wr = c_re(W[3 * l1]);
				   wi = c_im(W[3 * l1]);
				   tmpr = c_re(jp[3 * m]);
				   tmpi = c_im(jp[3 * m]);
				   r4_3 = ((wr * tmpr) - (wi * tmpi));
				   i4_3 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[11 * l1]);
				   wi = c_im(W[11 * l1]);
				   tmpr = c_re(jp[11 * m]);
				   tmpi = c_im(jp[11 * m]);
				   r4_11 = ((wr * tmpr) - (wi * tmpi));
				   i4_11 = ((wi * tmpr) + (wr * tmpi));
				   r3_3 = (r4_3 + r4_11);
				   i3_3 = (i4_3 + i4_11);
				   r3_11 = (r4_3 - r4_11);
				   i3_11 = (i4_3 - i4_11);
			      }
			      {
				   REAL r4_7, i4_7;
				   REAL r4_15, i4_15;
				   wr = c_re(W[7 * l1]);
				   wi = c_im(W[7 * l1]);
				   tmpr = c_re(jp[7 * m]);
				   tmpi = c_im(jp[7 * m]);
				   r4_7 = ((wr * tmpr) - (wi * tmpi));
				   i4_7 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[15 * l1]);
				   wi = c_im(W[15 * l1]);
				   tmpr = c_re(jp[15 * m]);
				   tmpi = c_im(jp[15 * m]);
				   r4_15 = ((wr * tmpr) - (wi * tmpi));
				   i4_15 = ((wi * tmpr) + (wr * tmpi));
				   r3_7 = (r4_7 + r4_15);
				   i3_7 = (i4_7 + i4_15);
				   r3_15 = (r4_7 - r4_15);
				   i3_15 = (i4_7 - i4_15);
			      }
			      r2_3 = (r3_3 + r3_7);
			      i2_3 = (i3_3 + i3_7);
			      r2_11 = (r3_3 - r3_7);
			      i2_11 = (i3_3 - i3_7);
			      r2_7 = (r3_11 + i3_15);
			      i2_7 = (i3_11 - r3_15);
			      r2_15 = (r3_11 - i3_15);
			      i2_15 = (i3_11 + r3_15);
			 }
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_9 = (r2_1 - r2_3);
			 i1_9 = (i2_1 - i2_3);
			 tmpr = (0.707106781187 * (r2_7 + i2_7));
			 tmpi = (0.707106781187 * (i2_7 - r2_7));
			 r1_3 = (r2_5 + tmpr);
			 i1_3 = (i2_5 + tmpi);
			 r1_11 = (r2_5 - tmpr);
			 i1_11 = (i2_5 - tmpi);
			 r1_5 = (r2_9 + i2_11);
			 i1_5 = (i2_9 - r2_11);
			 r1_13 = (r2_9 - i2_11);
			 i1_13 = (i2_9 + r2_11);
			 tmpr = (0.707106781187 * (i2_15 - r2_15));
			 tmpi = (0.707106781187 * (r2_15 + i2_15));
			 r1_7 = (r2_13 + tmpr);
			 i1_7 = (i2_13 - tmpi);
			 r1_15 = (r2_13 - tmpr);
			 i1_15 = (i2_13 + tmpi);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[8 * m]) = (r1_0 - r1_1);
		    c_im(kp[8 * m]) = (i1_0 - i1_1);
		    tmpr = ((0.923879532511 * r1_3) + (0.382683432365 * i1_3));
		    tmpi = ((0.923879532511 * i1_3) - (0.382683432365 * r1_3));
		    c_re(kp[1 * m]) = (r1_2 + tmpr);
		    c_im(kp[1 * m]) = (i1_2 + tmpi);
		    c_re(kp[9 * m]) = (r1_2 - tmpr);
		    c_im(kp[9 * m]) = (i1_2 - tmpi);
		    tmpr = (0.707106781187 * (r1_5 + i1_5));
		    tmpi = (0.707106781187 * (i1_5 - r1_5));
		    c_re(kp[2 * m]) = (r1_4 + tmpr);
		    c_im(kp[2 * m]) = (i1_4 + tmpi);
		    c_re(kp[10 * m]) = (r1_4 - tmpr);
		    c_im(kp[10 * m]) = (i1_4 - tmpi);
		    tmpr = ((0.382683432365 * r1_7) + (0.923879532511 * i1_7));
		    tmpi = ((0.382683432365 * i1_7) - (0.923879532511 * r1_7));
		    c_re(kp[3 * m]) = (r1_6 + tmpr);
		    c_im(kp[3 * m]) = (i1_6 + tmpi);
		    c_re(kp[11 * m]) = (r1_6 - tmpr);
		    c_im(kp[11 * m]) = (i1_6 - tmpi);
		    c_re(kp[4 * m]) = (r1_8 + i1_9);
		    c_im(kp[4 * m]) = (i1_8 - r1_9);
		    c_re(kp[12 * m]) = (r1_8 - i1_9);
		    c_im(kp[12 * m]) = (i1_8 + r1_9);
		    tmpr = ((0.923879532511 * i1_11) - (0.382683432365 * r1_11));
		    tmpi = ((0.923879532511 * r1_11) + (0.382683432365 * i1_11));
		    c_re(kp[5 * m]) = (r1_10 + tmpr);
		    c_im(kp[5 * m]) = (i1_10 - tmpi);
		    c_re(kp[13 * m]) = (r1_10 - tmpr);
		    c_im(kp[13 * m]) = (i1_10 + tmpi);
		    tmpr = (0.707106781187 * (i1_13 - r1_13));
		    tmpi = (0.707106781187 * (r1_13 + i1_13));
		    c_re(kp[6 * m]) = (r1_12 + tmpr);
		    c_im(kp[6 * m]) = (i1_12 - tmpi);
		    c_re(kp[14 * m]) = (r1_12 - tmpr);
		    c_im(kp[14 * m]) = (i1_12 + tmpi);
		    tmpr = ((0.382683432365 * i1_15) - (0.923879532511 * r1_15));
		    tmpi = ((0.382683432365 * r1_15) + (0.923879532511 * i1_15));
		    c_re(kp[7 * m]) = (r1_14 + tmpr);
		    c_im(kp[7 * m]) = (i1_14 - tmpi);
		    c_re(kp[15 * m]) = (r1_14 - tmpr);
		    c_im(kp[15 * m]) = (i1_14 + tmpi);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma1687_omp_task *new_ctx = (pragma1687_omp_task *)malloc(sizeof(pragma1687_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma1687_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma1689_omp_task *new_ctx = (pragma1689_omp_task *)malloc(sizeof(pragma1689_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma1689_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma1687_omp_task_hclib_async(void *____arg) {
    pragma1687_omp_task *ctx = (pragma1687_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_16((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma1689_omp_task_hclib_async(void *____arg) {
    pragma1689_omp_task *ctx = (pragma1689_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_16((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_twiddle_16_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    REAL r1_4, i1_4;
		    REAL r1_5, i1_5;
		    REAL r1_6, i1_6;
		    REAL r1_7, i1_7;
		    REAL r1_8, i1_8;
		    REAL r1_9, i1_9;
		    REAL r1_10, i1_10;
		    REAL r1_11, i1_11;
		    REAL r1_12, i1_12;
		    REAL r1_13, i1_13;
		    REAL r1_14, i1_14;
		    REAL r1_15, i1_15;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 REAL r2_4, i2_4;
			 REAL r2_6, i2_6;
			 REAL r2_8, i2_8;
			 REAL r2_10, i2_10;
			 REAL r2_12, i2_12;
			 REAL r2_14, i2_14;
			 {
			      REAL r3_0, i3_0;
			      REAL r3_4, i3_4;
			      REAL r3_8, i3_8;
			      REAL r3_12, i3_12;
			      {
				   REAL r4_0, i4_0;
				   REAL r4_8, i4_8;
				   r4_0 = c_re(jp[0 * m]);
				   i4_0 = c_im(jp[0 * m]);
				   wr = c_re(W[8 * l1]);
				   wi = c_im(W[8 * l1]);
				   tmpr = c_re(jp[8 * m]);
				   tmpi = c_im(jp[8 * m]);
				   r4_8 = ((wr * tmpr) - (wi * tmpi));
				   i4_8 = ((wi * tmpr) + (wr * tmpi));
				   r3_0 = (r4_0 + r4_8);
				   i3_0 = (i4_0 + i4_8);
				   r3_8 = (r4_0 - r4_8);
				   i3_8 = (i4_0 - i4_8);
			      }
			      {
				   REAL r4_4, i4_4;
				   REAL r4_12, i4_12;
				   wr = c_re(W[4 * l1]);
				   wi = c_im(W[4 * l1]);
				   tmpr = c_re(jp[4 * m]);
				   tmpi = c_im(jp[4 * m]);
				   r4_4 = ((wr * tmpr) - (wi * tmpi));
				   i4_4 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[12 * l1]);
				   wi = c_im(W[12 * l1]);
				   tmpr = c_re(jp[12 * m]);
				   tmpi = c_im(jp[12 * m]);
				   r4_12 = ((wr * tmpr) - (wi * tmpi));
				   i4_12 = ((wi * tmpr) + (wr * tmpi));
				   r3_4 = (r4_4 + r4_12);
				   i3_4 = (i4_4 + i4_12);
				   r3_12 = (r4_4 - r4_12);
				   i3_12 = (i4_4 - i4_12);
			      }
			      r2_0 = (r3_0 + r3_4);
			      i2_0 = (i3_0 + i3_4);
			      r2_8 = (r3_0 - r3_4);
			      i2_8 = (i3_0 - i3_4);
			      r2_4 = (r3_8 + i3_12);
			      i2_4 = (i3_8 - r3_12);
			      r2_12 = (r3_8 - i3_12);
			      i2_12 = (i3_8 + r3_12);
			 }
			 {
			      REAL r3_2, i3_2;
			      REAL r3_6, i3_6;
			      REAL r3_10, i3_10;
			      REAL r3_14, i3_14;
			      {
				   REAL r4_2, i4_2;
				   REAL r4_10, i4_10;
				   wr = c_re(W[2 * l1]);
				   wi = c_im(W[2 * l1]);
				   tmpr = c_re(jp[2 * m]);
				   tmpi = c_im(jp[2 * m]);
				   r4_2 = ((wr * tmpr) - (wi * tmpi));
				   i4_2 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[10 * l1]);
				   wi = c_im(W[10 * l1]);
				   tmpr = c_re(jp[10 * m]);
				   tmpi = c_im(jp[10 * m]);
				   r4_10 = ((wr * tmpr) - (wi * tmpi));
				   i4_10 = ((wi * tmpr) + (wr * tmpi));
				   r3_2 = (r4_2 + r4_10);
				   i3_2 = (i4_2 + i4_10);
				   r3_10 = (r4_2 - r4_10);
				   i3_10 = (i4_2 - i4_10);
			      }
			      {
				   REAL r4_6, i4_6;
				   REAL r4_14, i4_14;
				   wr = c_re(W[6 * l1]);
				   wi = c_im(W[6 * l1]);
				   tmpr = c_re(jp[6 * m]);
				   tmpi = c_im(jp[6 * m]);
				   r4_6 = ((wr * tmpr) - (wi * tmpi));
				   i4_6 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[14 * l1]);
				   wi = c_im(W[14 * l1]);
				   tmpr = c_re(jp[14 * m]);
				   tmpi = c_im(jp[14 * m]);
				   r4_14 = ((wr * tmpr) - (wi * tmpi));
				   i4_14 = ((wi * tmpr) + (wr * tmpi));
				   r3_6 = (r4_6 + r4_14);
				   i3_6 = (i4_6 + i4_14);
				   r3_14 = (r4_6 - r4_14);
				   i3_14 = (i4_6 - i4_14);
			      }
			      r2_2 = (r3_2 + r3_6);
			      i2_2 = (i3_2 + i3_6);
			      r2_10 = (r3_2 - r3_6);
			      i2_10 = (i3_2 - i3_6);
			      r2_6 = (r3_10 + i3_14);
			      i2_6 = (i3_10 - r3_14);
			      r2_14 = (r3_10 - i3_14);
			      i2_14 = (i3_10 + r3_14);
			 }
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_8 = (r2_0 - r2_2);
			 i1_8 = (i2_0 - i2_2);
			 tmpr = (0.707106781187 * (r2_6 + i2_6));
			 tmpi = (0.707106781187 * (i2_6 - r2_6));
			 r1_2 = (r2_4 + tmpr);
			 i1_2 = (i2_4 + tmpi);
			 r1_10 = (r2_4 - tmpr);
			 i1_10 = (i2_4 - tmpi);
			 r1_4 = (r2_8 + i2_10);
			 i1_4 = (i2_8 - r2_10);
			 r1_12 = (r2_8 - i2_10);
			 i1_12 = (i2_8 + r2_10);
			 tmpr = (0.707106781187 * (i2_14 - r2_14));
			 tmpi = (0.707106781187 * (r2_14 + i2_14));
			 r1_6 = (r2_12 + tmpr);
			 i1_6 = (i2_12 - tmpi);
			 r1_14 = (r2_12 - tmpr);
			 i1_14 = (i2_12 + tmpi);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 REAL r2_5, i2_5;
			 REAL r2_7, i2_7;
			 REAL r2_9, i2_9;
			 REAL r2_11, i2_11;
			 REAL r2_13, i2_13;
			 REAL r2_15, i2_15;
			 {
			      REAL r3_1, i3_1;
			      REAL r3_5, i3_5;
			      REAL r3_9, i3_9;
			      REAL r3_13, i3_13;
			      {
				   REAL r4_1, i4_1;
				   REAL r4_9, i4_9;
				   wr = c_re(W[1 * l1]);
				   wi = c_im(W[1 * l1]);
				   tmpr = c_re(jp[1 * m]);
				   tmpi = c_im(jp[1 * m]);
				   r4_1 = ((wr * tmpr) - (wi * tmpi));
				   i4_1 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[9 * l1]);
				   wi = c_im(W[9 * l1]);
				   tmpr = c_re(jp[9 * m]);
				   tmpi = c_im(jp[9 * m]);
				   r4_9 = ((wr * tmpr) - (wi * tmpi));
				   i4_9 = ((wi * tmpr) + (wr * tmpi));
				   r3_1 = (r4_1 + r4_9);
				   i3_1 = (i4_1 + i4_9);
				   r3_9 = (r4_1 - r4_9);
				   i3_9 = (i4_1 - i4_9);
			      }
			      {
				   REAL r4_5, i4_5;
				   REAL r4_13, i4_13;
				   wr = c_re(W[5 * l1]);
				   wi = c_im(W[5 * l1]);
				   tmpr = c_re(jp[5 * m]);
				   tmpi = c_im(jp[5 * m]);
				   r4_5 = ((wr * tmpr) - (wi * tmpi));
				   i4_5 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[13 * l1]);
				   wi = c_im(W[13 * l1]);
				   tmpr = c_re(jp[13 * m]);
				   tmpi = c_im(jp[13 * m]);
				   r4_13 = ((wr * tmpr) - (wi * tmpi));
				   i4_13 = ((wi * tmpr) + (wr * tmpi));
				   r3_5 = (r4_5 + r4_13);
				   i3_5 = (i4_5 + i4_13);
				   r3_13 = (r4_5 - r4_13);
				   i3_13 = (i4_5 - i4_13);
			      }
			      r2_1 = (r3_1 + r3_5);
			      i2_1 = (i3_1 + i3_5);
			      r2_9 = (r3_1 - r3_5);
			      i2_9 = (i3_1 - i3_5);
			      r2_5 = (r3_9 + i3_13);
			      i2_5 = (i3_9 - r3_13);
			      r2_13 = (r3_9 - i3_13);
			      i2_13 = (i3_9 + r3_13);
			 }
			 {
			      REAL r3_3, i3_3;
			      REAL r3_7, i3_7;
			      REAL r3_11, i3_11;
			      REAL r3_15, i3_15;
			      {
				   REAL r4_3, i4_3;
				   REAL r4_11, i4_11;
				   wr = c_re(W[3 * l1]);
				   wi = c_im(W[3 * l1]);
				   tmpr = c_re(jp[3 * m]);
				   tmpi = c_im(jp[3 * m]);
				   r4_3 = ((wr * tmpr) - (wi * tmpi));
				   i4_3 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[11 * l1]);
				   wi = c_im(W[11 * l1]);
				   tmpr = c_re(jp[11 * m]);
				   tmpi = c_im(jp[11 * m]);
				   r4_11 = ((wr * tmpr) - (wi * tmpi));
				   i4_11 = ((wi * tmpr) + (wr * tmpi));
				   r3_3 = (r4_3 + r4_11);
				   i3_3 = (i4_3 + i4_11);
				   r3_11 = (r4_3 - r4_11);
				   i3_11 = (i4_3 - i4_11);
			      }
			      {
				   REAL r4_7, i4_7;
				   REAL r4_15, i4_15;
				   wr = c_re(W[7 * l1]);
				   wi = c_im(W[7 * l1]);
				   tmpr = c_re(jp[7 * m]);
				   tmpi = c_im(jp[7 * m]);
				   r4_7 = ((wr * tmpr) - (wi * tmpi));
				   i4_7 = ((wi * tmpr) + (wr * tmpi));
				   wr = c_re(W[15 * l1]);
				   wi = c_im(W[15 * l1]);
				   tmpr = c_re(jp[15 * m]);
				   tmpi = c_im(jp[15 * m]);
				   r4_15 = ((wr * tmpr) - (wi * tmpi));
				   i4_15 = ((wi * tmpr) + (wr * tmpi));
				   r3_7 = (r4_7 + r4_15);
				   i3_7 = (i4_7 + i4_15);
				   r3_15 = (r4_7 - r4_15);
				   i3_15 = (i4_7 - i4_15);
			      }
			      r2_3 = (r3_3 + r3_7);
			      i2_3 = (i3_3 + i3_7);
			      r2_11 = (r3_3 - r3_7);
			      i2_11 = (i3_3 - i3_7);
			      r2_7 = (r3_11 + i3_15);
			      i2_7 = (i3_11 - r3_15);
			      r2_15 = (r3_11 - i3_15);
			      i2_15 = (i3_11 + r3_15);
			 }
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_9 = (r2_1 - r2_3);
			 i1_9 = (i2_1 - i2_3);
			 tmpr = (0.707106781187 * (r2_7 + i2_7));
			 tmpi = (0.707106781187 * (i2_7 - r2_7));
			 r1_3 = (r2_5 + tmpr);
			 i1_3 = (i2_5 + tmpi);
			 r1_11 = (r2_5 - tmpr);
			 i1_11 = (i2_5 - tmpi);
			 r1_5 = (r2_9 + i2_11);
			 i1_5 = (i2_9 - r2_11);
			 r1_13 = (r2_9 - i2_11);
			 i1_13 = (i2_9 + r2_11);
			 tmpr = (0.707106781187 * (i2_15 - r2_15));
			 tmpi = (0.707106781187 * (r2_15 + i2_15));
			 r1_7 = (r2_13 + tmpr);
			 i1_7 = (i2_13 - tmpi);
			 r1_15 = (r2_13 - tmpr);
			 i1_15 = (i2_13 + tmpi);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[8 * m]) = (r1_0 - r1_1);
		    c_im(kp[8 * m]) = (i1_0 - i1_1);
		    tmpr = ((0.923879532511 * r1_3) + (0.382683432365 * i1_3));
		    tmpi = ((0.923879532511 * i1_3) - (0.382683432365 * r1_3));
		    c_re(kp[1 * m]) = (r1_2 + tmpr);
		    c_im(kp[1 * m]) = (i1_2 + tmpi);
		    c_re(kp[9 * m]) = (r1_2 - tmpr);
		    c_im(kp[9 * m]) = (i1_2 - tmpi);
		    tmpr = (0.707106781187 * (r1_5 + i1_5));
		    tmpi = (0.707106781187 * (i1_5 - r1_5));
		    c_re(kp[2 * m]) = (r1_4 + tmpr);
		    c_im(kp[2 * m]) = (i1_4 + tmpi);
		    c_re(kp[10 * m]) = (r1_4 - tmpr);
		    c_im(kp[10 * m]) = (i1_4 - tmpi);
		    tmpr = ((0.382683432365 * r1_7) + (0.923879532511 * i1_7));
		    tmpi = ((0.382683432365 * i1_7) - (0.923879532511 * r1_7));
		    c_re(kp[3 * m]) = (r1_6 + tmpr);
		    c_im(kp[3 * m]) = (i1_6 + tmpi);
		    c_re(kp[11 * m]) = (r1_6 - tmpr);
		    c_im(kp[11 * m]) = (i1_6 - tmpi);
		    c_re(kp[4 * m]) = (r1_8 + i1_9);
		    c_im(kp[4 * m]) = (i1_8 - r1_9);
		    c_re(kp[12 * m]) = (r1_8 - i1_9);
		    c_im(kp[12 * m]) = (i1_8 + r1_9);
		    tmpr = ((0.923879532511 * i1_11) - (0.382683432365 * r1_11));
		    tmpi = ((0.923879532511 * r1_11) + (0.382683432365 * i1_11));
		    c_re(kp[5 * m]) = (r1_10 + tmpr);
		    c_im(kp[5 * m]) = (i1_10 - tmpi);
		    c_re(kp[13 * m]) = (r1_10 - tmpr);
		    c_im(kp[13 * m]) = (i1_10 + tmpi);
		    tmpr = (0.707106781187 * (i1_13 - r1_13));
		    tmpi = (0.707106781187 * (r1_13 + i1_13));
		    c_re(kp[6 * m]) = (r1_12 + tmpr);
		    c_im(kp[6 * m]) = (i1_12 - tmpi);
		    c_re(kp[14 * m]) = (r1_12 - tmpr);
		    c_im(kp[14 * m]) = (i1_12 + tmpi);
		    tmpr = ((0.382683432365 * i1_15) - (0.923879532511 * r1_15));
		    tmpi = ((0.382683432365 * r1_15) + (0.923879532511 * i1_15));
		    c_re(kp[7 * m]) = (r1_14 + tmpr);
		    c_im(kp[7 * m]) = (i1_14 - tmpi);
		    c_re(kp[15 * m]) = (r1_14 - tmpr);
		    c_im(kp[15 * m]) = (i1_14 + tmpi);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_twiddle_16_seq(a, ab, in, out, W, nW, nWdn, m);
	  fft_twiddle_16_seq(ab, b, in, out, W, nW, nWdn, m);
     }
}
typedef struct _pragma2087_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma2087_omp_task;

typedef struct _pragma2089_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma2089_omp_task;

static void pragma2087_omp_task_hclib_async(void *____arg);
static void pragma2089_omp_task_hclib_async(void *____arg);
void fft_unshuffle_16(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 16;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma2087_omp_task *new_ctx = (pragma2087_omp_task *)malloc(sizeof(pragma2087_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma2087_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma2089_omp_task *new_ctx = (pragma2089_omp_task *)malloc(sizeof(pragma2089_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma2089_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma2087_omp_task_hclib_async(void *____arg) {
    pragma2087_omp_task *ctx = (pragma2087_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_16((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma2089_omp_task_hclib_async(void *____arg) {
    pragma2089_omp_task *ctx = (pragma2089_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_16((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_unshuffle_16_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 16;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_unshuffle_16_seq(a, ab, in, out, m);
	  fft_unshuffle_16_seq(ab, b, in, out, m);
     }
}
void fft_base_32(COMPLEX * in, COMPLEX * out)
{
     REAL tmpr, tmpi;
     {
	  REAL r1_0, i1_0;
	  REAL r1_1, i1_1;
	  REAL r1_2, i1_2;
	  REAL r1_3, i1_3;
	  REAL r1_4, i1_4;
	  REAL r1_5, i1_5;
	  REAL r1_6, i1_6;
	  REAL r1_7, i1_7;
	  REAL r1_8, i1_8;
	  REAL r1_9, i1_9;
	  REAL r1_10, i1_10;
	  REAL r1_11, i1_11;
	  REAL r1_12, i1_12;
	  REAL r1_13, i1_13;
	  REAL r1_14, i1_14;
	  REAL r1_15, i1_15;
	  REAL r1_16, i1_16;
	  REAL r1_17, i1_17;
	  REAL r1_18, i1_18;
	  REAL r1_19, i1_19;
	  REAL r1_20, i1_20;
	  REAL r1_21, i1_21;
	  REAL r1_22, i1_22;
	  REAL r1_23, i1_23;
	  REAL r1_24, i1_24;
	  REAL r1_25, i1_25;
	  REAL r1_26, i1_26;
	  REAL r1_27, i1_27;
	  REAL r1_28, i1_28;
	  REAL r1_29, i1_29;
	  REAL r1_30, i1_30;
	  REAL r1_31, i1_31;
	  {
	       REAL r2_0, i2_0;
	       REAL r2_2, i2_2;
	       REAL r2_4, i2_4;
	       REAL r2_6, i2_6;
	       REAL r2_8, i2_8;
	       REAL r2_10, i2_10;
	       REAL r2_12, i2_12;
	       REAL r2_14, i2_14;
	       REAL r2_16, i2_16;
	       REAL r2_18, i2_18;
	       REAL r2_20, i2_20;
	       REAL r2_22, i2_22;
	       REAL r2_24, i2_24;
	       REAL r2_26, i2_26;
	       REAL r2_28, i2_28;
	       REAL r2_30, i2_30;
	       {
		    REAL r3_0, i3_0;
		    REAL r3_4, i3_4;
		    REAL r3_8, i3_8;
		    REAL r3_12, i3_12;
		    REAL r3_16, i3_16;
		    REAL r3_20, i3_20;
		    REAL r3_24, i3_24;
		    REAL r3_28, i3_28;
		    {
			 REAL r4_0, i4_0;
			 REAL r4_8, i4_8;
			 REAL r4_16, i4_16;
			 REAL r4_24, i4_24;
			 {
			      REAL r5_0, i5_0;
			      REAL r5_16, i5_16;
			      r5_0 = c_re(in[0]);
			      i5_0 = c_im(in[0]);
			      r5_16 = c_re(in[16]);
			      i5_16 = c_im(in[16]);
			      r4_0 = (r5_0 + r5_16);
			      i4_0 = (i5_0 + i5_16);
			      r4_16 = (r5_0 - r5_16);
			      i4_16 = (i5_0 - i5_16);
			 }
			 {
			      REAL r5_8, i5_8;
			      REAL r5_24, i5_24;
			      r5_8 = c_re(in[8]);
			      i5_8 = c_im(in[8]);
			      r5_24 = c_re(in[24]);
			      i5_24 = c_im(in[24]);
			      r4_8 = (r5_8 + r5_24);
			      i4_8 = (i5_8 + i5_24);
			      r4_24 = (r5_8 - r5_24);
			      i4_24 = (i5_8 - i5_24);
			 }
			 r3_0 = (r4_0 + r4_8);
			 i3_0 = (i4_0 + i4_8);
			 r3_16 = (r4_0 - r4_8);
			 i3_16 = (i4_0 - i4_8);
			 r3_8 = (r4_16 + i4_24);
			 i3_8 = (i4_16 - r4_24);
			 r3_24 = (r4_16 - i4_24);
			 i3_24 = (i4_16 + r4_24);
		    }
		    {
			 REAL r4_4, i4_4;
			 REAL r4_12, i4_12;
			 REAL r4_20, i4_20;
			 REAL r4_28, i4_28;
			 {
			      REAL r5_4, i5_4;
			      REAL r5_20, i5_20;
			      r5_4 = c_re(in[4]);
			      i5_4 = c_im(in[4]);
			      r5_20 = c_re(in[20]);
			      i5_20 = c_im(in[20]);
			      r4_4 = (r5_4 + r5_20);
			      i4_4 = (i5_4 + i5_20);
			      r4_20 = (r5_4 - r5_20);
			      i4_20 = (i5_4 - i5_20);
			 }
			 {
			      REAL r5_12, i5_12;
			      REAL r5_28, i5_28;
			      r5_12 = c_re(in[12]);
			      i5_12 = c_im(in[12]);
			      r5_28 = c_re(in[28]);
			      i5_28 = c_im(in[28]);
			      r4_12 = (r5_12 + r5_28);
			      i4_12 = (i5_12 + i5_28);
			      r4_28 = (r5_12 - r5_28);
			      i4_28 = (i5_12 - i5_28);
			 }
			 r3_4 = (r4_4 + r4_12);
			 i3_4 = (i4_4 + i4_12);
			 r3_20 = (r4_4 - r4_12);
			 i3_20 = (i4_4 - i4_12);
			 r3_12 = (r4_20 + i4_28);
			 i3_12 = (i4_20 - r4_28);
			 r3_28 = (r4_20 - i4_28);
			 i3_28 = (i4_20 + r4_28);
		    }
		    r2_0 = (r3_0 + r3_4);
		    i2_0 = (i3_0 + i3_4);
		    r2_16 = (r3_0 - r3_4);
		    i2_16 = (i3_0 - i3_4);
		    tmpr = (0.707106781187 * (r3_12 + i3_12));
		    tmpi = (0.707106781187 * (i3_12 - r3_12));
		    r2_4 = (r3_8 + tmpr);
		    i2_4 = (i3_8 + tmpi);
		    r2_20 = (r3_8 - tmpr);
		    i2_20 = (i3_8 - tmpi);
		    r2_8 = (r3_16 + i3_20);
		    i2_8 = (i3_16 - r3_20);
		    r2_24 = (r3_16 - i3_20);
		    i2_24 = (i3_16 + r3_20);
		    tmpr = (0.707106781187 * (i3_28 - r3_28));
		    tmpi = (0.707106781187 * (r3_28 + i3_28));
		    r2_12 = (r3_24 + tmpr);
		    i2_12 = (i3_24 - tmpi);
		    r2_28 = (r3_24 - tmpr);
		    i2_28 = (i3_24 + tmpi);
	       }
	       {
		    REAL r3_2, i3_2;
		    REAL r3_6, i3_6;
		    REAL r3_10, i3_10;
		    REAL r3_14, i3_14;
		    REAL r3_18, i3_18;
		    REAL r3_22, i3_22;
		    REAL r3_26, i3_26;
		    REAL r3_30, i3_30;
		    {
			 REAL r4_2, i4_2;
			 REAL r4_10, i4_10;
			 REAL r4_18, i4_18;
			 REAL r4_26, i4_26;
			 {
			      REAL r5_2, i5_2;
			      REAL r5_18, i5_18;
			      r5_2 = c_re(in[2]);
			      i5_2 = c_im(in[2]);
			      r5_18 = c_re(in[18]);
			      i5_18 = c_im(in[18]);
			      r4_2 = (r5_2 + r5_18);
			      i4_2 = (i5_2 + i5_18);
			      r4_18 = (r5_2 - r5_18);
			      i4_18 = (i5_2 - i5_18);
			 }
			 {
			      REAL r5_10, i5_10;
			      REAL r5_26, i5_26;
			      r5_10 = c_re(in[10]);
			      i5_10 = c_im(in[10]);
			      r5_26 = c_re(in[26]);
			      i5_26 = c_im(in[26]);
			      r4_10 = (r5_10 + r5_26);
			      i4_10 = (i5_10 + i5_26);
			      r4_26 = (r5_10 - r5_26);
			      i4_26 = (i5_10 - i5_26);
			 }
			 r3_2 = (r4_2 + r4_10);
			 i3_2 = (i4_2 + i4_10);
			 r3_18 = (r4_2 - r4_10);
			 i3_18 = (i4_2 - i4_10);
			 r3_10 = (r4_18 + i4_26);
			 i3_10 = (i4_18 - r4_26);
			 r3_26 = (r4_18 - i4_26);
			 i3_26 = (i4_18 + r4_26);
		    }
		    {
			 REAL r4_6, i4_6;
			 REAL r4_14, i4_14;
			 REAL r4_22, i4_22;
			 REAL r4_30, i4_30;
			 {
			      REAL r5_6, i5_6;
			      REAL r5_22, i5_22;
			      r5_6 = c_re(in[6]);
			      i5_6 = c_im(in[6]);
			      r5_22 = c_re(in[22]);
			      i5_22 = c_im(in[22]);
			      r4_6 = (r5_6 + r5_22);
			      i4_6 = (i5_6 + i5_22);
			      r4_22 = (r5_6 - r5_22);
			      i4_22 = (i5_6 - i5_22);
			 }
			 {
			      REAL r5_14, i5_14;
			      REAL r5_30, i5_30;
			      r5_14 = c_re(in[14]);
			      i5_14 = c_im(in[14]);
			      r5_30 = c_re(in[30]);
			      i5_30 = c_im(in[30]);
			      r4_14 = (r5_14 + r5_30);
			      i4_14 = (i5_14 + i5_30);
			      r4_30 = (r5_14 - r5_30);
			      i4_30 = (i5_14 - i5_30);
			 }
			 r3_6 = (r4_6 + r4_14);
			 i3_6 = (i4_6 + i4_14);
			 r3_22 = (r4_6 - r4_14);
			 i3_22 = (i4_6 - i4_14);
			 r3_14 = (r4_22 + i4_30);
			 i3_14 = (i4_22 - r4_30);
			 r3_30 = (r4_22 - i4_30);
			 i3_30 = (i4_22 + r4_30);
		    }
		    r2_2 = (r3_2 + r3_6);
		    i2_2 = (i3_2 + i3_6);
		    r2_18 = (r3_2 - r3_6);
		    i2_18 = (i3_2 - i3_6);
		    tmpr = (0.707106781187 * (r3_14 + i3_14));
		    tmpi = (0.707106781187 * (i3_14 - r3_14));
		    r2_6 = (r3_10 + tmpr);
		    i2_6 = (i3_10 + tmpi);
		    r2_22 = (r3_10 - tmpr);
		    i2_22 = (i3_10 - tmpi);
		    r2_10 = (r3_18 + i3_22);
		    i2_10 = (i3_18 - r3_22);
		    r2_26 = (r3_18 - i3_22);
		    i2_26 = (i3_18 + r3_22);
		    tmpr = (0.707106781187 * (i3_30 - r3_30));
		    tmpi = (0.707106781187 * (r3_30 + i3_30));
		    r2_14 = (r3_26 + tmpr);
		    i2_14 = (i3_26 - tmpi);
		    r2_30 = (r3_26 - tmpr);
		    i2_30 = (i3_26 + tmpi);
	       }
	       r1_0 = (r2_0 + r2_2);
	       i1_0 = (i2_0 + i2_2);
	       r1_16 = (r2_0 - r2_2);
	       i1_16 = (i2_0 - i2_2);
	       tmpr = ((0.923879532511 * r2_6) + (0.382683432365 * i2_6));
	       tmpi = ((0.923879532511 * i2_6) - (0.382683432365 * r2_6));
	       r1_2 = (r2_4 + tmpr);
	       i1_2 = (i2_4 + tmpi);
	       r1_18 = (r2_4 - tmpr);
	       i1_18 = (i2_4 - tmpi);
	       tmpr = (0.707106781187 * (r2_10 + i2_10));
	       tmpi = (0.707106781187 * (i2_10 - r2_10));
	       r1_4 = (r2_8 + tmpr);
	       i1_4 = (i2_8 + tmpi);
	       r1_20 = (r2_8 - tmpr);
	       i1_20 = (i2_8 - tmpi);
	       tmpr = ((0.382683432365 * r2_14) + (0.923879532511 * i2_14));
	       tmpi = ((0.382683432365 * i2_14) - (0.923879532511 * r2_14));
	       r1_6 = (r2_12 + tmpr);
	       i1_6 = (i2_12 + tmpi);
	       r1_22 = (r2_12 - tmpr);
	       i1_22 = (i2_12 - tmpi);
	       r1_8 = (r2_16 + i2_18);
	       i1_8 = (i2_16 - r2_18);
	       r1_24 = (r2_16 - i2_18);
	       i1_24 = (i2_16 + r2_18);
	       tmpr = ((0.923879532511 * i2_22) - (0.382683432365 * r2_22));
	       tmpi = ((0.923879532511 * r2_22) + (0.382683432365 * i2_22));
	       r1_10 = (r2_20 + tmpr);
	       i1_10 = (i2_20 - tmpi);
	       r1_26 = (r2_20 - tmpr);
	       i1_26 = (i2_20 + tmpi);
	       tmpr = (0.707106781187 * (i2_26 - r2_26));
	       tmpi = (0.707106781187 * (r2_26 + i2_26));
	       r1_12 = (r2_24 + tmpr);
	       i1_12 = (i2_24 - tmpi);
	       r1_28 = (r2_24 - tmpr);
	       i1_28 = (i2_24 + tmpi);
	       tmpr = ((0.382683432365 * i2_30) - (0.923879532511 * r2_30));
	       tmpi = ((0.382683432365 * r2_30) + (0.923879532511 * i2_30));
	       r1_14 = (r2_28 + tmpr);
	       i1_14 = (i2_28 - tmpi);
	       r1_30 = (r2_28 - tmpr);
	       i1_30 = (i2_28 + tmpi);
	  }
	  {
	       REAL r2_1, i2_1;
	       REAL r2_3, i2_3;
	       REAL r2_5, i2_5;
	       REAL r2_7, i2_7;
	       REAL r2_9, i2_9;
	       REAL r2_11, i2_11;
	       REAL r2_13, i2_13;
	       REAL r2_15, i2_15;
	       REAL r2_17, i2_17;
	       REAL r2_19, i2_19;
	       REAL r2_21, i2_21;
	       REAL r2_23, i2_23;
	       REAL r2_25, i2_25;
	       REAL r2_27, i2_27;
	       REAL r2_29, i2_29;
	       REAL r2_31, i2_31;
	       {
		    REAL r3_1, i3_1;
		    REAL r3_5, i3_5;
		    REAL r3_9, i3_9;
		    REAL r3_13, i3_13;
		    REAL r3_17, i3_17;
		    REAL r3_21, i3_21;
		    REAL r3_25, i3_25;
		    REAL r3_29, i3_29;
		    {
			 REAL r4_1, i4_1;
			 REAL r4_9, i4_9;
			 REAL r4_17, i4_17;
			 REAL r4_25, i4_25;
			 {
			      REAL r5_1, i5_1;
			      REAL r5_17, i5_17;
			      r5_1 = c_re(in[1]);
			      i5_1 = c_im(in[1]);
			      r5_17 = c_re(in[17]);
			      i5_17 = c_im(in[17]);
			      r4_1 = (r5_1 + r5_17);
			      i4_1 = (i5_1 + i5_17);
			      r4_17 = (r5_1 - r5_17);
			      i4_17 = (i5_1 - i5_17);
			 }
			 {
			      REAL r5_9, i5_9;
			      REAL r5_25, i5_25;
			      r5_9 = c_re(in[9]);
			      i5_9 = c_im(in[9]);
			      r5_25 = c_re(in[25]);
			      i5_25 = c_im(in[25]);
			      r4_9 = (r5_9 + r5_25);
			      i4_9 = (i5_9 + i5_25);
			      r4_25 = (r5_9 - r5_25);
			      i4_25 = (i5_9 - i5_25);
			 }
			 r3_1 = (r4_1 + r4_9);
			 i3_1 = (i4_1 + i4_9);
			 r3_17 = (r4_1 - r4_9);
			 i3_17 = (i4_1 - i4_9);
			 r3_9 = (r4_17 + i4_25);
			 i3_9 = (i4_17 - r4_25);
			 r3_25 = (r4_17 - i4_25);
			 i3_25 = (i4_17 + r4_25);
		    }
		    {
			 REAL r4_5, i4_5;
			 REAL r4_13, i4_13;
			 REAL r4_21, i4_21;
			 REAL r4_29, i4_29;
			 {
			      REAL r5_5, i5_5;
			      REAL r5_21, i5_21;
			      r5_5 = c_re(in[5]);
			      i5_5 = c_im(in[5]);
			      r5_21 = c_re(in[21]);
			      i5_21 = c_im(in[21]);
			      r4_5 = (r5_5 + r5_21);
			      i4_5 = (i5_5 + i5_21);
			      r4_21 = (r5_5 - r5_21);
			      i4_21 = (i5_5 - i5_21);
			 }
			 {
			      REAL r5_13, i5_13;
			      REAL r5_29, i5_29;
			      r5_13 = c_re(in[13]);
			      i5_13 = c_im(in[13]);
			      r5_29 = c_re(in[29]);
			      i5_29 = c_im(in[29]);
			      r4_13 = (r5_13 + r5_29);
			      i4_13 = (i5_13 + i5_29);
			      r4_29 = (r5_13 - r5_29);
			      i4_29 = (i5_13 - i5_29);
			 }
			 r3_5 = (r4_5 + r4_13);
			 i3_5 = (i4_5 + i4_13);
			 r3_21 = (r4_5 - r4_13);
			 i3_21 = (i4_5 - i4_13);
			 r3_13 = (r4_21 + i4_29);
			 i3_13 = (i4_21 - r4_29);
			 r3_29 = (r4_21 - i4_29);
			 i3_29 = (i4_21 + r4_29);
		    }
		    r2_1 = (r3_1 + r3_5);
		    i2_1 = (i3_1 + i3_5);
		    r2_17 = (r3_1 - r3_5);
		    i2_17 = (i3_1 - i3_5);
		    tmpr = (0.707106781187 * (r3_13 + i3_13));
		    tmpi = (0.707106781187 * (i3_13 - r3_13));
		    r2_5 = (r3_9 + tmpr);
		    i2_5 = (i3_9 + tmpi);
		    r2_21 = (r3_9 - tmpr);
		    i2_21 = (i3_9 - tmpi);
		    r2_9 = (r3_17 + i3_21);
		    i2_9 = (i3_17 - r3_21);
		    r2_25 = (r3_17 - i3_21);
		    i2_25 = (i3_17 + r3_21);
		    tmpr = (0.707106781187 * (i3_29 - r3_29));
		    tmpi = (0.707106781187 * (r3_29 + i3_29));
		    r2_13 = (r3_25 + tmpr);
		    i2_13 = (i3_25 - tmpi);
		    r2_29 = (r3_25 - tmpr);
		    i2_29 = (i3_25 + tmpi);
	       }
	       {
		    REAL r3_3, i3_3;
		    REAL r3_7, i3_7;
		    REAL r3_11, i3_11;
		    REAL r3_15, i3_15;
		    REAL r3_19, i3_19;
		    REAL r3_23, i3_23;
		    REAL r3_27, i3_27;
		    REAL r3_31, i3_31;
		    {
			 REAL r4_3, i4_3;
			 REAL r4_11, i4_11;
			 REAL r4_19, i4_19;
			 REAL r4_27, i4_27;
			 {
			      REAL r5_3, i5_3;
			      REAL r5_19, i5_19;
			      r5_3 = c_re(in[3]);
			      i5_3 = c_im(in[3]);
			      r5_19 = c_re(in[19]);
			      i5_19 = c_im(in[19]);
			      r4_3 = (r5_3 + r5_19);
			      i4_3 = (i5_3 + i5_19);
			      r4_19 = (r5_3 - r5_19);
			      i4_19 = (i5_3 - i5_19);
			 }
			 {
			      REAL r5_11, i5_11;
			      REAL r5_27, i5_27;
			      r5_11 = c_re(in[11]);
			      i5_11 = c_im(in[11]);
			      r5_27 = c_re(in[27]);
			      i5_27 = c_im(in[27]);
			      r4_11 = (r5_11 + r5_27);
			      i4_11 = (i5_11 + i5_27);
			      r4_27 = (r5_11 - r5_27);
			      i4_27 = (i5_11 - i5_27);
			 }
			 r3_3 = (r4_3 + r4_11);
			 i3_3 = (i4_3 + i4_11);
			 r3_19 = (r4_3 - r4_11);
			 i3_19 = (i4_3 - i4_11);
			 r3_11 = (r4_19 + i4_27);
			 i3_11 = (i4_19 - r4_27);
			 r3_27 = (r4_19 - i4_27);
			 i3_27 = (i4_19 + r4_27);
		    }
		    {
			 REAL r4_7, i4_7;
			 REAL r4_15, i4_15;
			 REAL r4_23, i4_23;
			 REAL r4_31, i4_31;
			 {
			      REAL r5_7, i5_7;
			      REAL r5_23, i5_23;
			      r5_7 = c_re(in[7]);
			      i5_7 = c_im(in[7]);
			      r5_23 = c_re(in[23]);
			      i5_23 = c_im(in[23]);
			      r4_7 = (r5_7 + r5_23);
			      i4_7 = (i5_7 + i5_23);
			      r4_23 = (r5_7 - r5_23);
			      i4_23 = (i5_7 - i5_23);
			 }
			 {
			      REAL r5_15, i5_15;
			      REAL r5_31, i5_31;
			      r5_15 = c_re(in[15]);
			      i5_15 = c_im(in[15]);
			      r5_31 = c_re(in[31]);
			      i5_31 = c_im(in[31]);
			      r4_15 = (r5_15 + r5_31);
			      i4_15 = (i5_15 + i5_31);
			      r4_31 = (r5_15 - r5_31);
			      i4_31 = (i5_15 - i5_31);
			 }
			 r3_7 = (r4_7 + r4_15);
			 i3_7 = (i4_7 + i4_15);
			 r3_23 = (r4_7 - r4_15);
			 i3_23 = (i4_7 - i4_15);
			 r3_15 = (r4_23 + i4_31);
			 i3_15 = (i4_23 - r4_31);
			 r3_31 = (r4_23 - i4_31);
			 i3_31 = (i4_23 + r4_31);
		    }
		    r2_3 = (r3_3 + r3_7);
		    i2_3 = (i3_3 + i3_7);
		    r2_19 = (r3_3 - r3_7);
		    i2_19 = (i3_3 - i3_7);
		    tmpr = (0.707106781187 * (r3_15 + i3_15));
		    tmpi = (0.707106781187 * (i3_15 - r3_15));
		    r2_7 = (r3_11 + tmpr);
		    i2_7 = (i3_11 + tmpi);
		    r2_23 = (r3_11 - tmpr);
		    i2_23 = (i3_11 - tmpi);
		    r2_11 = (r3_19 + i3_23);
		    i2_11 = (i3_19 - r3_23);
		    r2_27 = (r3_19 - i3_23);
		    i2_27 = (i3_19 + r3_23);
		    tmpr = (0.707106781187 * (i3_31 - r3_31));
		    tmpi = (0.707106781187 * (r3_31 + i3_31));
		    r2_15 = (r3_27 + tmpr);
		    i2_15 = (i3_27 - tmpi);
		    r2_31 = (r3_27 - tmpr);
		    i2_31 = (i3_27 + tmpi);
	       }
	       r1_1 = (r2_1 + r2_3);
	       i1_1 = (i2_1 + i2_3);
	       r1_17 = (r2_1 - r2_3);
	       i1_17 = (i2_1 - i2_3);
	       tmpr = ((0.923879532511 * r2_7) + (0.382683432365 * i2_7));
	       tmpi = ((0.923879532511 * i2_7) - (0.382683432365 * r2_7));
	       r1_3 = (r2_5 + tmpr);
	       i1_3 = (i2_5 + tmpi);
	       r1_19 = (r2_5 - tmpr);
	       i1_19 = (i2_5 - tmpi);
	       tmpr = (0.707106781187 * (r2_11 + i2_11));
	       tmpi = (0.707106781187 * (i2_11 - r2_11));
	       r1_5 = (r2_9 + tmpr);
	       i1_5 = (i2_9 + tmpi);
	       r1_21 = (r2_9 - tmpr);
	       i1_21 = (i2_9 - tmpi);
	       tmpr = ((0.382683432365 * r2_15) + (0.923879532511 * i2_15));
	       tmpi = ((0.382683432365 * i2_15) - (0.923879532511 * r2_15));
	       r1_7 = (r2_13 + tmpr);
	       i1_7 = (i2_13 + tmpi);
	       r1_23 = (r2_13 - tmpr);
	       i1_23 = (i2_13 - tmpi);
	       r1_9 = (r2_17 + i2_19);
	       i1_9 = (i2_17 - r2_19);
	       r1_25 = (r2_17 - i2_19);
	       i1_25 = (i2_17 + r2_19);
	       tmpr = ((0.923879532511 * i2_23) - (0.382683432365 * r2_23));
	       tmpi = ((0.923879532511 * r2_23) + (0.382683432365 * i2_23));
	       r1_11 = (r2_21 + tmpr);
	       i1_11 = (i2_21 - tmpi);
	       r1_27 = (r2_21 - tmpr);
	       i1_27 = (i2_21 + tmpi);
	       tmpr = (0.707106781187 * (i2_27 - r2_27));
	       tmpi = (0.707106781187 * (r2_27 + i2_27));
	       r1_13 = (r2_25 + tmpr);
	       i1_13 = (i2_25 - tmpi);
	       r1_29 = (r2_25 - tmpr);
	       i1_29 = (i2_25 + tmpi);
	       tmpr = ((0.382683432365 * i2_31) - (0.923879532511 * r2_31));
	       tmpi = ((0.382683432365 * r2_31) + (0.923879532511 * i2_31));
	       r1_15 = (r2_29 + tmpr);
	       i1_15 = (i2_29 - tmpi);
	       r1_31 = (r2_29 - tmpr);
	       i1_31 = (i2_29 + tmpi);
	  }
	  c_re(out[0]) = (r1_0 + r1_1);
	  c_im(out[0]) = (i1_0 + i1_1);
	  c_re(out[16]) = (r1_0 - r1_1);
	  c_im(out[16]) = (i1_0 - i1_1);
	  tmpr = ((0.980785280403 * r1_3) + (0.195090322016 * i1_3));
	  tmpi = ((0.980785280403 * i1_3) - (0.195090322016 * r1_3));
	  c_re(out[1]) = (r1_2 + tmpr);
	  c_im(out[1]) = (i1_2 + tmpi);
	  c_re(out[17]) = (r1_2 - tmpr);
	  c_im(out[17]) = (i1_2 - tmpi);
	  tmpr = ((0.923879532511 * r1_5) + (0.382683432365 * i1_5));
	  tmpi = ((0.923879532511 * i1_5) - (0.382683432365 * r1_5));
	  c_re(out[2]) = (r1_4 + tmpr);
	  c_im(out[2]) = (i1_4 + tmpi);
	  c_re(out[18]) = (r1_4 - tmpr);
	  c_im(out[18]) = (i1_4 - tmpi);
	  tmpr = ((0.831469612303 * r1_7) + (0.55557023302 * i1_7));
	  tmpi = ((0.831469612303 * i1_7) - (0.55557023302 * r1_7));
	  c_re(out[3]) = (r1_6 + tmpr);
	  c_im(out[3]) = (i1_6 + tmpi);
	  c_re(out[19]) = (r1_6 - tmpr);
	  c_im(out[19]) = (i1_6 - tmpi);
	  tmpr = (0.707106781187 * (r1_9 + i1_9));
	  tmpi = (0.707106781187 * (i1_9 - r1_9));
	  c_re(out[4]) = (r1_8 + tmpr);
	  c_im(out[4]) = (i1_8 + tmpi);
	  c_re(out[20]) = (r1_8 - tmpr);
	  c_im(out[20]) = (i1_8 - tmpi);
	  tmpr = ((0.55557023302 * r1_11) + (0.831469612303 * i1_11));
	  tmpi = ((0.55557023302 * i1_11) - (0.831469612303 * r1_11));
	  c_re(out[5]) = (r1_10 + tmpr);
	  c_im(out[5]) = (i1_10 + tmpi);
	  c_re(out[21]) = (r1_10 - tmpr);
	  c_im(out[21]) = (i1_10 - tmpi);
	  tmpr = ((0.382683432365 * r1_13) + (0.923879532511 * i1_13));
	  tmpi = ((0.382683432365 * i1_13) - (0.923879532511 * r1_13));
	  c_re(out[6]) = (r1_12 + tmpr);
	  c_im(out[6]) = (i1_12 + tmpi);
	  c_re(out[22]) = (r1_12 - tmpr);
	  c_im(out[22]) = (i1_12 - tmpi);
	  tmpr = ((0.195090322016 * r1_15) + (0.980785280403 * i1_15));
	  tmpi = ((0.195090322016 * i1_15) - (0.980785280403 * r1_15));
	  c_re(out[7]) = (r1_14 + tmpr);
	  c_im(out[7]) = (i1_14 + tmpi);
	  c_re(out[23]) = (r1_14 - tmpr);
	  c_im(out[23]) = (i1_14 - tmpi);
	  c_re(out[8]) = (r1_16 + i1_17);
	  c_im(out[8]) = (i1_16 - r1_17);
	  c_re(out[24]) = (r1_16 - i1_17);
	  c_im(out[24]) = (i1_16 + r1_17);
	  tmpr = ((0.980785280403 * i1_19) - (0.195090322016 * r1_19));
	  tmpi = ((0.980785280403 * r1_19) + (0.195090322016 * i1_19));
	  c_re(out[9]) = (r1_18 + tmpr);
	  c_im(out[9]) = (i1_18 - tmpi);
	  c_re(out[25]) = (r1_18 - tmpr);
	  c_im(out[25]) = (i1_18 + tmpi);
	  tmpr = ((0.923879532511 * i1_21) - (0.382683432365 * r1_21));
	  tmpi = ((0.923879532511 * r1_21) + (0.382683432365 * i1_21));
	  c_re(out[10]) = (r1_20 + tmpr);
	  c_im(out[10]) = (i1_20 - tmpi);
	  c_re(out[26]) = (r1_20 - tmpr);
	  c_im(out[26]) = (i1_20 + tmpi);
	  tmpr = ((0.831469612303 * i1_23) - (0.55557023302 * r1_23));
	  tmpi = ((0.831469612303 * r1_23) + (0.55557023302 * i1_23));
	  c_re(out[11]) = (r1_22 + tmpr);
	  c_im(out[11]) = (i1_22 - tmpi);
	  c_re(out[27]) = (r1_22 - tmpr);
	  c_im(out[27]) = (i1_22 + tmpi);
	  tmpr = (0.707106781187 * (i1_25 - r1_25));
	  tmpi = (0.707106781187 * (r1_25 + i1_25));
	  c_re(out[12]) = (r1_24 + tmpr);
	  c_im(out[12]) = (i1_24 - tmpi);
	  c_re(out[28]) = (r1_24 - tmpr);
	  c_im(out[28]) = (i1_24 + tmpi);
	  tmpr = ((0.55557023302 * i1_27) - (0.831469612303 * r1_27));
	  tmpi = ((0.55557023302 * r1_27) + (0.831469612303 * i1_27));
	  c_re(out[13]) = (r1_26 + tmpr);
	  c_im(out[13]) = (i1_26 - tmpi);
	  c_re(out[29]) = (r1_26 - tmpr);
	  c_im(out[29]) = (i1_26 + tmpi);
	  tmpr = ((0.382683432365 * i1_29) - (0.923879532511 * r1_29));
	  tmpi = ((0.382683432365 * r1_29) + (0.923879532511 * i1_29));
	  c_re(out[14]) = (r1_28 + tmpr);
	  c_im(out[14]) = (i1_28 - tmpi);
	  c_re(out[30]) = (r1_28 - tmpr);
	  c_im(out[30]) = (i1_28 + tmpi);
	  tmpr = ((0.195090322016 * i1_31) - (0.980785280403 * r1_31));
	  tmpi = ((0.195090322016 * r1_31) + (0.980785280403 * i1_31));
	  c_re(out[15]) = (r1_30 + tmpr);
	  c_im(out[15]) = (i1_30 - tmpi);
	  c_re(out[31]) = (r1_30 - tmpr);
	  c_im(out[31]) = (i1_30 + tmpi);
     }
}
typedef struct _pragma3629_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma3629_omp_task;

typedef struct _pragma3631_omp_task {
    int (*ab_ptr);
    int (*l1_ptr);
    int (*i_ptr);
    COMPLEX (*(*jp_ptr));
    COMPLEX (*(*kp_ptr));
    double (*tmpr_ptr);
    double (*tmpi_ptr);
    double (*wr_ptr);
    double (*wi_ptr);
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
    int (*nWdn_ptr);
    int (*m_ptr);
 } pragma3631_omp_task;

static void pragma3629_omp_task_hclib_async(void *____arg);
static void pragma3631_omp_task_hclib_async(void *____arg);
void fft_twiddle_32(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    REAL r1_4, i1_4;
		    REAL r1_5, i1_5;
		    REAL r1_6, i1_6;
		    REAL r1_7, i1_7;
		    REAL r1_8, i1_8;
		    REAL r1_9, i1_9;
		    REAL r1_10, i1_10;
		    REAL r1_11, i1_11;
		    REAL r1_12, i1_12;
		    REAL r1_13, i1_13;
		    REAL r1_14, i1_14;
		    REAL r1_15, i1_15;
		    REAL r1_16, i1_16;
		    REAL r1_17, i1_17;
		    REAL r1_18, i1_18;
		    REAL r1_19, i1_19;
		    REAL r1_20, i1_20;
		    REAL r1_21, i1_21;
		    REAL r1_22, i1_22;
		    REAL r1_23, i1_23;
		    REAL r1_24, i1_24;
		    REAL r1_25, i1_25;
		    REAL r1_26, i1_26;
		    REAL r1_27, i1_27;
		    REAL r1_28, i1_28;
		    REAL r1_29, i1_29;
		    REAL r1_30, i1_30;
		    REAL r1_31, i1_31;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 REAL r2_4, i2_4;
			 REAL r2_6, i2_6;
			 REAL r2_8, i2_8;
			 REAL r2_10, i2_10;
			 REAL r2_12, i2_12;
			 REAL r2_14, i2_14;
			 REAL r2_16, i2_16;
			 REAL r2_18, i2_18;
			 REAL r2_20, i2_20;
			 REAL r2_22, i2_22;
			 REAL r2_24, i2_24;
			 REAL r2_26, i2_26;
			 REAL r2_28, i2_28;
			 REAL r2_30, i2_30;
			 {
			      REAL r3_0, i3_0;
			      REAL r3_4, i3_4;
			      REAL r3_8, i3_8;
			      REAL r3_12, i3_12;
			      REAL r3_16, i3_16;
			      REAL r3_20, i3_20;
			      REAL r3_24, i3_24;
			      REAL r3_28, i3_28;
			      {
				   REAL r4_0, i4_0;
				   REAL r4_8, i4_8;
				   REAL r4_16, i4_16;
				   REAL r4_24, i4_24;
				   {
					REAL r5_0, i5_0;
					REAL r5_16, i5_16;
					r5_0 = c_re(jp[0 * m]);
					i5_0 = c_im(jp[0 * m]);
					wr = c_re(W[16 * l1]);
					wi = c_im(W[16 * l1]);
					tmpr = c_re(jp[16 * m]);
					tmpi = c_im(jp[16 * m]);
					r5_16 = ((wr * tmpr) - (wi * tmpi));
					i5_16 = ((wi * tmpr) + (wr * tmpi));
					r4_0 = (r5_0 + r5_16);
					i4_0 = (i5_0 + i5_16);
					r4_16 = (r5_0 - r5_16);
					i4_16 = (i5_0 - i5_16);
				   }
				   {
					REAL r5_8, i5_8;
					REAL r5_24, i5_24;
					wr = c_re(W[8 * l1]);
					wi = c_im(W[8 * l1]);
					tmpr = c_re(jp[8 * m]);
					tmpi = c_im(jp[8 * m]);
					r5_8 = ((wr * tmpr) - (wi * tmpi));
					i5_8 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[24 * l1]);
					wi = c_im(W[24 * l1]);
					tmpr = c_re(jp[24 * m]);
					tmpi = c_im(jp[24 * m]);
					r5_24 = ((wr * tmpr) - (wi * tmpi));
					i5_24 = ((wi * tmpr) + (wr * tmpi));
					r4_8 = (r5_8 + r5_24);
					i4_8 = (i5_8 + i5_24);
					r4_24 = (r5_8 - r5_24);
					i4_24 = (i5_8 - i5_24);
				   }
				   r3_0 = (r4_0 + r4_8);
				   i3_0 = (i4_0 + i4_8);
				   r3_16 = (r4_0 - r4_8);
				   i3_16 = (i4_0 - i4_8);
				   r3_8 = (r4_16 + i4_24);
				   i3_8 = (i4_16 - r4_24);
				   r3_24 = (r4_16 - i4_24);
				   i3_24 = (i4_16 + r4_24);
			      }
			      {
				   REAL r4_4, i4_4;
				   REAL r4_12, i4_12;
				   REAL r4_20, i4_20;
				   REAL r4_28, i4_28;
				   {
					REAL r5_4, i5_4;
					REAL r5_20, i5_20;
					wr = c_re(W[4 * l1]);
					wi = c_im(W[4 * l1]);
					tmpr = c_re(jp[4 * m]);
					tmpi = c_im(jp[4 * m]);
					r5_4 = ((wr * tmpr) - (wi * tmpi));
					i5_4 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[20 * l1]);
					wi = c_im(W[20 * l1]);
					tmpr = c_re(jp[20 * m]);
					tmpi = c_im(jp[20 * m]);
					r5_20 = ((wr * tmpr) - (wi * tmpi));
					i5_20 = ((wi * tmpr) + (wr * tmpi));
					r4_4 = (r5_4 + r5_20);
					i4_4 = (i5_4 + i5_20);
					r4_20 = (r5_4 - r5_20);
					i4_20 = (i5_4 - i5_20);
				   }
				   {
					REAL r5_12, i5_12;
					REAL r5_28, i5_28;
					wr = c_re(W[12 * l1]);
					wi = c_im(W[12 * l1]);
					tmpr = c_re(jp[12 * m]);
					tmpi = c_im(jp[12 * m]);
					r5_12 = ((wr * tmpr) - (wi * tmpi));
					i5_12 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[28 * l1]);
					wi = c_im(W[28 * l1]);
					tmpr = c_re(jp[28 * m]);
					tmpi = c_im(jp[28 * m]);
					r5_28 = ((wr * tmpr) - (wi * tmpi));
					i5_28 = ((wi * tmpr) + (wr * tmpi));
					r4_12 = (r5_12 + r5_28);
					i4_12 = (i5_12 + i5_28);
					r4_28 = (r5_12 - r5_28);
					i4_28 = (i5_12 - i5_28);
				   }
				   r3_4 = (r4_4 + r4_12);
				   i3_4 = (i4_4 + i4_12);
				   r3_20 = (r4_4 - r4_12);
				   i3_20 = (i4_4 - i4_12);
				   r3_12 = (r4_20 + i4_28);
				   i3_12 = (i4_20 - r4_28);
				   r3_28 = (r4_20 - i4_28);
				   i3_28 = (i4_20 + r4_28);
			      }
			      r2_0 = (r3_0 + r3_4);
			      i2_0 = (i3_0 + i3_4);
			      r2_16 = (r3_0 - r3_4);
			      i2_16 = (i3_0 - i3_4);
			      tmpr = (0.707106781187 * (r3_12 + i3_12));
			      tmpi = (0.707106781187 * (i3_12 - r3_12));
			      r2_4 = (r3_8 + tmpr);
			      i2_4 = (i3_8 + tmpi);
			      r2_20 = (r3_8 - tmpr);
			      i2_20 = (i3_8 - tmpi);
			      r2_8 = (r3_16 + i3_20);
			      i2_8 = (i3_16 - r3_20);
			      r2_24 = (r3_16 - i3_20);
			      i2_24 = (i3_16 + r3_20);
			      tmpr = (0.707106781187 * (i3_28 - r3_28));
			      tmpi = (0.707106781187 * (r3_28 + i3_28));
			      r2_12 = (r3_24 + tmpr);
			      i2_12 = (i3_24 - tmpi);
			      r2_28 = (r3_24 - tmpr);
			      i2_28 = (i3_24 + tmpi);
			 }
			 {
			      REAL r3_2, i3_2;
			      REAL r3_6, i3_6;
			      REAL r3_10, i3_10;
			      REAL r3_14, i3_14;
			      REAL r3_18, i3_18;
			      REAL r3_22, i3_22;
			      REAL r3_26, i3_26;
			      REAL r3_30, i3_30;
			      {
				   REAL r4_2, i4_2;
				   REAL r4_10, i4_10;
				   REAL r4_18, i4_18;
				   REAL r4_26, i4_26;
				   {
					REAL r5_2, i5_2;
					REAL r5_18, i5_18;
					wr = c_re(W[2 * l1]);
					wi = c_im(W[2 * l1]);
					tmpr = c_re(jp[2 * m]);
					tmpi = c_im(jp[2 * m]);
					r5_2 = ((wr * tmpr) - (wi * tmpi));
					i5_2 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[18 * l1]);
					wi = c_im(W[18 * l1]);
					tmpr = c_re(jp[18 * m]);
					tmpi = c_im(jp[18 * m]);
					r5_18 = ((wr * tmpr) - (wi * tmpi));
					i5_18 = ((wi * tmpr) + (wr * tmpi));
					r4_2 = (r5_2 + r5_18);
					i4_2 = (i5_2 + i5_18);
					r4_18 = (r5_2 - r5_18);
					i4_18 = (i5_2 - i5_18);
				   }
				   {
					REAL r5_10, i5_10;
					REAL r5_26, i5_26;
					wr = c_re(W[10 * l1]);
					wi = c_im(W[10 * l1]);
					tmpr = c_re(jp[10 * m]);
					tmpi = c_im(jp[10 * m]);
					r5_10 = ((wr * tmpr) - (wi * tmpi));
					i5_10 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[26 * l1]);
					wi = c_im(W[26 * l1]);
					tmpr = c_re(jp[26 * m]);
					tmpi = c_im(jp[26 * m]);
					r5_26 = ((wr * tmpr) - (wi * tmpi));
					i5_26 = ((wi * tmpr) + (wr * tmpi));
					r4_10 = (r5_10 + r5_26);
					i4_10 = (i5_10 + i5_26);
					r4_26 = (r5_10 - r5_26);
					i4_26 = (i5_10 - i5_26);
				   }
				   r3_2 = (r4_2 + r4_10);
				   i3_2 = (i4_2 + i4_10);
				   r3_18 = (r4_2 - r4_10);
				   i3_18 = (i4_2 - i4_10);
				   r3_10 = (r4_18 + i4_26);
				   i3_10 = (i4_18 - r4_26);
				   r3_26 = (r4_18 - i4_26);
				   i3_26 = (i4_18 + r4_26);
			      }
			      {
				   REAL r4_6, i4_6;
				   REAL r4_14, i4_14;
				   REAL r4_22, i4_22;
				   REAL r4_30, i4_30;
				   {
					REAL r5_6, i5_6;
					REAL r5_22, i5_22;
					wr = c_re(W[6 * l1]);
					wi = c_im(W[6 * l1]);
					tmpr = c_re(jp[6 * m]);
					tmpi = c_im(jp[6 * m]);
					r5_6 = ((wr * tmpr) - (wi * tmpi));
					i5_6 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[22 * l1]);
					wi = c_im(W[22 * l1]);
					tmpr = c_re(jp[22 * m]);
					tmpi = c_im(jp[22 * m]);
					r5_22 = ((wr * tmpr) - (wi * tmpi));
					i5_22 = ((wi * tmpr) + (wr * tmpi));
					r4_6 = (r5_6 + r5_22);
					i4_6 = (i5_6 + i5_22);
					r4_22 = (r5_6 - r5_22);
					i4_22 = (i5_6 - i5_22);
				   }
				   {
					REAL r5_14, i5_14;
					REAL r5_30, i5_30;
					wr = c_re(W[14 * l1]);
					wi = c_im(W[14 * l1]);
					tmpr = c_re(jp[14 * m]);
					tmpi = c_im(jp[14 * m]);
					r5_14 = ((wr * tmpr) - (wi * tmpi));
					i5_14 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[30 * l1]);
					wi = c_im(W[30 * l1]);
					tmpr = c_re(jp[30 * m]);
					tmpi = c_im(jp[30 * m]);
					r5_30 = ((wr * tmpr) - (wi * tmpi));
					i5_30 = ((wi * tmpr) + (wr * tmpi));
					r4_14 = (r5_14 + r5_30);
					i4_14 = (i5_14 + i5_30);
					r4_30 = (r5_14 - r5_30);
					i4_30 = (i5_14 - i5_30);
				   }
				   r3_6 = (r4_6 + r4_14);
				   i3_6 = (i4_6 + i4_14);
				   r3_22 = (r4_6 - r4_14);
				   i3_22 = (i4_6 - i4_14);
				   r3_14 = (r4_22 + i4_30);
				   i3_14 = (i4_22 - r4_30);
				   r3_30 = (r4_22 - i4_30);
				   i3_30 = (i4_22 + r4_30);
			      }
			      r2_2 = (r3_2 + r3_6);
			      i2_2 = (i3_2 + i3_6);
			      r2_18 = (r3_2 - r3_6);
			      i2_18 = (i3_2 - i3_6);
			      tmpr = (0.707106781187 * (r3_14 + i3_14));
			      tmpi = (0.707106781187 * (i3_14 - r3_14));
			      r2_6 = (r3_10 + tmpr);
			      i2_6 = (i3_10 + tmpi);
			      r2_22 = (r3_10 - tmpr);
			      i2_22 = (i3_10 - tmpi);
			      r2_10 = (r3_18 + i3_22);
			      i2_10 = (i3_18 - r3_22);
			      r2_26 = (r3_18 - i3_22);
			      i2_26 = (i3_18 + r3_22);
			      tmpr = (0.707106781187 * (i3_30 - r3_30));
			      tmpi = (0.707106781187 * (r3_30 + i3_30));
			      r2_14 = (r3_26 + tmpr);
			      i2_14 = (i3_26 - tmpi);
			      r2_30 = (r3_26 - tmpr);
			      i2_30 = (i3_26 + tmpi);
			 }
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_16 = (r2_0 - r2_2);
			 i1_16 = (i2_0 - i2_2);
			 tmpr = ((0.923879532511 * r2_6) + (0.382683432365 * i2_6));
			 tmpi = ((0.923879532511 * i2_6) - (0.382683432365 * r2_6));
			 r1_2 = (r2_4 + tmpr);
			 i1_2 = (i2_4 + tmpi);
			 r1_18 = (r2_4 - tmpr);
			 i1_18 = (i2_4 - tmpi);
			 tmpr = (0.707106781187 * (r2_10 + i2_10));
			 tmpi = (0.707106781187 * (i2_10 - r2_10));
			 r1_4 = (r2_8 + tmpr);
			 i1_4 = (i2_8 + tmpi);
			 r1_20 = (r2_8 - tmpr);
			 i1_20 = (i2_8 - tmpi);
			 tmpr = ((0.382683432365 * r2_14) + (0.923879532511 * i2_14));
			 tmpi = ((0.382683432365 * i2_14) - (0.923879532511 * r2_14));
			 r1_6 = (r2_12 + tmpr);
			 i1_6 = (i2_12 + tmpi);
			 r1_22 = (r2_12 - tmpr);
			 i1_22 = (i2_12 - tmpi);
			 r1_8 = (r2_16 + i2_18);
			 i1_8 = (i2_16 - r2_18);
			 r1_24 = (r2_16 - i2_18);
			 i1_24 = (i2_16 + r2_18);
			 tmpr = ((0.923879532511 * i2_22) - (0.382683432365 * r2_22));
			 tmpi = ((0.923879532511 * r2_22) + (0.382683432365 * i2_22));
			 r1_10 = (r2_20 + tmpr);
			 i1_10 = (i2_20 - tmpi);
			 r1_26 = (r2_20 - tmpr);
			 i1_26 = (i2_20 + tmpi);
			 tmpr = (0.707106781187 * (i2_26 - r2_26));
			 tmpi = (0.707106781187 * (r2_26 + i2_26));
			 r1_12 = (r2_24 + tmpr);
			 i1_12 = (i2_24 - tmpi);
			 r1_28 = (r2_24 - tmpr);
			 i1_28 = (i2_24 + tmpi);
			 tmpr = ((0.382683432365 * i2_30) - (0.923879532511 * r2_30));
			 tmpi = ((0.382683432365 * r2_30) + (0.923879532511 * i2_30));
			 r1_14 = (r2_28 + tmpr);
			 i1_14 = (i2_28 - tmpi);
			 r1_30 = (r2_28 - tmpr);
			 i1_30 = (i2_28 + tmpi);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 REAL r2_5, i2_5;
			 REAL r2_7, i2_7;
			 REAL r2_9, i2_9;
			 REAL r2_11, i2_11;
			 REAL r2_13, i2_13;
			 REAL r2_15, i2_15;
			 REAL r2_17, i2_17;
			 REAL r2_19, i2_19;
			 REAL r2_21, i2_21;
			 REAL r2_23, i2_23;
			 REAL r2_25, i2_25;
			 REAL r2_27, i2_27;
			 REAL r2_29, i2_29;
			 REAL r2_31, i2_31;
			 {
			      REAL r3_1, i3_1;
			      REAL r3_5, i3_5;
			      REAL r3_9, i3_9;
			      REAL r3_13, i3_13;
			      REAL r3_17, i3_17;
			      REAL r3_21, i3_21;
			      REAL r3_25, i3_25;
			      REAL r3_29, i3_29;
			      {
				   REAL r4_1, i4_1;
				   REAL r4_9, i4_9;
				   REAL r4_17, i4_17;
				   REAL r4_25, i4_25;
				   {
					REAL r5_1, i5_1;
					REAL r5_17, i5_17;
					wr = c_re(W[1 * l1]);
					wi = c_im(W[1 * l1]);
					tmpr = c_re(jp[1 * m]);
					tmpi = c_im(jp[1 * m]);
					r5_1 = ((wr * tmpr) - (wi * tmpi));
					i5_1 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[17 * l1]);
					wi = c_im(W[17 * l1]);
					tmpr = c_re(jp[17 * m]);
					tmpi = c_im(jp[17 * m]);
					r5_17 = ((wr * tmpr) - (wi * tmpi));
					i5_17 = ((wi * tmpr) + (wr * tmpi));
					r4_1 = (r5_1 + r5_17);
					i4_1 = (i5_1 + i5_17);
					r4_17 = (r5_1 - r5_17);
					i4_17 = (i5_1 - i5_17);
				   }
				   {
					REAL r5_9, i5_9;
					REAL r5_25, i5_25;
					wr = c_re(W[9 * l1]);
					wi = c_im(W[9 * l1]);
					tmpr = c_re(jp[9 * m]);
					tmpi = c_im(jp[9 * m]);
					r5_9 = ((wr * tmpr) - (wi * tmpi));
					i5_9 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[25 * l1]);
					wi = c_im(W[25 * l1]);
					tmpr = c_re(jp[25 * m]);
					tmpi = c_im(jp[25 * m]);
					r5_25 = ((wr * tmpr) - (wi * tmpi));
					i5_25 = ((wi * tmpr) + (wr * tmpi));
					r4_9 = (r5_9 + r5_25);
					i4_9 = (i5_9 + i5_25);
					r4_25 = (r5_9 - r5_25);
					i4_25 = (i5_9 - i5_25);
				   }
				   r3_1 = (r4_1 + r4_9);
				   i3_1 = (i4_1 + i4_9);
				   r3_17 = (r4_1 - r4_9);
				   i3_17 = (i4_1 - i4_9);
				   r3_9 = (r4_17 + i4_25);
				   i3_9 = (i4_17 - r4_25);
				   r3_25 = (r4_17 - i4_25);
				   i3_25 = (i4_17 + r4_25);
			      }
			      {
				   REAL r4_5, i4_5;
				   REAL r4_13, i4_13;
				   REAL r4_21, i4_21;
				   REAL r4_29, i4_29;
				   {
					REAL r5_5, i5_5;
					REAL r5_21, i5_21;
					wr = c_re(W[5 * l1]);
					wi = c_im(W[5 * l1]);
					tmpr = c_re(jp[5 * m]);
					tmpi = c_im(jp[5 * m]);
					r5_5 = ((wr * tmpr) - (wi * tmpi));
					i5_5 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[21 * l1]);
					wi = c_im(W[21 * l1]);
					tmpr = c_re(jp[21 * m]);
					tmpi = c_im(jp[21 * m]);
					r5_21 = ((wr * tmpr) - (wi * tmpi));
					i5_21 = ((wi * tmpr) + (wr * tmpi));
					r4_5 = (r5_5 + r5_21);
					i4_5 = (i5_5 + i5_21);
					r4_21 = (r5_5 - r5_21);
					i4_21 = (i5_5 - i5_21);
				   }
				   {
					REAL r5_13, i5_13;
					REAL r5_29, i5_29;
					wr = c_re(W[13 * l1]);
					wi = c_im(W[13 * l1]);
					tmpr = c_re(jp[13 * m]);
					tmpi = c_im(jp[13 * m]);
					r5_13 = ((wr * tmpr) - (wi * tmpi));
					i5_13 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[29 * l1]);
					wi = c_im(W[29 * l1]);
					tmpr = c_re(jp[29 * m]);
					tmpi = c_im(jp[29 * m]);
					r5_29 = ((wr * tmpr) - (wi * tmpi));
					i5_29 = ((wi * tmpr) + (wr * tmpi));
					r4_13 = (r5_13 + r5_29);
					i4_13 = (i5_13 + i5_29);
					r4_29 = (r5_13 - r5_29);
					i4_29 = (i5_13 - i5_29);
				   }
				   r3_5 = (r4_5 + r4_13);
				   i3_5 = (i4_5 + i4_13);
				   r3_21 = (r4_5 - r4_13);
				   i3_21 = (i4_5 - i4_13);
				   r3_13 = (r4_21 + i4_29);
				   i3_13 = (i4_21 - r4_29);
				   r3_29 = (r4_21 - i4_29);
				   i3_29 = (i4_21 + r4_29);
			      }
			      r2_1 = (r3_1 + r3_5);
			      i2_1 = (i3_1 + i3_5);
			      r2_17 = (r3_1 - r3_5);
			      i2_17 = (i3_1 - i3_5);
			      tmpr = (0.707106781187 * (r3_13 + i3_13));
			      tmpi = (0.707106781187 * (i3_13 - r3_13));
			      r2_5 = (r3_9 + tmpr);
			      i2_5 = (i3_9 + tmpi);
			      r2_21 = (r3_9 - tmpr);
			      i2_21 = (i3_9 - tmpi);
			      r2_9 = (r3_17 + i3_21);
			      i2_9 = (i3_17 - r3_21);
			      r2_25 = (r3_17 - i3_21);
			      i2_25 = (i3_17 + r3_21);
			      tmpr = (0.707106781187 * (i3_29 - r3_29));
			      tmpi = (0.707106781187 * (r3_29 + i3_29));
			      r2_13 = (r3_25 + tmpr);
			      i2_13 = (i3_25 - tmpi);
			      r2_29 = (r3_25 - tmpr);
			      i2_29 = (i3_25 + tmpi);
			 }
			 {
			      REAL r3_3, i3_3;
			      REAL r3_7, i3_7;
			      REAL r3_11, i3_11;
			      REAL r3_15, i3_15;
			      REAL r3_19, i3_19;
			      REAL r3_23, i3_23;
			      REAL r3_27, i3_27;
			      REAL r3_31, i3_31;
			      {
				   REAL r4_3, i4_3;
				   REAL r4_11, i4_11;
				   REAL r4_19, i4_19;
				   REAL r4_27, i4_27;
				   {
					REAL r5_3, i5_3;
					REAL r5_19, i5_19;
					wr = c_re(W[3 * l1]);
					wi = c_im(W[3 * l1]);
					tmpr = c_re(jp[3 * m]);
					tmpi = c_im(jp[3 * m]);
					r5_3 = ((wr * tmpr) - (wi * tmpi));
					i5_3 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[19 * l1]);
					wi = c_im(W[19 * l1]);
					tmpr = c_re(jp[19 * m]);
					tmpi = c_im(jp[19 * m]);
					r5_19 = ((wr * tmpr) - (wi * tmpi));
					i5_19 = ((wi * tmpr) + (wr * tmpi));
					r4_3 = (r5_3 + r5_19);
					i4_3 = (i5_3 + i5_19);
					r4_19 = (r5_3 - r5_19);
					i4_19 = (i5_3 - i5_19);
				   }
				   {
					REAL r5_11, i5_11;
					REAL r5_27, i5_27;
					wr = c_re(W[11 * l1]);
					wi = c_im(W[11 * l1]);
					tmpr = c_re(jp[11 * m]);
					tmpi = c_im(jp[11 * m]);
					r5_11 = ((wr * tmpr) - (wi * tmpi));
					i5_11 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[27 * l1]);
					wi = c_im(W[27 * l1]);
					tmpr = c_re(jp[27 * m]);
					tmpi = c_im(jp[27 * m]);
					r5_27 = ((wr * tmpr) - (wi * tmpi));
					i5_27 = ((wi * tmpr) + (wr * tmpi));
					r4_11 = (r5_11 + r5_27);
					i4_11 = (i5_11 + i5_27);
					r4_27 = (r5_11 - r5_27);
					i4_27 = (i5_11 - i5_27);
				   }
				   r3_3 = (r4_3 + r4_11);
				   i3_3 = (i4_3 + i4_11);
				   r3_19 = (r4_3 - r4_11);
				   i3_19 = (i4_3 - i4_11);
				   r3_11 = (r4_19 + i4_27);
				   i3_11 = (i4_19 - r4_27);
				   r3_27 = (r4_19 - i4_27);
				   i3_27 = (i4_19 + r4_27);
			      }
			      {
				   REAL r4_7, i4_7;
				   REAL r4_15, i4_15;
				   REAL r4_23, i4_23;
				   REAL r4_31, i4_31;
				   {
					REAL r5_7, i5_7;
					REAL r5_23, i5_23;
					wr = c_re(W[7 * l1]);
					wi = c_im(W[7 * l1]);
					tmpr = c_re(jp[7 * m]);
					tmpi = c_im(jp[7 * m]);
					r5_7 = ((wr * tmpr) - (wi * tmpi));
					i5_7 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[23 * l1]);
					wi = c_im(W[23 * l1]);
					tmpr = c_re(jp[23 * m]);
					tmpi = c_im(jp[23 * m]);
					r5_23 = ((wr * tmpr) - (wi * tmpi));
					i5_23 = ((wi * tmpr) + (wr * tmpi));
					r4_7 = (r5_7 + r5_23);
					i4_7 = (i5_7 + i5_23);
					r4_23 = (r5_7 - r5_23);
					i4_23 = (i5_7 - i5_23);
				   }
				   {
					REAL r5_15, i5_15;
					REAL r5_31, i5_31;
					wr = c_re(W[15 * l1]);
					wi = c_im(W[15 * l1]);
					tmpr = c_re(jp[15 * m]);
					tmpi = c_im(jp[15 * m]);
					r5_15 = ((wr * tmpr) - (wi * tmpi));
					i5_15 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[31 * l1]);
					wi = c_im(W[31 * l1]);
					tmpr = c_re(jp[31 * m]);
					tmpi = c_im(jp[31 * m]);
					r5_31 = ((wr * tmpr) - (wi * tmpi));
					i5_31 = ((wi * tmpr) + (wr * tmpi));
					r4_15 = (r5_15 + r5_31);
					i4_15 = (i5_15 + i5_31);
					r4_31 = (r5_15 - r5_31);
					i4_31 = (i5_15 - i5_31);
				   }
				   r3_7 = (r4_7 + r4_15);
				   i3_7 = (i4_7 + i4_15);
				   r3_23 = (r4_7 - r4_15);
				   i3_23 = (i4_7 - i4_15);
				   r3_15 = (r4_23 + i4_31);
				   i3_15 = (i4_23 - r4_31);
				   r3_31 = (r4_23 - i4_31);
				   i3_31 = (i4_23 + r4_31);
			      }
			      r2_3 = (r3_3 + r3_7);
			      i2_3 = (i3_3 + i3_7);
			      r2_19 = (r3_3 - r3_7);
			      i2_19 = (i3_3 - i3_7);
			      tmpr = (0.707106781187 * (r3_15 + i3_15));
			      tmpi = (0.707106781187 * (i3_15 - r3_15));
			      r2_7 = (r3_11 + tmpr);
			      i2_7 = (i3_11 + tmpi);
			      r2_23 = (r3_11 - tmpr);
			      i2_23 = (i3_11 - tmpi);
			      r2_11 = (r3_19 + i3_23);
			      i2_11 = (i3_19 - r3_23);
			      r2_27 = (r3_19 - i3_23);
			      i2_27 = (i3_19 + r3_23);
			      tmpr = (0.707106781187 * (i3_31 - r3_31));
			      tmpi = (0.707106781187 * (r3_31 + i3_31));
			      r2_15 = (r3_27 + tmpr);
			      i2_15 = (i3_27 - tmpi);
			      r2_31 = (r3_27 - tmpr);
			      i2_31 = (i3_27 + tmpi);
			 }
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_17 = (r2_1 - r2_3);
			 i1_17 = (i2_1 - i2_3);
			 tmpr = ((0.923879532511 * r2_7) + (0.382683432365 * i2_7));
			 tmpi = ((0.923879532511 * i2_7) - (0.382683432365 * r2_7));
			 r1_3 = (r2_5 + tmpr);
			 i1_3 = (i2_5 + tmpi);
			 r1_19 = (r2_5 - tmpr);
			 i1_19 = (i2_5 - tmpi);
			 tmpr = (0.707106781187 * (r2_11 + i2_11));
			 tmpi = (0.707106781187 * (i2_11 - r2_11));
			 r1_5 = (r2_9 + tmpr);
			 i1_5 = (i2_9 + tmpi);
			 r1_21 = (r2_9 - tmpr);
			 i1_21 = (i2_9 - tmpi);
			 tmpr = ((0.382683432365 * r2_15) + (0.923879532511 * i2_15));
			 tmpi = ((0.382683432365 * i2_15) - (0.923879532511 * r2_15));
			 r1_7 = (r2_13 + tmpr);
			 i1_7 = (i2_13 + tmpi);
			 r1_23 = (r2_13 - tmpr);
			 i1_23 = (i2_13 - tmpi);
			 r1_9 = (r2_17 + i2_19);
			 i1_9 = (i2_17 - r2_19);
			 r1_25 = (r2_17 - i2_19);
			 i1_25 = (i2_17 + r2_19);
			 tmpr = ((0.923879532511 * i2_23) - (0.382683432365 * r2_23));
			 tmpi = ((0.923879532511 * r2_23) + (0.382683432365 * i2_23));
			 r1_11 = (r2_21 + tmpr);
			 i1_11 = (i2_21 - tmpi);
			 r1_27 = (r2_21 - tmpr);
			 i1_27 = (i2_21 + tmpi);
			 tmpr = (0.707106781187 * (i2_27 - r2_27));
			 tmpi = (0.707106781187 * (r2_27 + i2_27));
			 r1_13 = (r2_25 + tmpr);
			 i1_13 = (i2_25 - tmpi);
			 r1_29 = (r2_25 - tmpr);
			 i1_29 = (i2_25 + tmpi);
			 tmpr = ((0.382683432365 * i2_31) - (0.923879532511 * r2_31));
			 tmpi = ((0.382683432365 * r2_31) + (0.923879532511 * i2_31));
			 r1_15 = (r2_29 + tmpr);
			 i1_15 = (i2_29 - tmpi);
			 r1_31 = (r2_29 - tmpr);
			 i1_31 = (i2_29 + tmpi);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[16 * m]) = (r1_0 - r1_1);
		    c_im(kp[16 * m]) = (i1_0 - i1_1);
		    tmpr = ((0.980785280403 * r1_3) + (0.195090322016 * i1_3));
		    tmpi = ((0.980785280403 * i1_3) - (0.195090322016 * r1_3));
		    c_re(kp[1 * m]) = (r1_2 + tmpr);
		    c_im(kp[1 * m]) = (i1_2 + tmpi);
		    c_re(kp[17 * m]) = (r1_2 - tmpr);
		    c_im(kp[17 * m]) = (i1_2 - tmpi);
		    tmpr = ((0.923879532511 * r1_5) + (0.382683432365 * i1_5));
		    tmpi = ((0.923879532511 * i1_5) - (0.382683432365 * r1_5));
		    c_re(kp[2 * m]) = (r1_4 + tmpr);
		    c_im(kp[2 * m]) = (i1_4 + tmpi);
		    c_re(kp[18 * m]) = (r1_4 - tmpr);
		    c_im(kp[18 * m]) = (i1_4 - tmpi);
		    tmpr = ((0.831469612303 * r1_7) + (0.55557023302 * i1_7));
		    tmpi = ((0.831469612303 * i1_7) - (0.55557023302 * r1_7));
		    c_re(kp[3 * m]) = (r1_6 + tmpr);
		    c_im(kp[3 * m]) = (i1_6 + tmpi);
		    c_re(kp[19 * m]) = (r1_6 - tmpr);
		    c_im(kp[19 * m]) = (i1_6 - tmpi);
		    tmpr = (0.707106781187 * (r1_9 + i1_9));
		    tmpi = (0.707106781187 * (i1_9 - r1_9));
		    c_re(kp[4 * m]) = (r1_8 + tmpr);
		    c_im(kp[4 * m]) = (i1_8 + tmpi);
		    c_re(kp[20 * m]) = (r1_8 - tmpr);
		    c_im(kp[20 * m]) = (i1_8 - tmpi);
		    tmpr = ((0.55557023302 * r1_11) + (0.831469612303 * i1_11));
		    tmpi = ((0.55557023302 * i1_11) - (0.831469612303 * r1_11));
		    c_re(kp[5 * m]) = (r1_10 + tmpr);
		    c_im(kp[5 * m]) = (i1_10 + tmpi);
		    c_re(kp[21 * m]) = (r1_10 - tmpr);
		    c_im(kp[21 * m]) = (i1_10 - tmpi);
		    tmpr = ((0.382683432365 * r1_13) + (0.923879532511 * i1_13));
		    tmpi = ((0.382683432365 * i1_13) - (0.923879532511 * r1_13));
		    c_re(kp[6 * m]) = (r1_12 + tmpr);
		    c_im(kp[6 * m]) = (i1_12 + tmpi);
		    c_re(kp[22 * m]) = (r1_12 - tmpr);
		    c_im(kp[22 * m]) = (i1_12 - tmpi);
		    tmpr = ((0.195090322016 * r1_15) + (0.980785280403 * i1_15));
		    tmpi = ((0.195090322016 * i1_15) - (0.980785280403 * r1_15));
		    c_re(kp[7 * m]) = (r1_14 + tmpr);
		    c_im(kp[7 * m]) = (i1_14 + tmpi);
		    c_re(kp[23 * m]) = (r1_14 - tmpr);
		    c_im(kp[23 * m]) = (i1_14 - tmpi);
		    c_re(kp[8 * m]) = (r1_16 + i1_17);
		    c_im(kp[8 * m]) = (i1_16 - r1_17);
		    c_re(kp[24 * m]) = (r1_16 - i1_17);
		    c_im(kp[24 * m]) = (i1_16 + r1_17);
		    tmpr = ((0.980785280403 * i1_19) - (0.195090322016 * r1_19));
		    tmpi = ((0.980785280403 * r1_19) + (0.195090322016 * i1_19));
		    c_re(kp[9 * m]) = (r1_18 + tmpr);
		    c_im(kp[9 * m]) = (i1_18 - tmpi);
		    c_re(kp[25 * m]) = (r1_18 - tmpr);
		    c_im(kp[25 * m]) = (i1_18 + tmpi);
		    tmpr = ((0.923879532511 * i1_21) - (0.382683432365 * r1_21));
		    tmpi = ((0.923879532511 * r1_21) + (0.382683432365 * i1_21));
		    c_re(kp[10 * m]) = (r1_20 + tmpr);
		    c_im(kp[10 * m]) = (i1_20 - tmpi);
		    c_re(kp[26 * m]) = (r1_20 - tmpr);
		    c_im(kp[26 * m]) = (i1_20 + tmpi);
		    tmpr = ((0.831469612303 * i1_23) - (0.55557023302 * r1_23));
		    tmpi = ((0.831469612303 * r1_23) + (0.55557023302 * i1_23));
		    c_re(kp[11 * m]) = (r1_22 + tmpr);
		    c_im(kp[11 * m]) = (i1_22 - tmpi);
		    c_re(kp[27 * m]) = (r1_22 - tmpr);
		    c_im(kp[27 * m]) = (i1_22 + tmpi);
		    tmpr = (0.707106781187 * (i1_25 - r1_25));
		    tmpi = (0.707106781187 * (r1_25 + i1_25));
		    c_re(kp[12 * m]) = (r1_24 + tmpr);
		    c_im(kp[12 * m]) = (i1_24 - tmpi);
		    c_re(kp[28 * m]) = (r1_24 - tmpr);
		    c_im(kp[28 * m]) = (i1_24 + tmpi);
		    tmpr = ((0.55557023302 * i1_27) - (0.831469612303 * r1_27));
		    tmpi = ((0.55557023302 * r1_27) + (0.831469612303 * i1_27));
		    c_re(kp[13 * m]) = (r1_26 + tmpr);
		    c_im(kp[13 * m]) = (i1_26 - tmpi);
		    c_re(kp[29 * m]) = (r1_26 - tmpr);
		    c_im(kp[29 * m]) = (i1_26 + tmpi);
		    tmpr = ((0.382683432365 * i1_29) - (0.923879532511 * r1_29));
		    tmpi = ((0.382683432365 * r1_29) + (0.923879532511 * i1_29));
		    c_re(kp[14 * m]) = (r1_28 + tmpr);
		    c_im(kp[14 * m]) = (i1_28 - tmpi);
		    c_re(kp[30 * m]) = (r1_28 - tmpr);
		    c_im(kp[30 * m]) = (i1_28 + tmpi);
		    tmpr = ((0.195090322016 * i1_31) - (0.980785280403 * r1_31));
		    tmpi = ((0.195090322016 * r1_31) + (0.980785280403 * i1_31));
		    c_re(kp[15 * m]) = (r1_30 + tmpr);
		    c_im(kp[15 * m]) = (i1_30 - tmpi);
		    c_re(kp[31 * m]) = (r1_30 - tmpr);
		    c_im(kp[31 * m]) = (i1_30 + tmpi);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma3629_omp_task *new_ctx = (pragma3629_omp_task *)malloc(sizeof(pragma3629_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma3629_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma3631_omp_task *new_ctx = (pragma3631_omp_task *)malloc(sizeof(pragma3631_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->l1_ptr = &(l1);
new_ctx->i_ptr = &(i);
new_ctx->jp_ptr = &(jp);
new_ctx->kp_ptr = &(kp);
new_ctx->tmpr_ptr = &(tmpr);
new_ctx->tmpi_ptr = &(tmpi);
new_ctx->wr_ptr = &(wr);
new_ctx->wi_ptr = &(wi);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
new_ctx->nWdn_ptr = &(nWdn);
new_ctx->m_ptr = &(m);
hclib_async(pragma3631_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma3629_omp_task_hclib_async(void *____arg) {
    pragma3629_omp_task *ctx = (pragma3629_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_32((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma3631_omp_task_hclib_async(void *____arg) {
    pragma3631_omp_task *ctx = (pragma3631_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_32((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nWdn_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_twiddle_32_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m)
{
     int l1, i;
     COMPLEX *jp, *kp;
     REAL tmpr, tmpi, wr, wi;
     if ((b - a) < 128) {
	  for (i = a, l1 = nWdn * i, kp = out + i; i < b;
	       i++, l1 += nWdn, kp++) {
	       jp = in + i;
	       {
		    REAL r1_0, i1_0;
		    REAL r1_1, i1_1;
		    REAL r1_2, i1_2;
		    REAL r1_3, i1_3;
		    REAL r1_4, i1_4;
		    REAL r1_5, i1_5;
		    REAL r1_6, i1_6;
		    REAL r1_7, i1_7;
		    REAL r1_8, i1_8;
		    REAL r1_9, i1_9;
		    REAL r1_10, i1_10;
		    REAL r1_11, i1_11;
		    REAL r1_12, i1_12;
		    REAL r1_13, i1_13;
		    REAL r1_14, i1_14;
		    REAL r1_15, i1_15;
		    REAL r1_16, i1_16;
		    REAL r1_17, i1_17;
		    REAL r1_18, i1_18;
		    REAL r1_19, i1_19;
		    REAL r1_20, i1_20;
		    REAL r1_21, i1_21;
		    REAL r1_22, i1_22;
		    REAL r1_23, i1_23;
		    REAL r1_24, i1_24;
		    REAL r1_25, i1_25;
		    REAL r1_26, i1_26;
		    REAL r1_27, i1_27;
		    REAL r1_28, i1_28;
		    REAL r1_29, i1_29;
		    REAL r1_30, i1_30;
		    REAL r1_31, i1_31;
		    {
			 REAL r2_0, i2_0;
			 REAL r2_2, i2_2;
			 REAL r2_4, i2_4;
			 REAL r2_6, i2_6;
			 REAL r2_8, i2_8;
			 REAL r2_10, i2_10;
			 REAL r2_12, i2_12;
			 REAL r2_14, i2_14;
			 REAL r2_16, i2_16;
			 REAL r2_18, i2_18;
			 REAL r2_20, i2_20;
			 REAL r2_22, i2_22;
			 REAL r2_24, i2_24;
			 REAL r2_26, i2_26;
			 REAL r2_28, i2_28;
			 REAL r2_30, i2_30;
			 {
			      REAL r3_0, i3_0;
			      REAL r3_4, i3_4;
			      REAL r3_8, i3_8;
			      REAL r3_12, i3_12;
			      REAL r3_16, i3_16;
			      REAL r3_20, i3_20;
			      REAL r3_24, i3_24;
			      REAL r3_28, i3_28;
			      {
				   REAL r4_0, i4_0;
				   REAL r4_8, i4_8;
				   REAL r4_16, i4_16;
				   REAL r4_24, i4_24;
				   {
					REAL r5_0, i5_0;
					REAL r5_16, i5_16;
					r5_0 = c_re(jp[0 * m]);
					i5_0 = c_im(jp[0 * m]);
					wr = c_re(W[16 * l1]);
					wi = c_im(W[16 * l1]);
					tmpr = c_re(jp[16 * m]);
					tmpi = c_im(jp[16 * m]);
					r5_16 = ((wr * tmpr) - (wi * tmpi));
					i5_16 = ((wi * tmpr) + (wr * tmpi));
					r4_0 = (r5_0 + r5_16);
					i4_0 = (i5_0 + i5_16);
					r4_16 = (r5_0 - r5_16);
					i4_16 = (i5_0 - i5_16);
				   }
				   {
					REAL r5_8, i5_8;
					REAL r5_24, i5_24;
					wr = c_re(W[8 * l1]);
					wi = c_im(W[8 * l1]);
					tmpr = c_re(jp[8 * m]);
					tmpi = c_im(jp[8 * m]);
					r5_8 = ((wr * tmpr) - (wi * tmpi));
					i5_8 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[24 * l1]);
					wi = c_im(W[24 * l1]);
					tmpr = c_re(jp[24 * m]);
					tmpi = c_im(jp[24 * m]);
					r5_24 = ((wr * tmpr) - (wi * tmpi));
					i5_24 = ((wi * tmpr) + (wr * tmpi));
					r4_8 = (r5_8 + r5_24);
					i4_8 = (i5_8 + i5_24);
					r4_24 = (r5_8 - r5_24);
					i4_24 = (i5_8 - i5_24);
				   }
				   r3_0 = (r4_0 + r4_8);
				   i3_0 = (i4_0 + i4_8);
				   r3_16 = (r4_0 - r4_8);
				   i3_16 = (i4_0 - i4_8);
				   r3_8 = (r4_16 + i4_24);
				   i3_8 = (i4_16 - r4_24);
				   r3_24 = (r4_16 - i4_24);
				   i3_24 = (i4_16 + r4_24);
			      }
			      {
				   REAL r4_4, i4_4;
				   REAL r4_12, i4_12;
				   REAL r4_20, i4_20;
				   REAL r4_28, i4_28;
				   {
					REAL r5_4, i5_4;
					REAL r5_20, i5_20;
					wr = c_re(W[4 * l1]);
					wi = c_im(W[4 * l1]);
					tmpr = c_re(jp[4 * m]);
					tmpi = c_im(jp[4 * m]);
					r5_4 = ((wr * tmpr) - (wi * tmpi));
					i5_4 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[20 * l1]);
					wi = c_im(W[20 * l1]);
					tmpr = c_re(jp[20 * m]);
					tmpi = c_im(jp[20 * m]);
					r5_20 = ((wr * tmpr) - (wi * tmpi));
					i5_20 = ((wi * tmpr) + (wr * tmpi));
					r4_4 = (r5_4 + r5_20);
					i4_4 = (i5_4 + i5_20);
					r4_20 = (r5_4 - r5_20);
					i4_20 = (i5_4 - i5_20);
				   }
				   {
					REAL r5_12, i5_12;
					REAL r5_28, i5_28;
					wr = c_re(W[12 * l1]);
					wi = c_im(W[12 * l1]);
					tmpr = c_re(jp[12 * m]);
					tmpi = c_im(jp[12 * m]);
					r5_12 = ((wr * tmpr) - (wi * tmpi));
					i5_12 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[28 * l1]);
					wi = c_im(W[28 * l1]);
					tmpr = c_re(jp[28 * m]);
					tmpi = c_im(jp[28 * m]);
					r5_28 = ((wr * tmpr) - (wi * tmpi));
					i5_28 = ((wi * tmpr) + (wr * tmpi));
					r4_12 = (r5_12 + r5_28);
					i4_12 = (i5_12 + i5_28);
					r4_28 = (r5_12 - r5_28);
					i4_28 = (i5_12 - i5_28);
				   }
				   r3_4 = (r4_4 + r4_12);
				   i3_4 = (i4_4 + i4_12);
				   r3_20 = (r4_4 - r4_12);
				   i3_20 = (i4_4 - i4_12);
				   r3_12 = (r4_20 + i4_28);
				   i3_12 = (i4_20 - r4_28);
				   r3_28 = (r4_20 - i4_28);
				   i3_28 = (i4_20 + r4_28);
			      }
			      r2_0 = (r3_0 + r3_4);
			      i2_0 = (i3_0 + i3_4);
			      r2_16 = (r3_0 - r3_4);
			      i2_16 = (i3_0 - i3_4);
			      tmpr = (0.707106781187 * (r3_12 + i3_12));
			      tmpi = (0.707106781187 * (i3_12 - r3_12));
			      r2_4 = (r3_8 + tmpr);
			      i2_4 = (i3_8 + tmpi);
			      r2_20 = (r3_8 - tmpr);
			      i2_20 = (i3_8 - tmpi);
			      r2_8 = (r3_16 + i3_20);
			      i2_8 = (i3_16 - r3_20);
			      r2_24 = (r3_16 - i3_20);
			      i2_24 = (i3_16 + r3_20);
			      tmpr = (0.707106781187 * (i3_28 - r3_28));
			      tmpi = (0.707106781187 * (r3_28 + i3_28));
			      r2_12 = (r3_24 + tmpr);
			      i2_12 = (i3_24 - tmpi);
			      r2_28 = (r3_24 - tmpr);
			      i2_28 = (i3_24 + tmpi);
			 }
			 {
			      REAL r3_2, i3_2;
			      REAL r3_6, i3_6;
			      REAL r3_10, i3_10;
			      REAL r3_14, i3_14;
			      REAL r3_18, i3_18;
			      REAL r3_22, i3_22;
			      REAL r3_26, i3_26;
			      REAL r3_30, i3_30;
			      {
				   REAL r4_2, i4_2;
				   REAL r4_10, i4_10;
				   REAL r4_18, i4_18;
				   REAL r4_26, i4_26;
				   {
					REAL r5_2, i5_2;
					REAL r5_18, i5_18;
					wr = c_re(W[2 * l1]);
					wi = c_im(W[2 * l1]);
					tmpr = c_re(jp[2 * m]);
					tmpi = c_im(jp[2 * m]);
					r5_2 = ((wr * tmpr) - (wi * tmpi));
					i5_2 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[18 * l1]);
					wi = c_im(W[18 * l1]);
					tmpr = c_re(jp[18 * m]);
					tmpi = c_im(jp[18 * m]);
					r5_18 = ((wr * tmpr) - (wi * tmpi));
					i5_18 = ((wi * tmpr) + (wr * tmpi));
					r4_2 = (r5_2 + r5_18);
					i4_2 = (i5_2 + i5_18);
					r4_18 = (r5_2 - r5_18);
					i4_18 = (i5_2 - i5_18);
				   }
				   {
					REAL r5_10, i5_10;
					REAL r5_26, i5_26;
					wr = c_re(W[10 * l1]);
					wi = c_im(W[10 * l1]);
					tmpr = c_re(jp[10 * m]);
					tmpi = c_im(jp[10 * m]);
					r5_10 = ((wr * tmpr) - (wi * tmpi));
					i5_10 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[26 * l1]);
					wi = c_im(W[26 * l1]);
					tmpr = c_re(jp[26 * m]);
					tmpi = c_im(jp[26 * m]);
					r5_26 = ((wr * tmpr) - (wi * tmpi));
					i5_26 = ((wi * tmpr) + (wr * tmpi));
					r4_10 = (r5_10 + r5_26);
					i4_10 = (i5_10 + i5_26);
					r4_26 = (r5_10 - r5_26);
					i4_26 = (i5_10 - i5_26);
				   }
				   r3_2 = (r4_2 + r4_10);
				   i3_2 = (i4_2 + i4_10);
				   r3_18 = (r4_2 - r4_10);
				   i3_18 = (i4_2 - i4_10);
				   r3_10 = (r4_18 + i4_26);
				   i3_10 = (i4_18 - r4_26);
				   r3_26 = (r4_18 - i4_26);
				   i3_26 = (i4_18 + r4_26);
			      }
			      {
				   REAL r4_6, i4_6;
				   REAL r4_14, i4_14;
				   REAL r4_22, i4_22;
				   REAL r4_30, i4_30;
				   {
					REAL r5_6, i5_6;
					REAL r5_22, i5_22;
					wr = c_re(W[6 * l1]);
					wi = c_im(W[6 * l1]);
					tmpr = c_re(jp[6 * m]);
					tmpi = c_im(jp[6 * m]);
					r5_6 = ((wr * tmpr) - (wi * tmpi));
					i5_6 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[22 * l1]);
					wi = c_im(W[22 * l1]);
					tmpr = c_re(jp[22 * m]);
					tmpi = c_im(jp[22 * m]);
					r5_22 = ((wr * tmpr) - (wi * tmpi));
					i5_22 = ((wi * tmpr) + (wr * tmpi));
					r4_6 = (r5_6 + r5_22);
					i4_6 = (i5_6 + i5_22);
					r4_22 = (r5_6 - r5_22);
					i4_22 = (i5_6 - i5_22);
				   }
				   {
					REAL r5_14, i5_14;
					REAL r5_30, i5_30;
					wr = c_re(W[14 * l1]);
					wi = c_im(W[14 * l1]);
					tmpr = c_re(jp[14 * m]);
					tmpi = c_im(jp[14 * m]);
					r5_14 = ((wr * tmpr) - (wi * tmpi));
					i5_14 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[30 * l1]);
					wi = c_im(W[30 * l1]);
					tmpr = c_re(jp[30 * m]);
					tmpi = c_im(jp[30 * m]);
					r5_30 = ((wr * tmpr) - (wi * tmpi));
					i5_30 = ((wi * tmpr) + (wr * tmpi));
					r4_14 = (r5_14 + r5_30);
					i4_14 = (i5_14 + i5_30);
					r4_30 = (r5_14 - r5_30);
					i4_30 = (i5_14 - i5_30);
				   }
				   r3_6 = (r4_6 + r4_14);
				   i3_6 = (i4_6 + i4_14);
				   r3_22 = (r4_6 - r4_14);
				   i3_22 = (i4_6 - i4_14);
				   r3_14 = (r4_22 + i4_30);
				   i3_14 = (i4_22 - r4_30);
				   r3_30 = (r4_22 - i4_30);
				   i3_30 = (i4_22 + r4_30);
			      }
			      r2_2 = (r3_2 + r3_6);
			      i2_2 = (i3_2 + i3_6);
			      r2_18 = (r3_2 - r3_6);
			      i2_18 = (i3_2 - i3_6);
			      tmpr = (0.707106781187 * (r3_14 + i3_14));
			      tmpi = (0.707106781187 * (i3_14 - r3_14));
			      r2_6 = (r3_10 + tmpr);
			      i2_6 = (i3_10 + tmpi);
			      r2_22 = (r3_10 - tmpr);
			      i2_22 = (i3_10 - tmpi);
			      r2_10 = (r3_18 + i3_22);
			      i2_10 = (i3_18 - r3_22);
			      r2_26 = (r3_18 - i3_22);
			      i2_26 = (i3_18 + r3_22);
			      tmpr = (0.707106781187 * (i3_30 - r3_30));
			      tmpi = (0.707106781187 * (r3_30 + i3_30));
			      r2_14 = (r3_26 + tmpr);
			      i2_14 = (i3_26 - tmpi);
			      r2_30 = (r3_26 - tmpr);
			      i2_30 = (i3_26 + tmpi);
			 }
			 r1_0 = (r2_0 + r2_2);
			 i1_0 = (i2_0 + i2_2);
			 r1_16 = (r2_0 - r2_2);
			 i1_16 = (i2_0 - i2_2);
			 tmpr = ((0.923879532511 * r2_6) + (0.382683432365 * i2_6));
			 tmpi = ((0.923879532511 * i2_6) - (0.382683432365 * r2_6));
			 r1_2 = (r2_4 + tmpr);
			 i1_2 = (i2_4 + tmpi);
			 r1_18 = (r2_4 - tmpr);
			 i1_18 = (i2_4 - tmpi);
			 tmpr = (0.707106781187 * (r2_10 + i2_10));
			 tmpi = (0.707106781187 * (i2_10 - r2_10));
			 r1_4 = (r2_8 + tmpr);
			 i1_4 = (i2_8 + tmpi);
			 r1_20 = (r2_8 - tmpr);
			 i1_20 = (i2_8 - tmpi);
			 tmpr = ((0.382683432365 * r2_14) + (0.923879532511 * i2_14));
			 tmpi = ((0.382683432365 * i2_14) - (0.923879532511 * r2_14));
			 r1_6 = (r2_12 + tmpr);
			 i1_6 = (i2_12 + tmpi);
			 r1_22 = (r2_12 - tmpr);
			 i1_22 = (i2_12 - tmpi);
			 r1_8 = (r2_16 + i2_18);
			 i1_8 = (i2_16 - r2_18);
			 r1_24 = (r2_16 - i2_18);
			 i1_24 = (i2_16 + r2_18);
			 tmpr = ((0.923879532511 * i2_22) - (0.382683432365 * r2_22));
			 tmpi = ((0.923879532511 * r2_22) + (0.382683432365 * i2_22));
			 r1_10 = (r2_20 + tmpr);
			 i1_10 = (i2_20 - tmpi);
			 r1_26 = (r2_20 - tmpr);
			 i1_26 = (i2_20 + tmpi);
			 tmpr = (0.707106781187 * (i2_26 - r2_26));
			 tmpi = (0.707106781187 * (r2_26 + i2_26));
			 r1_12 = (r2_24 + tmpr);
			 i1_12 = (i2_24 - tmpi);
			 r1_28 = (r2_24 - tmpr);
			 i1_28 = (i2_24 + tmpi);
			 tmpr = ((0.382683432365 * i2_30) - (0.923879532511 * r2_30));
			 tmpi = ((0.382683432365 * r2_30) + (0.923879532511 * i2_30));
			 r1_14 = (r2_28 + tmpr);
			 i1_14 = (i2_28 - tmpi);
			 r1_30 = (r2_28 - tmpr);
			 i1_30 = (i2_28 + tmpi);
		    }
		    {
			 REAL r2_1, i2_1;
			 REAL r2_3, i2_3;
			 REAL r2_5, i2_5;
			 REAL r2_7, i2_7;
			 REAL r2_9, i2_9;
			 REAL r2_11, i2_11;
			 REAL r2_13, i2_13;
			 REAL r2_15, i2_15;
			 REAL r2_17, i2_17;
			 REAL r2_19, i2_19;
			 REAL r2_21, i2_21;
			 REAL r2_23, i2_23;
			 REAL r2_25, i2_25;
			 REAL r2_27, i2_27;
			 REAL r2_29, i2_29;
			 REAL r2_31, i2_31;
			 {
			      REAL r3_1, i3_1;
			      REAL r3_5, i3_5;
			      REAL r3_9, i3_9;
			      REAL r3_13, i3_13;
			      REAL r3_17, i3_17;
			      REAL r3_21, i3_21;
			      REAL r3_25, i3_25;
			      REAL r3_29, i3_29;
			      {
				   REAL r4_1, i4_1;
				   REAL r4_9, i4_9;
				   REAL r4_17, i4_17;
				   REAL r4_25, i4_25;
				   {
					REAL r5_1, i5_1;
					REAL r5_17, i5_17;
					wr = c_re(W[1 * l1]);
					wi = c_im(W[1 * l1]);
					tmpr = c_re(jp[1 * m]);
					tmpi = c_im(jp[1 * m]);
					r5_1 = ((wr * tmpr) - (wi * tmpi));
					i5_1 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[17 * l1]);
					wi = c_im(W[17 * l1]);
					tmpr = c_re(jp[17 * m]);
					tmpi = c_im(jp[17 * m]);
					r5_17 = ((wr * tmpr) - (wi * tmpi));
					i5_17 = ((wi * tmpr) + (wr * tmpi));
					r4_1 = (r5_1 + r5_17);
					i4_1 = (i5_1 + i5_17);
					r4_17 = (r5_1 - r5_17);
					i4_17 = (i5_1 - i5_17);
				   }
				   {
					REAL r5_9, i5_9;
					REAL r5_25, i5_25;
					wr = c_re(W[9 * l1]);
					wi = c_im(W[9 * l1]);
					tmpr = c_re(jp[9 * m]);
					tmpi = c_im(jp[9 * m]);
					r5_9 = ((wr * tmpr) - (wi * tmpi));
					i5_9 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[25 * l1]);
					wi = c_im(W[25 * l1]);
					tmpr = c_re(jp[25 * m]);
					tmpi = c_im(jp[25 * m]);
					r5_25 = ((wr * tmpr) - (wi * tmpi));
					i5_25 = ((wi * tmpr) + (wr * tmpi));
					r4_9 = (r5_9 + r5_25);
					i4_9 = (i5_9 + i5_25);
					r4_25 = (r5_9 - r5_25);
					i4_25 = (i5_9 - i5_25);
				   }
				   r3_1 = (r4_1 + r4_9);
				   i3_1 = (i4_1 + i4_9);
				   r3_17 = (r4_1 - r4_9);
				   i3_17 = (i4_1 - i4_9);
				   r3_9 = (r4_17 + i4_25);
				   i3_9 = (i4_17 - r4_25);
				   r3_25 = (r4_17 - i4_25);
				   i3_25 = (i4_17 + r4_25);
			      }
			      {
				   REAL r4_5, i4_5;
				   REAL r4_13, i4_13;
				   REAL r4_21, i4_21;
				   REAL r4_29, i4_29;
				   {
					REAL r5_5, i5_5;
					REAL r5_21, i5_21;
					wr = c_re(W[5 * l1]);
					wi = c_im(W[5 * l1]);
					tmpr = c_re(jp[5 * m]);
					tmpi = c_im(jp[5 * m]);
					r5_5 = ((wr * tmpr) - (wi * tmpi));
					i5_5 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[21 * l1]);
					wi = c_im(W[21 * l1]);
					tmpr = c_re(jp[21 * m]);
					tmpi = c_im(jp[21 * m]);
					r5_21 = ((wr * tmpr) - (wi * tmpi));
					i5_21 = ((wi * tmpr) + (wr * tmpi));
					r4_5 = (r5_5 + r5_21);
					i4_5 = (i5_5 + i5_21);
					r4_21 = (r5_5 - r5_21);
					i4_21 = (i5_5 - i5_21);
				   }
				   {
					REAL r5_13, i5_13;
					REAL r5_29, i5_29;
					wr = c_re(W[13 * l1]);
					wi = c_im(W[13 * l1]);
					tmpr = c_re(jp[13 * m]);
					tmpi = c_im(jp[13 * m]);
					r5_13 = ((wr * tmpr) - (wi * tmpi));
					i5_13 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[29 * l1]);
					wi = c_im(W[29 * l1]);
					tmpr = c_re(jp[29 * m]);
					tmpi = c_im(jp[29 * m]);
					r5_29 = ((wr * tmpr) - (wi * tmpi));
					i5_29 = ((wi * tmpr) + (wr * tmpi));
					r4_13 = (r5_13 + r5_29);
					i4_13 = (i5_13 + i5_29);
					r4_29 = (r5_13 - r5_29);
					i4_29 = (i5_13 - i5_29);
				   }
				   r3_5 = (r4_5 + r4_13);
				   i3_5 = (i4_5 + i4_13);
				   r3_21 = (r4_5 - r4_13);
				   i3_21 = (i4_5 - i4_13);
				   r3_13 = (r4_21 + i4_29);
				   i3_13 = (i4_21 - r4_29);
				   r3_29 = (r4_21 - i4_29);
				   i3_29 = (i4_21 + r4_29);
			      }
			      r2_1 = (r3_1 + r3_5);
			      i2_1 = (i3_1 + i3_5);
			      r2_17 = (r3_1 - r3_5);
			      i2_17 = (i3_1 - i3_5);
			      tmpr = (0.707106781187 * (r3_13 + i3_13));
			      tmpi = (0.707106781187 * (i3_13 - r3_13));
			      r2_5 = (r3_9 + tmpr);
			      i2_5 = (i3_9 + tmpi);
			      r2_21 = (r3_9 - tmpr);
			      i2_21 = (i3_9 - tmpi);
			      r2_9 = (r3_17 + i3_21);
			      i2_9 = (i3_17 - r3_21);
			      r2_25 = (r3_17 - i3_21);
			      i2_25 = (i3_17 + r3_21);
			      tmpr = (0.707106781187 * (i3_29 - r3_29));
			      tmpi = (0.707106781187 * (r3_29 + i3_29));
			      r2_13 = (r3_25 + tmpr);
			      i2_13 = (i3_25 - tmpi);
			      r2_29 = (r3_25 - tmpr);
			      i2_29 = (i3_25 + tmpi);
			 }
			 {
			      REAL r3_3, i3_3;
			      REAL r3_7, i3_7;
			      REAL r3_11, i3_11;
			      REAL r3_15, i3_15;
			      REAL r3_19, i3_19;
			      REAL r3_23, i3_23;
			      REAL r3_27, i3_27;
			      REAL r3_31, i3_31;
			      {
				   REAL r4_3, i4_3;
				   REAL r4_11, i4_11;
				   REAL r4_19, i4_19;
				   REAL r4_27, i4_27;
				   {
					REAL r5_3, i5_3;
					REAL r5_19, i5_19;
					wr = c_re(W[3 * l1]);
					wi = c_im(W[3 * l1]);
					tmpr = c_re(jp[3 * m]);
					tmpi = c_im(jp[3 * m]);
					r5_3 = ((wr * tmpr) - (wi * tmpi));
					i5_3 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[19 * l1]);
					wi = c_im(W[19 * l1]);
					tmpr = c_re(jp[19 * m]);
					tmpi = c_im(jp[19 * m]);
					r5_19 = ((wr * tmpr) - (wi * tmpi));
					i5_19 = ((wi * tmpr) + (wr * tmpi));
					r4_3 = (r5_3 + r5_19);
					i4_3 = (i5_3 + i5_19);
					r4_19 = (r5_3 - r5_19);
					i4_19 = (i5_3 - i5_19);
				   }
				   {
					REAL r5_11, i5_11;
					REAL r5_27, i5_27;
					wr = c_re(W[11 * l1]);
					wi = c_im(W[11 * l1]);
					tmpr = c_re(jp[11 * m]);
					tmpi = c_im(jp[11 * m]);
					r5_11 = ((wr * tmpr) - (wi * tmpi));
					i5_11 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[27 * l1]);
					wi = c_im(W[27 * l1]);
					tmpr = c_re(jp[27 * m]);
					tmpi = c_im(jp[27 * m]);
					r5_27 = ((wr * tmpr) - (wi * tmpi));
					i5_27 = ((wi * tmpr) + (wr * tmpi));
					r4_11 = (r5_11 + r5_27);
					i4_11 = (i5_11 + i5_27);
					r4_27 = (r5_11 - r5_27);
					i4_27 = (i5_11 - i5_27);
				   }
				   r3_3 = (r4_3 + r4_11);
				   i3_3 = (i4_3 + i4_11);
				   r3_19 = (r4_3 - r4_11);
				   i3_19 = (i4_3 - i4_11);
				   r3_11 = (r4_19 + i4_27);
				   i3_11 = (i4_19 - r4_27);
				   r3_27 = (r4_19 - i4_27);
				   i3_27 = (i4_19 + r4_27);
			      }
			      {
				   REAL r4_7, i4_7;
				   REAL r4_15, i4_15;
				   REAL r4_23, i4_23;
				   REAL r4_31, i4_31;
				   {
					REAL r5_7, i5_7;
					REAL r5_23, i5_23;
					wr = c_re(W[7 * l1]);
					wi = c_im(W[7 * l1]);
					tmpr = c_re(jp[7 * m]);
					tmpi = c_im(jp[7 * m]);
					r5_7 = ((wr * tmpr) - (wi * tmpi));
					i5_7 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[23 * l1]);
					wi = c_im(W[23 * l1]);
					tmpr = c_re(jp[23 * m]);
					tmpi = c_im(jp[23 * m]);
					r5_23 = ((wr * tmpr) - (wi * tmpi));
					i5_23 = ((wi * tmpr) + (wr * tmpi));
					r4_7 = (r5_7 + r5_23);
					i4_7 = (i5_7 + i5_23);
					r4_23 = (r5_7 - r5_23);
					i4_23 = (i5_7 - i5_23);
				   }
				   {
					REAL r5_15, i5_15;
					REAL r5_31, i5_31;
					wr = c_re(W[15 * l1]);
					wi = c_im(W[15 * l1]);
					tmpr = c_re(jp[15 * m]);
					tmpi = c_im(jp[15 * m]);
					r5_15 = ((wr * tmpr) - (wi * tmpi));
					i5_15 = ((wi * tmpr) + (wr * tmpi));
					wr = c_re(W[31 * l1]);
					wi = c_im(W[31 * l1]);
					tmpr = c_re(jp[31 * m]);
					tmpi = c_im(jp[31 * m]);
					r5_31 = ((wr * tmpr) - (wi * tmpi));
					i5_31 = ((wi * tmpr) + (wr * tmpi));
					r4_15 = (r5_15 + r5_31);
					i4_15 = (i5_15 + i5_31);
					r4_31 = (r5_15 - r5_31);
					i4_31 = (i5_15 - i5_31);
				   }
				   r3_7 = (r4_7 + r4_15);
				   i3_7 = (i4_7 + i4_15);
				   r3_23 = (r4_7 - r4_15);
				   i3_23 = (i4_7 - i4_15);
				   r3_15 = (r4_23 + i4_31);
				   i3_15 = (i4_23 - r4_31);
				   r3_31 = (r4_23 - i4_31);
				   i3_31 = (i4_23 + r4_31);
			      }
			      r2_3 = (r3_3 + r3_7);
			      i2_3 = (i3_3 + i3_7);
			      r2_19 = (r3_3 - r3_7);
			      i2_19 = (i3_3 - i3_7);
			      tmpr = (0.707106781187 * (r3_15 + i3_15));
			      tmpi = (0.707106781187 * (i3_15 - r3_15));
			      r2_7 = (r3_11 + tmpr);
			      i2_7 = (i3_11 + tmpi);
			      r2_23 = (r3_11 - tmpr);
			      i2_23 = (i3_11 - tmpi);
			      r2_11 = (r3_19 + i3_23);
			      i2_11 = (i3_19 - r3_23);
			      r2_27 = (r3_19 - i3_23);
			      i2_27 = (i3_19 + r3_23);
			      tmpr = (0.707106781187 * (i3_31 - r3_31));
			      tmpi = (0.707106781187 * (r3_31 + i3_31));
			      r2_15 = (r3_27 + tmpr);
			      i2_15 = (i3_27 - tmpi);
			      r2_31 = (r3_27 - tmpr);
			      i2_31 = (i3_27 + tmpi);
			 }
			 r1_1 = (r2_1 + r2_3);
			 i1_1 = (i2_1 + i2_3);
			 r1_17 = (r2_1 - r2_3);
			 i1_17 = (i2_1 - i2_3);
			 tmpr = ((0.923879532511 * r2_7) + (0.382683432365 * i2_7));
			 tmpi = ((0.923879532511 * i2_7) - (0.382683432365 * r2_7));
			 r1_3 = (r2_5 + tmpr);
			 i1_3 = (i2_5 + tmpi);
			 r1_19 = (r2_5 - tmpr);
			 i1_19 = (i2_5 - tmpi);
			 tmpr = (0.707106781187 * (r2_11 + i2_11));
			 tmpi = (0.707106781187 * (i2_11 - r2_11));
			 r1_5 = (r2_9 + tmpr);
			 i1_5 = (i2_9 + tmpi);
			 r1_21 = (r2_9 - tmpr);
			 i1_21 = (i2_9 - tmpi);
			 tmpr = ((0.382683432365 * r2_15) + (0.923879532511 * i2_15));
			 tmpi = ((0.382683432365 * i2_15) - (0.923879532511 * r2_15));
			 r1_7 = (r2_13 + tmpr);
			 i1_7 = (i2_13 + tmpi);
			 r1_23 = (r2_13 - tmpr);
			 i1_23 = (i2_13 - tmpi);
			 r1_9 = (r2_17 + i2_19);
			 i1_9 = (i2_17 - r2_19);
			 r1_25 = (r2_17 - i2_19);
			 i1_25 = (i2_17 + r2_19);
			 tmpr = ((0.923879532511 * i2_23) - (0.382683432365 * r2_23));
			 tmpi = ((0.923879532511 * r2_23) + (0.382683432365 * i2_23));
			 r1_11 = (r2_21 + tmpr);
			 i1_11 = (i2_21 - tmpi);
			 r1_27 = (r2_21 - tmpr);
			 i1_27 = (i2_21 + tmpi);
			 tmpr = (0.707106781187 * (i2_27 - r2_27));
			 tmpi = (0.707106781187 * (r2_27 + i2_27));
			 r1_13 = (r2_25 + tmpr);
			 i1_13 = (i2_25 - tmpi);
			 r1_29 = (r2_25 - tmpr);
			 i1_29 = (i2_25 + tmpi);
			 tmpr = ((0.382683432365 * i2_31) - (0.923879532511 * r2_31));
			 tmpi = ((0.382683432365 * r2_31) + (0.923879532511 * i2_31));
			 r1_15 = (r2_29 + tmpr);
			 i1_15 = (i2_29 - tmpi);
			 r1_31 = (r2_29 - tmpr);
			 i1_31 = (i2_29 + tmpi);
		    }
		    c_re(kp[0 * m]) = (r1_0 + r1_1);
		    c_im(kp[0 * m]) = (i1_0 + i1_1);
		    c_re(kp[16 * m]) = (r1_0 - r1_1);
		    c_im(kp[16 * m]) = (i1_0 - i1_1);
		    tmpr = ((0.980785280403 * r1_3) + (0.195090322016 * i1_3));
		    tmpi = ((0.980785280403 * i1_3) - (0.195090322016 * r1_3));
		    c_re(kp[1 * m]) = (r1_2 + tmpr);
		    c_im(kp[1 * m]) = (i1_2 + tmpi);
		    c_re(kp[17 * m]) = (r1_2 - tmpr);
		    c_im(kp[17 * m]) = (i1_2 - tmpi);
		    tmpr = ((0.923879532511 * r1_5) + (0.382683432365 * i1_5));
		    tmpi = ((0.923879532511 * i1_5) - (0.382683432365 * r1_5));
		    c_re(kp[2 * m]) = (r1_4 + tmpr);
		    c_im(kp[2 * m]) = (i1_4 + tmpi);
		    c_re(kp[18 * m]) = (r1_4 - tmpr);
		    c_im(kp[18 * m]) = (i1_4 - tmpi);
		    tmpr = ((0.831469612303 * r1_7) + (0.55557023302 * i1_7));
		    tmpi = ((0.831469612303 * i1_7) - (0.55557023302 * r1_7));
		    c_re(kp[3 * m]) = (r1_6 + tmpr);
		    c_im(kp[3 * m]) = (i1_6 + tmpi);
		    c_re(kp[19 * m]) = (r1_6 - tmpr);
		    c_im(kp[19 * m]) = (i1_6 - tmpi);
		    tmpr = (0.707106781187 * (r1_9 + i1_9));
		    tmpi = (0.707106781187 * (i1_9 - r1_9));
		    c_re(kp[4 * m]) = (r1_8 + tmpr);
		    c_im(kp[4 * m]) = (i1_8 + tmpi);
		    c_re(kp[20 * m]) = (r1_8 - tmpr);
		    c_im(kp[20 * m]) = (i1_8 - tmpi);
		    tmpr = ((0.55557023302 * r1_11) + (0.831469612303 * i1_11));
		    tmpi = ((0.55557023302 * i1_11) - (0.831469612303 * r1_11));
		    c_re(kp[5 * m]) = (r1_10 + tmpr);
		    c_im(kp[5 * m]) = (i1_10 + tmpi);
		    c_re(kp[21 * m]) = (r1_10 - tmpr);
		    c_im(kp[21 * m]) = (i1_10 - tmpi);
		    tmpr = ((0.382683432365 * r1_13) + (0.923879532511 * i1_13));
		    tmpi = ((0.382683432365 * i1_13) - (0.923879532511 * r1_13));
		    c_re(kp[6 * m]) = (r1_12 + tmpr);
		    c_im(kp[6 * m]) = (i1_12 + tmpi);
		    c_re(kp[22 * m]) = (r1_12 - tmpr);
		    c_im(kp[22 * m]) = (i1_12 - tmpi);
		    tmpr = ((0.195090322016 * r1_15) + (0.980785280403 * i1_15));
		    tmpi = ((0.195090322016 * i1_15) - (0.980785280403 * r1_15));
		    c_re(kp[7 * m]) = (r1_14 + tmpr);
		    c_im(kp[7 * m]) = (i1_14 + tmpi);
		    c_re(kp[23 * m]) = (r1_14 - tmpr);
		    c_im(kp[23 * m]) = (i1_14 - tmpi);
		    c_re(kp[8 * m]) = (r1_16 + i1_17);
		    c_im(kp[8 * m]) = (i1_16 - r1_17);
		    c_re(kp[24 * m]) = (r1_16 - i1_17);
		    c_im(kp[24 * m]) = (i1_16 + r1_17);
		    tmpr = ((0.980785280403 * i1_19) - (0.195090322016 * r1_19));
		    tmpi = ((0.980785280403 * r1_19) + (0.195090322016 * i1_19));
		    c_re(kp[9 * m]) = (r1_18 + tmpr);
		    c_im(kp[9 * m]) = (i1_18 - tmpi);
		    c_re(kp[25 * m]) = (r1_18 - tmpr);
		    c_im(kp[25 * m]) = (i1_18 + tmpi);
		    tmpr = ((0.923879532511 * i1_21) - (0.382683432365 * r1_21));
		    tmpi = ((0.923879532511 * r1_21) + (0.382683432365 * i1_21));
		    c_re(kp[10 * m]) = (r1_20 + tmpr);
		    c_im(kp[10 * m]) = (i1_20 - tmpi);
		    c_re(kp[26 * m]) = (r1_20 - tmpr);
		    c_im(kp[26 * m]) = (i1_20 + tmpi);
		    tmpr = ((0.831469612303 * i1_23) - (0.55557023302 * r1_23));
		    tmpi = ((0.831469612303 * r1_23) + (0.55557023302 * i1_23));
		    c_re(kp[11 * m]) = (r1_22 + tmpr);
		    c_im(kp[11 * m]) = (i1_22 - tmpi);
		    c_re(kp[27 * m]) = (r1_22 - tmpr);
		    c_im(kp[27 * m]) = (i1_22 + tmpi);
		    tmpr = (0.707106781187 * (i1_25 - r1_25));
		    tmpi = (0.707106781187 * (r1_25 + i1_25));
		    c_re(kp[12 * m]) = (r1_24 + tmpr);
		    c_im(kp[12 * m]) = (i1_24 - tmpi);
		    c_re(kp[28 * m]) = (r1_24 - tmpr);
		    c_im(kp[28 * m]) = (i1_24 + tmpi);
		    tmpr = ((0.55557023302 * i1_27) - (0.831469612303 * r1_27));
		    tmpi = ((0.55557023302 * r1_27) + (0.831469612303 * i1_27));
		    c_re(kp[13 * m]) = (r1_26 + tmpr);
		    c_im(kp[13 * m]) = (i1_26 - tmpi);
		    c_re(kp[29 * m]) = (r1_26 - tmpr);
		    c_im(kp[29 * m]) = (i1_26 + tmpi);
		    tmpr = ((0.382683432365 * i1_29) - (0.923879532511 * r1_29));
		    tmpi = ((0.382683432365 * r1_29) + (0.923879532511 * i1_29));
		    c_re(kp[14 * m]) = (r1_28 + tmpr);
		    c_im(kp[14 * m]) = (i1_28 - tmpi);
		    c_re(kp[30 * m]) = (r1_28 - tmpr);
		    c_im(kp[30 * m]) = (i1_28 + tmpi);
		    tmpr = ((0.195090322016 * i1_31) - (0.980785280403 * r1_31));
		    tmpi = ((0.195090322016 * r1_31) + (0.980785280403 * i1_31));
		    c_re(kp[15 * m]) = (r1_30 + tmpr);
		    c_im(kp[15 * m]) = (i1_30 - tmpi);
		    c_re(kp[31 * m]) = (r1_30 - tmpr);
		    c_im(kp[31 * m]) = (i1_30 + tmpi);
	       }
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_twiddle_32_seq(a, ab, in, out, W, nW, nWdn, m);
	  fft_twiddle_32_seq(ab, b, in, out, W, nW, nWdn, m);
     }
}
typedef struct _pragma4525_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma4525_omp_task;

typedef struct _pragma4527_omp_task {
    int (*ab_ptr);
    int (*i_ptr);
    COMPLEX (*(*ip_ptr));
    COMPLEX (*(*jp_ptr));
    int (*a_ptr);
    int (*b_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*m_ptr);
 } pragma4527_omp_task;

static void pragma4525_omp_task_hclib_async(void *____arg);
static void pragma4527_omp_task_hclib_async(void *____arg);
void fft_unshuffle_32(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 32;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
 { 
pragma4525_omp_task *new_ctx = (pragma4525_omp_task *)malloc(sizeof(pragma4525_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma4525_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 { 
pragma4527_omp_task *new_ctx = (pragma4527_omp_task *)malloc(sizeof(pragma4527_omp_task));
new_ctx->ab_ptr = &(ab);
new_ctx->i_ptr = &(i);
new_ctx->ip_ptr = &(ip);
new_ctx->jp_ptr = &(jp);
new_ctx->a_ptr = &(a);
new_ctx->b_ptr = &(b);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->m_ptr = &(m);
hclib_async(pragma4527_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
 hclib_end_finish(); hclib_start_finish(); ;
     }
} 
static void pragma4525_omp_task_hclib_async(void *____arg) {
    pragma4525_omp_task *ctx = (pragma4525_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_32((*(ctx->a_ptr)), (*(ctx->ab_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4527_omp_task_hclib_async(void *____arg) {
    pragma4527_omp_task *ctx = (pragma4527_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_32((*(ctx->ab_ptr)), (*(ctx->b_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_unshuffle_32_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m)
{
     int i;
     COMPLEX *ip;
     COMPLEX *jp;
     if ((b - a) < 128) {
	  ip = in + a * 32;
	  for (i = a; i < b; ++i) {
	       jp = out + i;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	       jp += 2 * m;
	       jp[0] = ip[0];
	       jp[m] = ip[1];
	       ip += 2;
	  }
     } else {
	  int ab = (a + b) / 2;
	  fft_unshuffle_32_seq(a, ab, in, out, m);
	  fft_unshuffle_32_seq(ab, b, in, out, m);
     }
}
/* end of machine-generated code */

/*
 * Recursive complex FFT on the n complex components of the array in:
 * basic Cooley-Tukey algorithm, with some improvements for
 * n power of two. The result is placed in the array out. n is arbitrary. 
 * The algorithm runs in time O(n*(r1 + ... + rk)) where r1, ..., rk
 * are prime numbers, and r1 * r2 * ... * rk = n.
 *
 * n: size of the input
 * in: pointer to input
 * out: pointer to output
 * factors: list of factors of n, precomputed
 * W: twiddle factors
 * nW: size of W, that is, size of the original transform
 *
 */
typedef struct _pragma4667_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4667_omp_task;

typedef struct _pragma4670_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4670_omp_task;

typedef struct _pragma4673_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4673_omp_task;

typedef struct _pragma4676_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4676_omp_task;

typedef struct _pragma4679_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4679_omp_task;

typedef struct _pragma4687_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int k;
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4687_omp_task;

typedef struct _pragma4697_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4697_omp_task;

typedef struct _pragma4700_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4700_omp_task;

typedef struct _pragma4703_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4703_omp_task;

typedef struct _pragma4706_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4706_omp_task;

typedef struct _pragma4709_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4709_omp_task;

typedef struct _pragma4712_omp_task {
    int (*r_ptr);
    int (*m_ptr);
    int (*k_ptr);
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
    int (*(*factors_ptr));
    COMPLEX (*(*W_ptr));
    int (*nW_ptr);
 } pragma4712_omp_task;

static void pragma4667_omp_task_hclib_async(void *____arg);
static void pragma4670_omp_task_hclib_async(void *____arg);
static void pragma4673_omp_task_hclib_async(void *____arg);
static void pragma4676_omp_task_hclib_async(void *____arg);
static void pragma4679_omp_task_hclib_async(void *____arg);
static void pragma4687_omp_task_hclib_async(void *____arg);
static void pragma4697_omp_task_hclib_async(void *____arg);
static void pragma4700_omp_task_hclib_async(void *____arg);
static void pragma4703_omp_task_hclib_async(void *____arg);
static void pragma4706_omp_task_hclib_async(void *____arg);
static void pragma4709_omp_task_hclib_async(void *____arg);
static void pragma4712_omp_task_hclib_async(void *____arg);
void fft_aux(int n, COMPLEX * in, COMPLEX * out, int *factors, COMPLEX * W, int nW)
{
     int r, m;
     int k;

     /* special cases */
     if (n == 32) {
	  fft_base_32(in, out);
	  return;
     }
     if (n == 16) {
	  fft_base_16(in, out);
	  return;
     }
     if (n == 8) {
	  fft_base_8(in, out);
	  return;
     }
     if (n == 4) {
	  fft_base_4(in, out);
	  return;
     }
     if (n == 2) {
	  fft_base_2(in, out);
	  return;
     }
     /* 
      * the cases n == 3, n == 5, and maybe 7 should be implemented as well
      */

     r = *factors;
     m = n / r;

     if (r < n) {
	  /* 
	   * split the DFT of length n into r DFTs of length n/r,  and
	   * recurse 
	   */
	  if (r == 32) {
 { 
pragma4667_omp_task *new_ctx = (pragma4667_omp_task *)malloc(sizeof(pragma4667_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4667_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
	  } else if (r == 16) {
 { 
pragma4670_omp_task *new_ctx = (pragma4670_omp_task *)malloc(sizeof(pragma4670_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4670_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
	  } else if (r == 8) {
 { 
pragma4673_omp_task *new_ctx = (pragma4673_omp_task *)malloc(sizeof(pragma4673_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4673_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
	  } else if (r == 4) {
 { 
pragma4676_omp_task *new_ctx = (pragma4676_omp_task *)malloc(sizeof(pragma4676_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4676_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
	  } else if (r == 2) {
 { 
pragma4679_omp_task *new_ctx = (pragma4679_omp_task *)malloc(sizeof(pragma4679_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4679_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
	  } else
	       unshuffle(0, m, in, out, r, m);

 hclib_end_finish(); hclib_start_finish(); ;

	  for (k = 0; k < n; k += m) {
 { 
pragma4687_omp_task *new_ctx = (pragma4687_omp_task *)malloc(sizeof(pragma4687_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k = k;
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4687_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
	  }
 hclib_end_finish(); hclib_start_finish(); ;
     }
     /* 
      * now multiply by the twiddle factors, and perform m FFTs
      * of length r
      */
     if (r == 2) {
 { 
pragma4697_omp_task *new_ctx = (pragma4697_omp_task *)malloc(sizeof(pragma4697_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4697_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     } else if (r == 4) {
 { 
pragma4700_omp_task *new_ctx = (pragma4700_omp_task *)malloc(sizeof(pragma4700_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4700_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     } else if (r == 8) {
 { 
pragma4703_omp_task *new_ctx = (pragma4703_omp_task *)malloc(sizeof(pragma4703_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4703_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     } else if (r == 16) {
 { 
pragma4706_omp_task *new_ctx = (pragma4706_omp_task *)malloc(sizeof(pragma4706_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4706_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     } else if (r == 32) {
 { 
pragma4709_omp_task *new_ctx = (pragma4709_omp_task *)malloc(sizeof(pragma4709_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4709_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     } else {
 { 
pragma4712_omp_task *new_ctx = (pragma4712_omp_task *)malloc(sizeof(pragma4712_omp_task));
new_ctx->r_ptr = &(r);
new_ctx->m_ptr = &(m);
new_ctx->k_ptr = &(k);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
new_ctx->factors_ptr = &(factors);
new_ctx->W_ptr = &(W);
new_ctx->nW_ptr = &(nW);
hclib_async(pragma4712_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } ;
     }

 hclib_end_finish(); hclib_start_finish(); ;

     return;
} 
static void pragma4667_omp_task_hclib_async(void *____arg) {
    pragma4667_omp_task *ctx = (pragma4667_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_32(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4670_omp_task_hclib_async(void *____arg) {
    pragma4670_omp_task *ctx = (pragma4670_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_16(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4673_omp_task_hclib_async(void *____arg) {
    pragma4673_omp_task *ctx = (pragma4673_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_8(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4676_omp_task_hclib_async(void *____arg) {
    pragma4676_omp_task *ctx = (pragma4676_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_4(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4679_omp_task_hclib_async(void *____arg) {
    pragma4679_omp_task *ctx = (pragma4679_omp_task *)____arg;
    hclib_start_finish();
fft_unshuffle_2(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4687_omp_task_hclib_async(void *____arg) {
    pragma4687_omp_task *ctx = (pragma4687_omp_task *)____arg;
    int k; k = ctx->k;
    hclib_start_finish();
fft_aux((*(ctx->m_ptr)), (*(ctx->out_ptr)) + k, (*(ctx->in_ptr)) + k, (*(ctx->factors_ptr)) + 1, (*(ctx->W_ptr)), (*(ctx->nW_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4697_omp_task_hclib_async(void *____arg) {
    pragma4697_omp_task *ctx = (pragma4697_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_2(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nW_ptr)) / (*(ctx->n_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4700_omp_task_hclib_async(void *____arg) {
    pragma4700_omp_task *ctx = (pragma4700_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_4(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nW_ptr)) / (*(ctx->n_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4703_omp_task_hclib_async(void *____arg) {
    pragma4703_omp_task *ctx = (pragma4703_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_8(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nW_ptr)) / (*(ctx->n_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4706_omp_task_hclib_async(void *____arg) {
    pragma4706_omp_task *ctx = (pragma4706_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_16(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nW_ptr)) / (*(ctx->n_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4709_omp_task_hclib_async(void *____arg) {
    pragma4709_omp_task *ctx = (pragma4709_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_32(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nW_ptr)) / (*(ctx->n_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4712_omp_task_hclib_async(void *____arg) {
    pragma4712_omp_task *ctx = (pragma4712_omp_task *)____arg;
    hclib_start_finish();
fft_twiddle_gen(0, (*(ctx->m_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->W_ptr)), (*(ctx->nW_ptr)), (*(ctx->nW_ptr)) / (*(ctx->n_ptr)), (*(ctx->r_ptr)), (*(ctx->m_ptr))) ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}



void fft_aux_seq(int n, COMPLEX * in, COMPLEX * out, int *factors, COMPLEX * W, int nW)
{
     int r, m;
     int k;

     /* special cases */
     if (n == 32) {
	  fft_base_32(in, out);
	  return;
     }
     if (n == 16) {
	  fft_base_16(in, out);
	  return;
     }
     if (n == 8) {
	  fft_base_8(in, out);
	  return;
     }
     if (n == 4) {
	  fft_base_4(in, out);
	  return;
     }
     if (n == 2) {
	  fft_base_2(in, out);
	  return;
     }
     /* 
      * the cases n == 3, n == 5, and maybe 7 should be implemented as well
      */

     r = *factors;
     m = n / r;

     if (r < n) {
	  /* 
	   * split the DFT of length n into r DFTs of length n/r,  and
	   * recurse 
	   */
	  if      (r == 32) fft_unshuffle_32_seq(0, m, in, out, m);
	  else if (r == 16) fft_unshuffle_16_seq(0, m, in, out, m);
	  else if (r ==  8) fft_unshuffle_8_seq(0, m, in, out, m);
	  else if (r ==  4) fft_unshuffle_4_seq(0, m, in, out, m);
	  else if (r ==  2) fft_unshuffle_2_seq(0, m, in, out, m);
	  else              unshuffle_seq(0, m, in, out, r, m);

	  for (k = 0; k < n; k += m) {
	       fft_aux_seq(m, out + k, in + k, factors + 1, W, nW);
	  }
     }
     /* 
      * now multiply by the twiddle factors, and perform m FFTs
      * of length r
      */
     if      (r ==  2) fft_twiddle_2_seq(0, m, in, out, W, nW, nW / n, m);
     else if (r ==  4) fft_twiddle_4_seq(0, m, in, out, W, nW, nW / n, m);
     else if (r ==  8) fft_twiddle_8_seq(0, m, in, out, W, nW, nW / n, m);
     else if (r == 16) fft_twiddle_16_seq(0, m, in, out, W, nW, nW / n, m);
     else if (r == 32) fft_twiddle_32_seq(0, m, in, out, W, nW, nW / n, m);
     else              fft_twiddle_gen_seq(0, m, in, out, W, nW, nW / n, r, m);

     return;
}
/*
 * user interface for fft_aux
 */
typedef struct _pragma4804_omp_task {
    int (*(*factors_ptr));
    int (*(*p_ptr));
    int (*l_ptr);
    int (*r_ptr);
    COMPLEX (*(*W_ptr));
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
 } pragma4804_omp_task;

typedef struct _pragma4827_omp_task {
    int (*(*factors_ptr));
    int (*(*p_ptr));
    int (*l_ptr);
    int (*r_ptr);
    COMPLEX (*(*W_ptr));
    int (*n_ptr);
    COMPLEX (*(*in_ptr));
    COMPLEX (*(*out_ptr));
 } pragma4827_omp_task;

static void pragma4804_omp_task_hclib_async(void *____arg);
static void pragma4827_omp_task_hclib_async(void *____arg);
typedef struct _main_entrypoint_ctx {
    int (*factors);
    int (*p);
    int l;
    int r;
    COMPLEX (*W);
    int n;
    COMPLEX (*in);
    COMPLEX (*out);
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    int (*factors); factors = ctx->factors;
    int (*p); p = ctx->p;
    int l; l = ctx->l;
    int r; r = ctx->r;
    COMPLEX (*W); W = ctx->W;
    int n; n = ctx->n;
    COMPLEX (*in); in = ctx->in;
    COMPLEX (*out); out = ctx->out;
{
     W = (COMPLEX *) malloc((n + 1) * sizeof(COMPLEX));
hclib_start_finish(); {
 { 
pragma4804_omp_task *new_ctx = (pragma4804_omp_task *)malloc(sizeof(pragma4804_omp_task));
new_ctx->factors_ptr = &(factors);
new_ctx->p_ptr = &(p);
new_ctx->l_ptr = &(l);
new_ctx->r_ptr = &(r);
new_ctx->W_ptr = &(W);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
hclib_async(pragma4804_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
         } ; hclib_end_finish(); 
     bots_message(" completed!\n");

     /* 
      * find factors of n, first 8, then 4 and then primes in ascending
      * order 
      */
     do {
	  r = factor(l);
	  *p++ = r;
	  l /= r;
     } while (l > 1);

     bots_message("Computing FFT ");
hclib_start_finish(); {
 { 
pragma4827_omp_task *new_ctx = (pragma4827_omp_task *)malloc(sizeof(pragma4827_omp_task));
new_ctx->factors_ptr = &(factors);
new_ctx->p_ptr = &(p);
new_ctx->l_ptr = &(l);
new_ctx->r_ptr = &(r);
new_ctx->W_ptr = &(W);
new_ctx->n_ptr = &(n);
new_ctx->in_ptr = &(in);
new_ctx->out_ptr = &(out);
hclib_async(pragma4827_omp_task_hclib_async, new_ctx, NO_FUTURE, ANY_PLACE);
 } 
         } ; hclib_end_finish(); 

     free(W);
     } ;     free(____arg);
}

void fft(int n, COMPLEX * in, COMPLEX * out)
{
    int *factors = (int *)malloc(40 * sizeof(int));
     // int factors[40];		/* allows FFTs up to at least 3^40 */
     int *p = factors;
     int l = n;
     int r;
     COMPLEX *W;


     bots_message("Computing coefficients ");
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->factors = factors;
new_ctx->p = p;
new_ctx->l = l;
new_ctx->r = r;
new_ctx->W = W;
new_ctx->n = n;
new_ctx->in = in;
new_ctx->out = out;
hclib_launch(main_entrypoint, new_ctx);

     bots_message(" completed!\n");

     return;
}  
static void pragma4804_omp_task_hclib_async(void *____arg) {
    pragma4804_omp_task *ctx = (pragma4804_omp_task *)____arg;
    hclib_start_finish();
{
                 compute_w_coefficients((*(ctx->n_ptr)), 0, (*(ctx->n_ptr)) / 2, (*(ctx->W_ptr)));
             } ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


static void pragma4827_omp_task_hclib_async(void *____arg) {
    pragma4827_omp_task *ctx = (pragma4827_omp_task *)____arg;
    hclib_start_finish();
{
                 fft_aux((*(ctx->n_ptr)), (*(ctx->in_ptr)), (*(ctx->out_ptr)), (*(ctx->factors_ptr)), (*(ctx->W_ptr)), (*(ctx->n_ptr)));
             } ;     ; hclib_end_finish_nonblocking();

    free(____arg);
}


void fft_seq(int n, COMPLEX * in, COMPLEX * out)
{
     int factors[40];		/* allows FFTs up to at least 3^40 */
     int *p = factors;
     int l = n;
     int r;
     COMPLEX *W;

     W = (COMPLEX *) malloc((n + 1) * sizeof(COMPLEX));
     compute_w_coefficients_seq(n, 0, n / 2, W);

     /* 
      * find factors of n, first 8, then 4 and then primes in ascending
      * order 
      */
     do {
	  r = factor(l);
	  *p++ = r;
	  l /= r;
     } while (l > 1);

     fft_aux_seq(n, in, out, factors, W, n);

     free(W);
     return;
}
int test_correctness(int n, COMPLEX *out1, COMPLEX *out2)
{
  int i;
  double a,d,error = 0.0;

  for (i = 0; i < n; ++i) {
       a = sqrt((c_re(out1[i]) - c_re(out2[i])) *
		(c_re(out1[i]) - c_re(out2[i])) +
		(c_im(out1[i]) - c_im(out2[i])) *
		(c_im(out1[i]) - c_im(out2[i])));
       d =  sqrt(c_re(out2[i]) * c_re(out2[i]) + 
		 c_im(out2[i]) * c_im(out2[i]));
       if (d < -1.0e-10 || d > 1.0e-10) a /= d;
       if (a > error) error = a;
  }
  bots_message("relative error=%e\n", error);
  if (error > 1e-3) return BOTS_RESULT_UNSUCCESSFUL;
  else return BOTS_RESULT_SUCCESSFUL;
}

