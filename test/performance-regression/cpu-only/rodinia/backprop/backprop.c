#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#define OPEN

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

/*** Return random number between 0.0 and 1.0 ***/
float drnd()
{
  return ((float) rand() / (float) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
float dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(float x)
{
  float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(int n)
{
  float *new_alloc;

  new_alloc = (float *) malloc ((unsigned) (n * sizeof (float)));
  if (new_alloc == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return (new_alloc);
}


/*** Allocate 2d array of floats ***/

float **alloc_2d_dbl(int m, int n)
{
  int i;
  float **new_alloc;

  new_alloc = (float **) malloc ((unsigned) (m * sizeof (float *)));
  if (new_alloc == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    new_alloc[i] = alloc_1d_dbl(n);
  }

  return (new_alloc);
}


void bpnn_randomize_weights(float **w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[i][j] = (float) rand()/RAND_MAX;
    //  w[i][j] = dpn1();
    }
  }
}

void bpnn_randomize_row(float *w, int m)
{
	int i;
	for (i = 0; i <= m; i++) {
     //w[i] = (float) rand()/RAND_MAX;
	 w[i] = 0.1;
    }
}


void bpnn_zero_weights(float **w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}


void bpnn_initialize(int seed)
{
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}


BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}


void bpnn_free(BPNN *net)
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i <= n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(int n_in, int n_hidden, int n_out)
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}


typedef struct _pragma242_omp_parallel {
    float sum;
    int j;
    int k;
    float (*(*l1_ptr));
    float (*(*l2_ptr));
    float (*(*(*conn_ptr)));
    int (*n1_ptr);
    int (*n2_ptr);
    pthread_mutex_t reduction_mutex;
 } pragma242_omp_parallel;

static void pragma242_omp_parallel_hclib_async(void *____arg, const int ___iter0);
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2)
{
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
 { 
pragma242_omp_parallel *new_ctx = (pragma242_omp_parallel *)malloc(sizeof(pragma242_omp_parallel));
new_ctx->sum = sum;
new_ctx->j = j;
new_ctx->k = k;
new_ctx->l1_ptr = &(l1);
new_ctx->l2_ptr = &(l2);
new_ctx->conn_ptr = &(conn);
new_ctx->n1_ptr = &(n1);
new_ctx->n2_ptr = &(n2);
new_ctx->sum = 0;
const int init_err = pthread_mutex_init(&new_ctx->reduction_mutex, NULL);
assert(init_err == 0);
hclib_loop_domain_t domain[1];
domain[0].low = 1;
domain[0].high = (n2) + 1;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma242_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
free(new_ctx);
sum = new_ctx->sum;
 } 
} 
static void pragma242_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma242_omp_parallel *ctx = (pragma242_omp_parallel *)____arg;
    float sum; sum = ctx->sum;
    int j; j = ctx->j;
    int k; k = ctx->k;
    hclib_start_finish();
    do {
    j = ___iter0;
{

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= (*(ctx->n1_ptr)); k++) {	
      sum += (*(ctx->conn_ptr))[k][j] * (*(ctx->l1_ptr))[k]; 
    }
    (*(ctx->l2_ptr))[j] = squash(sum);
  } ;     } while (0);
    const int lock_err = pthread_mutex_lock(&ctx->reduction_mutex);
    assert(lock_err == 0);
    ctx->sum += sum;
    const int unlock_err = pthread_mutex_unlock(&ctx->reduction_mutex);
    assert(unlock_err == 0);
    ; hclib_end_finish_nonblocking();

}



//extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err)  
{
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}


void bpnn_hidden_error(float *delta_h,   
					   int nh, 
					   float *delta_o, 
					   int no, 
					   float **who, 
					   float *hidden, 
					   float *err)
{
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}


typedef struct _pragma304_omp_parallel {
    float new_dw;
    int k;
    int j;
    float (*(*delta_ptr));
    int ndelta;
    float (*(*ly_ptr));
    int nly;
    float (*(*(*w_ptr)));
    float (*(*(*oldw_ptr)));
 } pragma304_omp_parallel;

static void pragma304_omp_parallel_hclib_async(void *____arg, const int ___iter0);
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw)
{
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  //eta = 0.3;
  //momentum = 0.3;

 { 
pragma304_omp_parallel *new_ctx = (pragma304_omp_parallel *)malloc(sizeof(pragma304_omp_parallel));
new_ctx->new_dw = new_dw;
new_ctx->k = k;
new_ctx->j = j;
new_ctx->delta_ptr = &(delta);
new_ctx->ndelta = ndelta;
new_ctx->ly_ptr = &(ly);
new_ctx->nly = nly;
new_ctx->w_ptr = &(w);
new_ctx->oldw_ptr = &(oldw);
hclib_loop_domain_t domain[1];
domain[0].low = 1;
domain[0].high = (ndelta) + 1;
domain[0].stride = 1;
domain[0].tile = -1;
hclib_future_t *fut = hclib_forasync_future((void *)pragma304_omp_parallel_hclib_async, new_ctx, 1, domain, HCLIB_FORASYNC_MODE);
hclib_future_wait(fut);
free(new_ctx);
 } 
} 
static void pragma304_omp_parallel_hclib_async(void *____arg, const int ___iter0) {
    pragma304_omp_parallel *ctx = (pragma304_omp_parallel *)____arg;
    float new_dw; new_dw = ctx->new_dw;
    int k; k = ctx->k;
    int j; j = ctx->j;
    int ndelta; ndelta = ctx->ndelta;
    int nly; nly = ctx->nly;
    do {
    j = ___iter0;
{
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * (*(ctx->delta_ptr))[j] * (*(ctx->ly_ptr))[k]) + (MOMENTUM * (*(ctx->oldw_ptr))[k][j]));
	  (*(ctx->w_ptr))[k][j] += new_dw;
	  (*(ctx->oldw_ptr))[k][j] = new_dw;
    }
  } ;     } while (0);
}




void bpnn_feedforward(BPNN *net)
{
  int in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

}


typedef struct _main_entrypoint_ctx {
    BPNN (*net);
    float (*eo);
    float (*eh);
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    BPNN (*net); net = ctx->net;
    float (*eo); eo = ctx->eo;
    float (*eh); eh = ctx->eh;
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(net->output_delta, net->target, net->output_units,
      out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
      net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
      net->input_weights, net->input_prev_weights);
    } ;     free(____arg);
}

void bpnn_train(BPNN *net, float *eo, float *eh)
{
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->net = net;
new_ctx->eo = eo;
new_ctx->eh = eh;
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);


} 


void bpnn_save(BPNN *net, char *filename)
{
  int n1, n2, n3, i, j, memcnt;
  float dvalue, **w;
  char *mem;
  ///add//
  FILE *pFile;
  pFile = fopen( filename, "w+" );
  ///////
  /*
  if ((fd = creat(filename, 0644)) == -1) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }
  */

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  //fflush(stdout);

  //write(fd, (char *) &n1, sizeof(int));
  //write(fd, (char *) &n2, sizeof(int));
  //write(fd, (char *) &n3, sizeof(int));

  fwrite( (char *) &n1 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n2 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n3 , sizeof(char), sizeof(char), pFile);

  

  memcnt = 0;
  w = net->input_weights;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  fwrite( mem , (unsigned)(sizeof(float)), (unsigned) ((n1+1) * (n2+1) * sizeof(float)) , pFile);
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  fwrite( mem , sizeof(float), (unsigned) ((n2+1) * (n3+1) * sizeof(float)) , pFile);
  free(mem);

  fclose(pFile);
  return;
}


BPNN *bpnn_read(char *filename)
{
  char *mem;
  BPNN *new_alloc;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return (NULL);
  }

  printf("Reading '%s'\n", filename);  //fflush(stdout);

  read(fd, (char *) &n1, sizeof(int));
  read(fd, (char *) &n2, sizeof(int));
  read(fd, (char *) &n3, sizeof(int));
  new_alloc = bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  read(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(new_alloc->input_weights[i][j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);

  printf("Done\nReading hidden weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  read(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(new_alloc->hidden_weights[i][j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);
  close(fd);

  printf("Done\n");  //fflush(stdout);

  bpnn_zero_weights(new_alloc->input_prev_weights, n1, n2);
  bpnn_zero_weights(new_alloc->hidden_prev_weights, n2, n3);

  return (new_alloc);
}
