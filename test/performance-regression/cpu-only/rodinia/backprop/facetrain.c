#include "hclib.h"
#ifdef __cplusplus
#include "hclib_cpp.h"
#include "hclib_system.h"
#ifdef __CUDACC__
#include "hclib_cuda.h"
#endif
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

typedef struct _main_entrypoint_ctx {
    BPNN (*net);
    int i;
 } main_entrypoint_ctx;


static void main_entrypoint(void *____arg) {
    main_entrypoint_ctx *ctx = (main_entrypoint_ctx *)____arg;
    BPNN (*net); net = ctx->net;
    int i; i = ctx->i;
{
    float out_err, hid_err;
    bpnn_train_kernel(net, &out_err, &hid_err);
  } ;     free(____arg);
}

void backprop_face()
{
  BPNN *net;
  int i;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
main_entrypoint_ctx *new_ctx = (main_entrypoint_ctx *)malloc(sizeof(main_entrypoint_ctx));
new_ctx->net = net;
new_ctx->i = i;
const char *deps[] = { "system" };
hclib_launch(main_entrypoint, new_ctx, deps, 1);

  bpnn_free(net);
  printf("Training done\n");
} 

int setup(int argc, char *argv[])
{
  if(argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }

  layer_size = atoi(argv[1]);
  
  int seed;

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
