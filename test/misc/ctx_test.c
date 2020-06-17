#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

//#define VERBOSE

#include <litectx.h>

LiteCtx *mainCtx;
LiteCtx* ctx1;
LiteCtx* ctx2;

volatile int value = 0;

void printStack(LiteCtx* ctx) {
  int i;
  
  ++value;

  
  printf("\n%s\n", (const char*)ctx->arg1);
  printf("  prev context was %p\n", ctx->prev);
  printf("  address from current stack: %p\n", &i);
}

void func1(LiteCtx* ctx) {
  LiteCtx* prev = ctx->prev;
  
  printStack(ctx);
  
  LiteCtx_swap(ctx, ctx2, __func__);
  
  printStack(ctx);
  
  LiteCtx_swap(ctx, prev, __func__);
}

void func2(LiteCtx* ctx) {
  LiteCtx* prev = ctx->prev;
  
  printStack(ctx);
  
  LiteCtx_swap(ctx, prev, __func__);
}

LiteCtx* makeCtx(void (*fn)(LiteCtx*), const char* fn_str) {
  LiteCtx*ctx = LiteCtx_create(fn);
  
  printf("Created context for %s: %p\n", fn_str, ctx);
  printf("  fctx: %p\n", ctx->_fctx);
  char *const stack_top = ctx->_stack + LITECTX_STACK_SIZE;
  printf("  stack: %p-%p\n",ctx->_stack, stack_top);
  
  ctx->arg1 = (void *)fn_str;
  
  return ctx;
}

int main() {
//  dup2(1, 2);
  fclose(stderr);
  stderr = fdopen(1, "a");

  mainCtx = LiteCtx_proxy_create(__func__);
  mainCtx->arg1 = (void *)__func__;
  
  printf("Created proxy context for main: %p\n", mainCtx);
  printf("  fctx: %p\n", mainCtx->_fctx);
  
  ctx1 = makeCtx(func1, "func1");
  ctx2 = makeCtx(func2, "func2");
  
  printStack(mainCtx);
  
  LiteCtx_swap(mainCtx, ctx1, __func__);
  
  printStack(mainCtx);
  
  if (value == 5) {
    printf("\npassed\n");
    return 0;
  } else {
    printf("\nfailed\n");
    return 1;
  }
}
