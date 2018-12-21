#include "hclib_cpp.h"
#include <unistd.h>

/*
 * This program is computing the following task graph using promises, futures and async_await
 *
 *       (A)
 *       /\
 *      /  \
 *    (B)  (C)
 *     \    /\
 *      \  /  \
 *       (D)  (E)
 *         \   /
 *          \ /
 *          (F)
 */

int main(int argc, char **argv) {
  char const *deps[] = { "system" }; 
  hclib::launch(deps, 1, [&]() {
    hclib::promise_t<void> *prom_A = new hclib::promise_t<void>();
    hclib::promise_t<void> *prom_B = new hclib::promise_t<void>();
    hclib::promise_t<void> *prom_C = new hclib::promise_t<void>();
    hclib::promise_t<void> *prom_D = new hclib::promise_t<void>();
    hclib::promise_t<void> *prom_E = new hclib::promise_t<void>();
    hclib::promise_t<void> *prom_F = new hclib::promise_t<void>();
    hclib::finish([=]() {
      hclib::async([=]() { printf("A\n"); prom_A->put(); });
      hclib::async_await([=]() { printf("B\n"); prom_B->put(); }, prom_A->get_future());
      hclib::async_await([=]() { printf("C\n"); prom_C->put(); }, prom_A->get_future());
      hclib::async_await([=]() { printf("D\n"); prom_D->put(); }, prom_B->get_future(), prom_C->get_future());
      hclib::async_await([=]() { printf("E\n"); prom_E->put(); }, prom_C->get_future());
      hclib::async_await([=]() { printf("F\n"); prom_F->put(); }, prom_D->get_future(), prom_E->get_future());
    });
    printf("Terminating\n");
  });
  return 0;
}
