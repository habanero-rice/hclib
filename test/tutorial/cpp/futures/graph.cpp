#include "hclib_cpp.h"
#include <unistd.h>

/*
 * This program is computing the following task graph using futures
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
    hclib::future_t<void> *a = hclib::async_future([=]() {
      sleep(1);
      printf("A\n");
      return;
    });
    hclib::future_t<void> *b = hclib::async_future([=]() {
      a->get();
      printf("B\n");
      return;
    });
    hclib::future_t<void> *c = hclib::async_future([=]() {
      a->get();
      printf("C\n");
      return;
    });
    hclib::future_t<void> *d = hclib::async_future([=]() {
      sleep(1);
      b->get();
      c->get();
      printf("D\n");
      return;
    });
    hclib::future_t<void> *e = hclib::async_future([=]() {
      c->get();
      printf("E\n");
      return;
    });
    hclib::future_t<void> *f = hclib::async_future([=]() {
      d->get();
      e->get();
      printf("F\n");
      return;
    });
    f->get();
    printf("Terminating\n");
  });
  return 0;
}
