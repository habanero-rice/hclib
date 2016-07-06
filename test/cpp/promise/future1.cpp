/*
 *  RICE University
 *  Habanero Team
 *  
 *  This file is part of HC Test.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "hclib_cpp.h"

int main(int argc, char ** argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, []() {
        hclib::finish([]() {
            hclib::promise_t *event = new hclib::promise_t();
            hclib::async([=]() {
                    int *signal = (int *)event->get_future()->wait();
                    assert(*signal == 42);
                    printf("signal = %d\n", *signal);
                });
            hclib::async([=]() {
                    int *signal = (int *)malloc(sizeof(int));
                    assert(signal);
                    *signal = 42;

                    sleep(5);
                    event->put(signal);
                });
        });
    });
    printf("Exiting...\n");
    return 0;
}
