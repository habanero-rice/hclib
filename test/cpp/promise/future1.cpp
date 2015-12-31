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
    hclib::launch(&argc, argv, []() {
        hclib::finish([]() {
            hclib::promise_t *event = hclib::promise_create();
            hclib::async([=]() {
                    int *signal = (int *)hclib::promise_wait(event);
                    assert(*signal == 42);
                    printf("signal = %d\n", *signal);
                });
            hclib::async([=]() {
                    int *signal = (int *)malloc(sizeof(int));
                    assert(signal);
                    *signal = 42;

                    sleep(5);
                    hclib::promise_put(event, signal);
                });
        });
    });
    printf("Exiting...\n");
    return 0;
}
