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
    constexpr int SIGNAL_VALUE = 42;
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, []() {
        hclib::finish([]() {
            hclib::promise_t<int> *event = new hclib::promise_t<int>();
            hclib::async([=]() {
                    int signal = event->get_future()->wait();
                    assert(signal == SIGNAL_VALUE);
                    printf("signal = %d\n", signal);
                });
            hclib::async([=]() {
                    sleep(1);
                    event->put(SIGNAL_VALUE);
                });
        });
    });
    printf("Exiting...\n");
    return 0;
}
