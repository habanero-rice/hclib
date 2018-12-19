#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

#define PUT_VALUE 3

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [=]() {
        hclib::finish([=]() {
            hclib::promise_t<int> *prom = new hclib::promise_t<int>();
            hclib::locale_t *locale = hclib::get_closest_locale();

            // auto lambda = ;
            hclib::future_t<int> *fut = hclib::async_future_await_at(
                [prom, locale]() -> int { return prom->get_future()->get() + 1; }, prom->get_future(), locale);

            prom->put(PUT_VALUE);
            int result = fut->wait();
            assert(result == PUT_VALUE + 1);
        });
    });

    return 0;
}

