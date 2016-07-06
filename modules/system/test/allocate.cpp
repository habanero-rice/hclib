#include "hclib_cpp.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    const char *deps[] = { "system" };
    hclib::launch(deps, 1, [] {
        int i;
        int N = 1024;

        hclib::locale_t *locale = hclib::get_closest_locale();
        hclib::future_t *fut = hclib::allocate_at(N * sizeof(double), locale);
        double *alloc = (double *)fut->wait();
        assert(alloc);
        for (i = 0; i < N; i++) alloc[i] = i;

        fut = hclib::reallocate_at(alloc, 2 * N * sizeof(double), locale);
        double *new_alloc = (double *)fut->wait();
        assert(new_alloc);
        for (i = 0; i < N; i++) assert(new_alloc[i] == i);
        for (i = N; i < 2 * N; i++) new_alloc[i] = i;

        fut = hclib::memset_at(new_alloc, 0x00, 2 * N * sizeof(double), locale);
        fut->wait();
        for (i = 0; i < 2 * N; i++) assert(new_alloc[i] == 0);

        fut = hclib::allocate_at(2 * N * sizeof(double), locale);
        double *other = (double *)fut->wait();
        assert(other);
        for (i = 0; i < 2 * N; i++) other[i] = i;

        fut = hclib::async_copy(locale, new_alloc, locale, other,
                2 * N * sizeof(double));
        fut->wait();
        for (i = 0; i < 2 * N; i++) assert(new_alloc[i] == i);

        hclib::free_at(new_alloc, locale);
    });
    return 0;
}
