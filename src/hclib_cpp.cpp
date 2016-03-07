#include "hclib_cpp.h"

hclib::promise_t **hclib::promise_create_n(const size_t nb_promises,
        const int null_terminated) {
    hclib::promise_t **promises = (hclib::promise_t **)malloc(
                                      (null_terminated ? nb_promises + 1 : nb_promises) *
                                      sizeof(hclib::promise_t *));
    for (unsigned i = 0; i < nb_promises; i++) {
        promises[i] = new promise_t();
    }

    if (null_terminated) {
        promises[nb_promises] = NULL;
    }
    return promises;
}

hclib_worker_state *hclib::current_ws() {
    return CURRENT_WS_INTERNAL;
}

int hclib::get_current_worker() {
    return hclib_get_current_worker();
}

int hclib::num_workers() {
    return hclib_num_workers();
}

int hclib::get_num_locales() {
    return hclib_get_num_locales();
}

hclib::hclib_locale *hclib::get_closest_locale() {
    return hclib_get_closest_locale();
}

hclib::hclib_locale *hclib::get_all_locales() {
    return hclib_get_all_locales();
}
