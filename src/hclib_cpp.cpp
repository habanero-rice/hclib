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

hclib::locale_t *hclib::get_closest_locale() {
    return hclib_get_closest_locale();
}

hclib::locale_t *hclib::get_all_locales() {
    return hclib_get_all_locales();
}

hclib_future_t *hclib::allocate_at(size_t nbytes, hclib::locale_t *locale) {
    return hclib_allocate_at(nbytes, locale);
}

hclib_future_t *hclib::reallocate_at(void *ptr, size_t nbytes,
        hclib::locale_t *locale) {
    return hclib_reallocate_at(ptr, nbytes, locale);
}

void hclib::free_at(void *ptr, hclib::locale_t *locale) {
    hclib_free_at(ptr, locale);
}

hclib_future_t *hclib::memset_at(void *ptr, int pattern, size_t nbytes,
        hclib::locale_t *locale) {
    return hclib_memset_at(ptr, pattern, nbytes, locale);
}

hclib_future_t *hclib::async_copy(hclib::locale_t *dst_locale, void *dst,
        hclib::locale_t *src_locale, void *src, size_t nbytes) {
    return hclib_async_copy(dst_locale, dst, src_locale, src, nbytes);
}
