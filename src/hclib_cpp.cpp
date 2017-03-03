#include "hclib_cpp.h"
#include "hclib_future.h"

hclib_worker_state *hclib::current_ws() {
    return CURRENT_WS_INTERNAL;
}

int hclib::get_current_worker() {
    return hclib_get_current_worker();
}

int hclib::get_num_workers() {
    return hclib_get_num_workers();
}

int hclib::get_num_locales() {
    return hclib_get_num_locales();
}

hclib::locale_t *hclib::get_closest_locale() {
    return hclib_get_closest_locale();
}

hclib::locale_t **hclib::get_thread_private_locales() {
    return hclib_get_thread_private_locales();
}

hclib::locale_t *hclib::get_all_locales() {
    return hclib_get_all_locales();
}

hclib_locale_t **hclib::get_all_locales_of_type(int type, int *out_count) {
    return hclib_get_all_locales_of_type(type, out_count);
}

hclib::future_t<void*> *hclib::allocate_at(size_t nbytes, hclib::locale_t *locale) {
    hclib_future_t *actual = hclib_allocate_at(nbytes, locale);
    return (hclib::future_t<void*> *)actual;
}

hclib::future_t<void*> *hclib::reallocate_at(void *ptr, size_t nbytes,
        hclib::locale_t *locale) {
    hclib_future_t *actual = hclib_reallocate_at(ptr, nbytes, locale);
    return (hclib::future_t<void*> *)actual;
}

void hclib::free_at(void *ptr, hclib::locale_t *locale) {
    hclib_free_at(ptr, locale);
}

hclib::future_t<void*> *hclib::memset_at(void *ptr, int pattern, size_t nbytes,
        hclib::locale_t *locale) {
    hclib_future_t *actual = hclib_memset_at(ptr, pattern, nbytes, locale);
    return (hclib::future_t<void*> *)actual;
}
