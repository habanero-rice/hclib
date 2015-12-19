#include "hclib_cpp.h"

hclib::ddf_t *hclib::ddf_create() {
    return hclib_ddf_create();
}

void hclib::ddf_free(hclib::ddf_t *ddf) {
    hclib_ddf_free(ddf);
}

void hclib::ddf_put(hclib::ddf_t *ddf, void *datum) {
    hclib_ddf_put(ddf, datum);
}

void *hclib::ddf_get(hclib::ddf_t *ddf) {
    return hclib_ddf_get(ddf);
}

hc_workerState *hclib::current_ws() {
    return CURRENT_WS_INTERNAL;
}

int hclib::current_worker() {
    return get_current_worker();
}

int hclib::num_workers() {
    return hclib_num_workers();
}

int hclib::get_num_places(hclib::place_type_t type) {
    return hclib_get_num_places(type);
}

void hclib::get_places(hclib::place_t **pls, hclib::place_type_t type) {
    hclib_get_places(pls, type);
}

place_t *hclib::get_current_place() {
    return hclib_get_current_place();
}

place_t **hclib::get_children_of_place(place_t *pl, int *num_children) {
    return hclib_get_children_of_place(pl, num_children);
}

place_t *hclib::get_root_place() {
    return hclib_get_root_place();
}

place_t **hclib::get_nvgpu_places(int *n_nvgpu_places) {
    return hclib_get_nvgpu_places(n_nvgpu_places);
}

char *hclib::get_place_name(place_t *pl) {
    return hclib_get_place_name(pl);
}

void *hclib::ddf_wait(hclib::ddf_t *ddf) {
    return hclib_ddf_wait(ddf);
}
