#include "hclib_cpp.h"

namespace hclib {

void init(int *argc, char **argv) {
    hclib_init(argc, argv);
}

void finalize() {
    hclib_finalize();
}

void start_finish() {
    hclib_start_finish();
}

void end_finish() {
    hclib_end_finish();
}

ddf_t *ddf_create() {
    return hclib_ddf_create();
}

void ddf_put(ddf_t *ddf, void *datum) {
    hclib_ddf_put(ddf, datum);
}

void *ddf_get(ddf_t *ddf) {
    return hclib_ddf_get(ddf);
}

}
