#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hcpp-rt.h"

#include "hcpp-async.h"
#include "hcpp-asyncAwait.h"
#include "hcpp-forasync.h"

namespace hclib {

typedef hclib_ddf_t ddf_t;

void init(int *argc, char **argv);
void finalize();
void start_finish();
void end_finish();

ddf_t *ddf_create();
void ddf_put(ddf_t *ddf, void *datum);
void *ddf_get(ddf_t *ddf);

}

#endif
