#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib_common.h"
#include "hcpp-rt.h"
#include "hcpp-async.h"
#include "hcpp-asyncAwait.h"
#include "hcpp-forasync.h"

namespace hclib {

typedef hclib_ddf_t ddf_t;
typedef loop_domain_t loop_domain_t;
typedef place_t place_t;
typedef place_type_t place_type_t;

template <typename T>
void launch(int *argc, char **argv, T lambda) {
    task_t *user_task = _allocate_async(lambda, false);
    hclib_launch(argc, argv, (generic_framePtr)spawn, user_task);
}

void start_finish();
void end_finish();

ddf_t *ddf_create();
void ddf_free(ddf_t *ddf);
void ddf_put(ddf_t *ddf, void *datum);
void *ddf_get(ddf_t *ddf);
void *ddf_wait(ddf_t *ddf);

hc_workerState *current_ws();
int current_worker();
int num_workers();
int get_num_places(place_type_t type);
void get_places(place_t **pls, place_type_t type);

}

#endif
