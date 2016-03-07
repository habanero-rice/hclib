#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib_common.h"
#include "hclib-rt.h"
#include "hclib-async.h"
#include "hclib-forasync.h"
#include "hclib_promise.h"
#include "hclib.h"

namespace hclib {

typedef hclib_triggered_task_t triggered_task_t;
typedef loop_domain_t loop_domain_t;
typedef hclib_locale hclib_locale;

template <typename T>
void launch(int *argc, char **argv, T lambda) {
    hclib_task_t *user_task = _allocate_async(lambda, false);
    hclib_launch(argc, argv, (generic_frame_ptr)spawn, user_task);
}

promise_t **promise_create_n(size_t nb_promises, int null_terminated);

extern hclib_worker_state *current_ws();
int current_worker();
int num_workers();
}

#endif
