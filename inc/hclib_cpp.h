#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib_common.h"
#include "hclib-rt.h"
#include "hclib-async.h"
#include "hclib-forasync.h"
#include "hclib_promise.h"
#include "hclib.h"
#include "hclib-locality-graph.h"

namespace hclib {

typedef hclib_triggered_task_t triggered_task_t;
typedef hclib_locale hclib_locale;

template <typename T>
void launch(T lambda) {
    hclib_task_t *user_task = _allocate_async(lambda, false);
    hclib_launch((generic_frame_ptr)spawn, user_task);
}

promise_t **promise_create_n(size_t nb_promises, int null_terminated);

extern hclib_worker_state *current_ws();
int get_current_worker();
int num_workers();

int get_num_locales();
hclib_locale *get_closest_locale();
hclib_locale *get_all_locales();
}

#endif
