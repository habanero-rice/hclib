/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib.h"
#include "hclib_common.h"
#include "hclib-rt.h"
#include "hclib-async.hpp"
#include "hclib-forasync.hpp"
#include "hclib-promise.hpp"

namespace hclib {

typedef loop_domain_t loop_domain_t;
typedef place_t place_t;
typedef place_type_t place_type_t;

template <typename T>
inline void launch(T &&lambda) {
    hclib_launch(lambda_wrapper<T>, new T(lambda));
}

extern hclib_worker_state *current_ws();
int current_worker();
int num_workers();
int get_num_places(place_type_t type);
void get_places(place_t **pls, place_type_t type);
place_t *get_current_place();
place_t **get_children_of_place(place_t *pl, int *num_children);
place_t *get_root_place();
char *get_place_name(place_t *pl);

}

#endif
