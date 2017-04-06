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

#include "hclib-async-struct.h"
#include "hclib.h"

#ifndef HCSHMEM_SUPPORT_H_
#define HCSHMEM_SUPPORT_H_

#ifdef HCSHEMM
#ifdef __cplusplus
extern "C" {
#endif
void hclib_gather_comm_worker_stats(int* push_outd, int* push_ind, int* steal_ind);
int totalPendingLocalAsyncs();
void hclib_display_runtime();
volatile int* hclib_start_finish_special();
#ifdef __cplusplus
}
#endif
#endif

#endif //HCSHMEM_SUPPORT_H_
