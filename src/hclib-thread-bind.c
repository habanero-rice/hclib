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

#include <stdio.h>
#define _GNU_SOURCE
#define __USE_GNU
#include <xlocale.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <stdlib.h>
/** Platform specific thread binding implementations -- > ONLY FOR LINUX **/

#include "hclib-rt.h"

#ifdef __linux
int get_nb_cpus() {
    int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    return numCPU;
}

void bind_thread_with_mask(int *mask, int lg) {
    cpu_set_t cpuset;
    if (mask != NULL) {
        CPU_ZERO(&cpuset);

        /* Copy the mask from the int array to the cpuset */
        int i;
        for (i = 0; i < lg; i++) {
            CPU_SET(mask[i], &cpuset);
        }

        /* Set affinity */
        int res = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        if (res != 0) {
            printf("ERROR: ");
            if (errno == ESRCH) {
                HASSERT("THREADBINDING ERROR: ESRCH: Process not found!\n");
            }
            if (errno == EINVAL) {
                HASSERT("THREADBINDING ERROR: EINVAL: CPU mask does not contain any actual physical processor\n");
            }
            if (errno == EFAULT) {
                HASSERT("THREADBINDING ERROR: EFAULT: memory address was invalid\n");
            }
            if (errno == EPERM) {
                HASSERT("THREADBINDING ERROR: EPERM: process does not have appropriate privileges\n");
            }
        }
    }
}

/* Bind threads in a round-robin fashion */
void bind_thread_rr(int worker_id) {
    /*bind worker_id to cpu_id round-robin fashion*/
    int nbCPU = get_nb_cpus();
    int mask = worker_id % nbCPU;
    //printf("HCLIB: INFO -- Binding worker %d to cpu_id %d\n", worker_id, mask);
    bind_thread_with_mask(&mask, 1);
}

/* Bind threads according to bind map */
void bind_thread_map(int worker_id, int *bind_map, int bind_map_size) {
    int mask = bind_map[worker_id % bind_map_size];
    //printf("HCLIB: INFO -- Binding worker %d to cpu_id %d\n", worker_id, mask);
    bind_thread_with_mask(&mask, 1);
}

/** Thread binding api to bind a worker thread using a particular binding strategy **/
void bind_thread(int worker_id, int *bind_map, int bind_map_size) {
    if (bind_map_size == 0) {
        /* Round robin binding */
        bind_thread_rr(worker_id);
    } else if (bind_map_size > 0 && bind_map != NULL) {
        bind_thread_map(worker_id, bind_map, bind_map_size);
    } else {
        fprintf(stderr, "ERROR: unknown thread binding strategy\n");
        HASSERT(0);
    }
}
#else
void bind_thread(int worker_id, int *bind_map, int bind_map_size) {

}
#endif


