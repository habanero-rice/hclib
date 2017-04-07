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

#include <hwloc.h>
#include <stdio.h>
#include "common.h"

/*
 * os_index is unique in a given layer of the hierarchy (it seems), but
 * definitely not unique across all hwloc objects.
 */
static void print_hwloc_obj(hwloc_obj_t obj, int indent) {
    for (int i = 0; i < indent; i++) {
        printf("|  ");
    }
    /*
     * Unforunately, it seems that on many platforms hwloc does not provide
     * distances between child devices (i.e. distances_count == 0).
     */
    printf("obj %u (type=%s , depth=%d, %d child/children, parent=%u, name=%s, "
            "distance count=%u, infos_count=%u)\n", obj->os_index, str_for_type(obj->type),
            obj->depth, obj->arity, obj->parent ? obj->parent->os_index : 0,
            obj->name ? obj->name : "", obj->distances_count, obj->infos_count);
    if (obj->infos_count > 0) {
        for (int i = 0; i < indent + 2; i++) {
            printf("--");
        }
        struct hwloc_obj_info_s * infos = obj->infos;
        for (int i = 0; i < obj->infos_count; i++) {
            if (i != 0) printf(", ");
            printf("%s=%s", infos[i].name, infos[i].value);
        }
        printf("\n");
    }

    for (int i = 0; i < obj->arity; i++) {
        print_hwloc_obj(obj->children[i], indent + 1);
    }
}

int main(void) {
    hwloc_topology_t topology;

    hwloc_topology_init(&topology);  // initialization
    hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
    hwloc_topology_load(topology);   // actual detection

    hwloc_obj_t root = hwloc_get_root_obj(topology);

    print_hwloc_obj(root, 0);

    hwloc_topology_destroy(topology);

    return 0;
}
