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

#include "common.h"

const char *str_for_type(hwloc_obj_type_t type) {
    switch (type) {
        case (HWLOC_OBJ_SYSTEM):
            return HWLOC_OBJ_SYSTEM_STR;
        case (HWLOC_OBJ_MACHINE):
            return (HWLOC_OBJ_MACHINE_STR);
        case (HWLOC_OBJ_NUMANODE):
            return (HWLOC_OBJ_NUMANODE_STR);
        case (HWLOC_OBJ_PACKAGE):
            return (HWLOC_OBJ_PACKAGE_STR);
        case (HWLOC_OBJ_CACHE):
            return (HWLOC_OBJ_CACHE_STR);
        case (HWLOC_OBJ_CORE):
            return (HWLOC_OBJ_CORE_STR);
        case (HWLOC_OBJ_PU):
            return (HWLOC_OBJ_PU_STR);
        case (HWLOC_OBJ_GROUP):
            return (HWLOC_OBJ_GROUP_STR);
        case (HWLOC_OBJ_MISC):
            return (HWLOC_OBJ_MISC_STR);
        case (HWLOC_OBJ_BRIDGE):
            return (HWLOC_OBJ_BRIDGE_STR);
        case (HWLOC_OBJ_PCI_DEVICE):
            return (HWLOC_OBJ_PCI_DEVICE_STR);
        case (HWLOC_OBJ_OS_DEVICE):
            return (HWLOC_OBJ_OS_DEVICE_STR);
        default:
            fprintf(stderr, "Unsupported type %d\n", type);
            exit(1);
    }
}

