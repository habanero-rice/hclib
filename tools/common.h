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

#ifndef COMMON_H
#define COMMON_H

#include <hwloc.h>

static const char* HWLOC_OBJ_SYSTEM_STR     = "HWLOC_OBJ_SYSTEM";
static const char* HWLOC_OBJ_MACHINE_STR    = "HWLOC_OBJ_MACHINE";
static const char* HWLOC_OBJ_NUMANODE_STR   = "HWLOC_OBJ_NUMANODE";
static const char* HWLOC_OBJ_PACKAGE_STR    = "HWLOC_OBJ_PACKAGE";
static const char* HWLOC_OBJ_CACHE_STR      = "HWLOC_OBJ_CACHE";
static const char* HWLOC_OBJ_CORE_STR       = "HWLOC_OBJ_CORE";
static const char* HWLOC_OBJ_PU_STR         = "HWLOC_OBJ_PU";
static const char* HWLOC_OBJ_GROUP_STR      = "HWLOC_OBJ_GROUP";
static const char* HWLOC_OBJ_MISC_STR       = "HWLOC_OBJ_MISC";
static const char* HWLOC_OBJ_BRIDGE_STR     = "HWLOC_OBJ_BRIDGE";
static const char* HWLOC_OBJ_PCI_DEVICE_STR = "HWLOC_OBJ_PCI_DEVICE";
static const char* HWLOC_OBJ_OS_DEVICE_STR  = "HWLOC_OBJ_OS_DEVICE";

const char *str_for_type(hwloc_obj_type_t type);

#endif
