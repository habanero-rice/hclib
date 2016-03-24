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
