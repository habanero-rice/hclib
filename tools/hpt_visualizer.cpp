#include <hwloc.h>
#include <stdio.h>

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

static const char *str_for_type(hwloc_obj_type_t type) {
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

/*
 * os_index is unique in a given layer of the hierarchy (it seems), but
 * definitely not unique across all hwloc objects.
 */
static void print_hwloc_obj(hwloc_obj_t obj, int indent) {
    for (int i = 0; i < indent; i++) {
        printf("|  ");
    }
    printf("obj %lu (type=%s , depth=%d, %d child/children, parent=%lu)\n",
            obj->os_index, str_for_type(obj->type), obj->depth, obj->arity,
            obj->parent ? obj->parent->os_index : 0);

    for (int i = 0; i < obj->arity; i++) {
        print_hwloc_obj(obj->children[i], indent + 1);
    }
}

int main(void) {
    hwloc_topology_t topology;
    int nbcores;

    hwloc_topology_init(&topology);  // initialization
    hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
    hwloc_topology_load(topology);   // actual detection

    hwloc_obj_t root = hwloc_get_root_obj(topology);

    print_hwloc_obj(root, 0);

    hwloc_topology_destroy(topology);

    return 0;
}
