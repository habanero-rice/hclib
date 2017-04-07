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
#include <iostream>
#include <fstream>
#include <assert.h>
#include "common.h"

#define BOOL_STR(cond) ((cond) ? "true" : "false")

static void write_hpt_tree(std::ofstream &output, hwloc_obj_t obj, int indent);

static void write_hpt_header(std::ofstream &output) {
    output << "<?xml version=\"1.0\"?>" << std::endl;
    output << "<!DOCTYPE HPT SYSTEM \"hpt.dtd\">" << std::endl;
    output << std::endl;
    output << "<HPT version=\"0.1\" info=\"Auto-generated using hwloc\">" <<
        std::endl;
}

static void write_hpt_footer(std::ofstream &output) {
    output << "</HPT>" << std::endl;
}

static bool is_memory_place(hwloc_obj_t obj) {
    return obj->type == HWLOC_OBJ_MACHINE;
}

static bool is_cache_place(hwloc_obj_t obj) {
    return obj->type == HWLOC_OBJ_CACHE;
}

static bool is_cpu_worker(hwloc_obj_t obj) {
    return obj->type == HWLOC_OBJ_PU;
}

static char *get_hwloc_info(hwloc_obj_t obj, const char *name) {
    for (int i = 0; i < obj->infos_count; i++) {
        if (strcmp(obj->infos[i].name, name) == 0) {
            return obj->infos[i].value;
        }
    }
    return NULL;
}

static bool is_nvgpu_place(hwloc_obj_t obj) {
    /*
     * CoProcType=CUDA, Backend=CUDA, GPUVendor=NVIDIA Corporation,
     *     GPUModel=Tesla K20c, CUDAGlobalMemorySize=4914752,
     *     CUDAL2CacheSize=1280, CUDAMultiProcessors=13, CUDACoresPerMP=192,
     *     CUDASharedMemorySizePerMP=48
     */
    if (obj->type != HWLOC_OBJ_OS_DEVICE || obj->infos_count == 0) return false;

    char *co_proc_type = get_hwloc_info(obj, "CoProcType");
    return co_proc_type != NULL && strcmp(co_proc_type, "CUDA") == 0;
}

static void recur_on_children(std::ofstream &output, hwloc_obj_t obj,
        int indent) {
    for (int i = 0; i < obj->arity; i++) {
        write_hpt_tree(output, obj->children[i], indent);
    }
}

static void write_indent(std::ofstream &output, int indent) {
    for (int i = 0; i < indent; i++) {
        output << "  ";
    }
}

static void write_hpt_tree(std::ofstream &output, hwloc_obj_t obj, int indent) {
#ifdef VERBOSE
    std::cout << "obj " << obj->os_index << " (type=" <<
        std::string(str_for_type(obj->type)) << ", memory=" <<
        is_memory_place(obj) << ", cache=" << is_cache_place(obj) <<
        ", worker=" << is_cpu_worker(obj) << ", nvgpu=" <<
        is_nvgpu_place(obj) << std::endl;
#endif

    if (is_memory_place(obj)) {
        write_indent(output, indent);
        output << "<place num=\"1\" type=\"mem\">" << std::endl;
        recur_on_children(output, obj, indent + 1);
        write_indent(output, indent);
        output << "</place>" << std::endl;
    } else if (is_cache_place(obj)) {
        write_indent(output, indent);
        output << "<place num=\"1\" type=\"cache\">" << std::endl;
        recur_on_children(output, obj, indent + 1);
        write_indent(output, indent);
        output << "</place>" << std::endl;
    } else if (is_cpu_worker(obj)) {
        assert(obj->arity == 0);
        write_indent(output, indent);
        output << "<worker num=\"1\"/>" << std::endl;
    } else if (is_nvgpu_place(obj)) {
        write_indent(output, indent);
        output << "<place num=\"1\" type=\"nvgpu\" info=\"" <<
            std::string(get_hwloc_info(obj, "GPUVendor")) << ", " <<
            std::string(get_hwloc_info(obj, "GPUModel")) << "\">" << std::endl;
        recur_on_children(output, obj, indent + 1);
        write_indent(output, indent);
        output << "</place>" << std::endl;
    } else {
        // Just continue down the tree, ignoring whatever the current node is
        recur_on_children(output, obj, indent);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << std::string(argv[0]) << " output-file" <<
            std::endl;
        return 1;
    }
    std::ofstream output;
    output.open(argv[1]);

    hwloc_topology_t topology;

    hwloc_topology_init(&topology);  // initialization
    hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
    hwloc_topology_load(topology);   // actual detection

    hwloc_obj_t root = hwloc_get_root_obj(topology);

    write_hpt_header(output);
    write_hpt_tree(output, root, 1);
    write_hpt_footer(output);

    output.close();

    hwloc_topology_destroy(topology);

    return 0;
}
