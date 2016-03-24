#ifndef HCLIB_CUDA_H
#define HCLIB_CUDA_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

#define CHECK_CUDA(func) { \
    const cudaError_t err = (func); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_pre_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(cuda_post_initialize);

int get_gpu_locale_id();
hclib::locale_t *get_closest_gpu_locale();

}

#endif
